//go:build !nocuda && cgo

package ops_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
)

// The batched GEMM must read KV-cache views in place: a (B, KH, cap, HD)
// buffer narrowed to len positions and folded to (B·KH, len, HD) keeps
// the buffer's strides, and its transpose is the kᵀ operand. CPU and CUDA
// must agree with the contiguous computation.
func TestMatMulBatchedStridedViewsCUDA(t *testing.T) {
	gpu := gpuOrSkip(t)
	rng := rand.New(rand.NewPCG(7, 11))
	B, KH, capacity, L, HD, g := 2, 3, 16, 7, 8, 2
	for _, dt := range []dtype.DType{dtype.F32, dtype.BF16} {
		tol := 1e-4
		if dt == dtype.BF16 {
			tol = 2e-2
		}
		kbuf := randTensor(t, rng, []int{B, KH, capacity, HD}, 1, device.CPU(), dt)
		q := randTensor(t, rng, []int{B * KH, g, HD}, 1, device.CPU(), dt)
		attn := randTensor(t, rng, []int{B * KH, g, L}, 1, device.CPU(), dt)
		run := func(dev device.Device) (scores, ctx []float32) {
			kb, _ := kbuf.To(dev)
			qd, _ := q.To(dev)
			ad, _ := attn.To(dev)
			kn, err := kb.Narrow(2, 0, L)
			if err != nil {
				t.Fatal(err)
			}
			st := kn.Strides().Slice()
			k, err := kn.AsStrided([]int{B * KH, L, HD}, []int{st[1], st[2], st[3]}, 0)
			if err != nil {
				t.Fatal(err)
			}
			kT, _ := k.Transpose(1, 2)
			s, err := ops.MatMul(qd, kT) // (B·KH, g, L)
			if err != nil {
				t.Fatal(err)
			}
			c, err := ops.MatMul(ad, k) // (B·KH, g, HD)
			if err != nil {
				t.Fatal(err)
			}
			return f32(t, s), f32(t, c)
		}
		// Reference: contiguous copies.
		kc, _ := kbuf.Narrow(2, 0, L)
		kcc, _ := kc.Contiguous()
		kr, _ := kcc.Reshape(B*KH, L, HD)
		krT, _ := kr.Transpose(1, 2)
		krTc, _ := krT.Contiguous()
		refS, _ := ops.MatMul(q, krTc)
		refC, _ := ops.MatMul(attn, kr)
		for _, dev := range []device.Device{device.CPU(), gpu} {
			s, c := run(dev)
			check := func(name string, got, want []float32) {
				for i := range got {
					if d := math.Abs(float64(got[i] - want[i])); d > tol*math.Max(1, math.Abs(float64(want[i]))) {
						t.Fatalf("%v %s on %v: [%d] = %v, want %v", dt, name, dev, i, got[i], want[i])
					}
				}
			}
			check("q·kᵀ", s, f32(t, refS))
			check("attn·v", c, f32(t, refC))
		}
	}
}
