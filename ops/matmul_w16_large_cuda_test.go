//go:build !nocuda && cgo

package ops_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// F32 activations against BF16 weights at FLUX.1's linear shapes: CUDA
// against the CPU.
func TestMatMulW16FluxShapesParity(t *testing.T) {
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skip(err)
	}
	rng := rand.New(rand.NewPCG(9, 10))
	rnd := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	for _, c := range []struct{ M, K, N int }{{1280, 3072, 3072}, {1280, 3072, 18432}, {1280, 12288, 3072}, {1280, 15360, 3072}, {1, 3072, 9216}} {
		x, w := rnd(c.M*c.K), rnd(c.N*c.K)
		run := func(dev device.Device) []float32 {
			xt, _ := tensor.New(x, []int{c.M, c.K}, dev)
			wf, _ := tensor.New(w, []int{c.N, c.K}, dev)
			wb, err := ops.Cast(wf, dtype.BF16)
			if err != nil {
				t.Fatal(err)
			}
			y, err := ops.MatMulW16(xt, wb)
			if err != nil {
				t.Fatal(err)
			}
			cpu, _ := y.To(device.CPU())
			return tensor.MustSlice[float32](cpu)
		}
		a, b := run(device.CPU()), run(gpu)
		md, ma := 0.0, 0.0
		for i := range a {
			md = math.Max(md, math.Abs(float64(a[i]-b[i])))
			ma = math.Max(ma, math.Abs(float64(a[i])))
		}
		t.Logf("(%d x %d) · (%d x %d)ᵀ: max diff %.3g (max |y| %.3g)", c.M, c.K, c.N, c.K, md, ma)
		if md > 1e-2 {
			t.Errorf("differ by %.3g", md)
		}
	}
}
