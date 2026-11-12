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

// The CUDA decode kernels (append at a device-side position, split-K
// attention with GQA over the valid prefix, increment) must match the
// composed CPU path, for F32 and BF16, short and long caches (several
// split-K blocks), and the fallback path must match itself on CUDA.
func TestDecodeAttentionCUDAParity(t *testing.T) {
	gpu := gpuOrSkip(t)
	rng := rand.New(rand.NewPCG(5, 9))
	for _, tc := range []struct {
		name          string
		B, NH, KH, HD int
		capacity, n   int
		window        int
		dt            dtype.DType
	}{
		{"f32 short", 2, 8, 2, 64, 256, 7, 0, dtype.F32},
		{"bf16 llama-ish", 1, 32, 8, 64, 512, 300, 0, dtype.BF16},
		{"bf16 qwen-ish", 1, 16, 2, 128, 512, 129, 0, dtype.BF16},
		{"f32 hd256 g1", 1, 4, 4, 256, 256, 200, 0, dtype.F32},
		{"bf16 gemma-ish window", 1, 4, 1, 256, 1024, 700, 512, dtype.BF16}, // window starts mid-block (188)
		{"f32 window short", 2, 8, 2, 64, 256, 100, 33, dtype.F32},
		{"f32 window wider than n", 1, 8, 2, 64, 256, 20, 512, dtype.F32},
		{"bf16 gemma4 global hd512", 1, 8, 2, 512, 1024, 650, 0, dtype.BF16},
	} {
		tol := 1e-4
		if tc.dt == dtype.BF16 {
			tol = 3e-2
		}
		kbuf := randTensor(t, rng, []int{tc.B, tc.KH, tc.capacity, tc.HD}, 1, device.CPU(), tc.dt)
		vbuf := randTensor(t, rng, []int{tc.B, tc.KH, tc.capacity, tc.HD}, 1, device.CPU(), tc.dt)
		q := randTensor(t, rng, []int{tc.B, tc.NH, 1, tc.HD}, 1, device.CPU(), tc.dt)
		kNew := randTensor(t, rng, []int{tc.B, tc.KH, 1, tc.HD}, 1, device.CPU(), tc.dt)
		vNew := randTensor(t, rng, []int{tc.B, tc.KH, 1, tc.HD}, 1, device.CPU(), tc.dt)
		scale := 1 / math.Sqrt(float64(tc.HD))
		run := func(dev device.Device) ([]float32, []float32, int64) {
			kb, _ := kbuf.To(dev)
			vb, _ := vbuf.To(dev)
			qd, _ := q.To(dev)
			kn, _ := kNew.To(dev)
			vn, _ := vNew.To(dev)
			pos, err := tensor.New([]int64{int64(tc.n - 1)}, []int{1}, dev)
			if err != nil {
				t.Fatal(err)
			}
			if err := ops.KVAppend(kb, vb, kn, vn, pos); err != nil {
				t.Fatalf("%s on %v: append: %v", tc.name, dev, err)
			}
			out, err := ops.DecodeAttention(qd, kb, vb, pos, scale, tc.window)
			if err != nil {
				t.Fatalf("%s on %v: attention: %v", tc.name, dev, err)
			}
			if err := ops.IncrementI64(pos); err != nil {
				t.Fatal(err)
			}
			pc, _ := pos.To(device.CPU())
			row, _ := kb.Narrow(2, tc.n-1, 1)
			row, _ = row.Contiguous()
			return f32(t, out), f32(t, row), tensor.MustSlice[int64](pc)[0]
		}
		cpuOut, cpuRow, cpuPos := run(device.CPU())
		gpuOut, gpuRow, gpuPos := run(gpu)
		if cpuPos != int64(tc.n) || gpuPos != int64(tc.n) {
			t.Fatalf("%s: position after increment: cpu %d gpu %d, want %d", tc.name, cpuPos, gpuPos, tc.n)
		}
		want := f32(t, kNew)
		for i := range want {
			if cpuRow[i] != want[i] || gpuRow[i] != want[i] {
				t.Fatalf("%s: appended key row [%d]: cpu %v gpu %v want %v", tc.name, i, cpuRow[i], gpuRow[i], want[i])
			}
		}
		for i := range cpuOut {
			if d := math.Abs(float64(cpuOut[i] - gpuOut[i])); d > tol*math.Max(1, math.Abs(float64(cpuOut[i]))) {
				t.Fatalf("%s: out[%d] cpu %v gpu %v (tol %g)", tc.name, i, cpuOut[i], gpuOut[i], tol)
			}
		}
	}
}
