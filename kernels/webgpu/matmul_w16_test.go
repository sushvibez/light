//go:build !nowebgpu

package webgpu_test

import (
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// ops.MatMulW16 on WebGPU (gemv16 for M <= 8, matmul16 above) against the
// CPU kernel for BF16 and F16 weights, plus the F32 ↔ 16-bit casts.
func TestMatMulW16(t *testing.T) {
	g, err := device.WebGPU(0)
	if err != nil {
		t.Skip(err)
	}
	cpu := device.CPU()
	rng := rand.New(rand.NewPCG(5, 6))
	randT := func(_ device.Device, dims ...int) *tensor.Tensor {
		n := 1
		for _, d := range dims {
			n *= d
		}
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		c, err := tensor.New(x, dims, cpu)
		if err != nil {
			t.Fatal(err)
		}
		return c
	}
	for _, dt := range []dtype.DType{dtype.BF16, dtype.F16} {
		for _, M := range []int{1, 5, 8, 9, 37} {
			const N, K = 129, 258
			xc := randT(cpu, M, K)
			w32 := randT(cpu, N, K)
			wc, err := ops.Cast(w32, dt)
			if err != nil {
				t.Fatal(err)
			}
			want, err := ops.MatMulW16(xc, wc)
			if err != nil {
				t.Fatal(err)
			}
			xg, _ := xc.To(g)
			// Narrow on the GPU to cover cast16 too: it must equal the CPU's.
			w32g, _ := w32.To(g)
			wg, err := ops.Cast(w32g, dt)
			if err != nil {
				t.Fatal(err)
			}
			wgc, _ := wg.To(cpu)
			if a, b := wc.Storage().CPUBytes(), wgc.Storage().CPUBytes(); string(a) != string(b) {
				t.Fatalf("%v: GPU narrowing differs from the CPU's", dt)
			}
			back, err := ops.Cast(wg, dtype.F32)
			if err != nil {
				t.Fatal(err)
			}
			backC, _ := back.To(cpu)
			wide, _ := ops.Cast(wc, dtype.F32)
			if d := maxDiff(t, wide, backC); d != 0 {
				t.Fatalf("%v widening differs by %g", dt, d)
			}
			got, err := ops.MatMulW16(xg, wg)
			if err != nil {
				t.Fatal(err)
			}
			gc, _ := got.To(cpu)
			if d := maxDiff(t, want, gc); d > 1e-3 {
				t.Errorf("%v M=%d: max diff %g", dt, M, d)
			}
		}
	}
}

func maxDiff(t *testing.T, a, b *tensor.Tensor) float64 {
	t.Helper()
	av, bv := tensor.MustSlice[float32](a), tensor.MustSlice[float32](b)
	d := 0.0
	for i := range av {
		d = math.Max(d, math.Abs(float64(av[i]-bv[i])))
	}
	return d
}

// Decode-shaped GEMV, F32 weight vs BF16 storage (skipped with -short).
func TestMatMulW16Speed(t *testing.T) {
	if testing.Short() {
		t.Skip()
	}
	g, err := device.WebGPU(0)
	if err != nil {
		t.Skip(err)
	}
	const N, K = 8192, 3072
	w32, _ := tensor.Zeros(dtype.F32, []int{N, K}, g)
	w16, _ := ops.Cast(w32, dtype.BF16)
	x, _ := tensor.Zeros(dtype.F32, []int{1, K}, g)
	wT, _ := w32.Transpose(0, 1)
	run := func(name string, f func() (*tensor.Tensor, error)) {
		for range 3 {
			f()
		}
		x.To(device.CPU())
		const it = 50
		t0 := time.Now()
		var y *tensor.Tensor
		for range it {
			y, _ = f()
		}
		y.To(device.CPU())
		d := time.Since(t0) / it
		t.Logf("%-5s %v/call", name, d)
	}
	run("f32", func() (*tensor.Tensor, error) { return ops.MatMul(x, wT) })
	run("bf16", func() (*tensor.Tensor, error) { return ops.MatMulW16(x, w16) })
}
