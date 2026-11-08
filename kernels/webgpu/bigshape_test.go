//go:build !nowebgpu

package webgpu_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// GPT-2-sized ops (the LM head over a 50257 × 768 tied embedding is a
// 154 MB weight; the token lookup gathers rows of the same table).
func TestGPT2SizedOps(t *testing.T) {
	g, err := device.WebGPU(0)
	if err != nil {
		t.Skip(err)
	}
	cpu := device.CPU()
	rng := rand.New(rand.NewPCG(3, 4))
	randF := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64()) * 0.1
		}
		return x
	}
	const V, D, T = 50257, 768, 5
	wData, xData := randF(V*D), randF(T*D)
	both := func(data []float32, dims ...int) (*tensor.Tensor, *tensor.Tensor) {
		c, err := tensor.New(data, dims, cpu)
		if err != nil {
			t.Fatal(err)
		}
		w, err := c.To(g)
		if err != nil {
			t.Fatal(err)
		}
		return c, w
	}
	cmp := func(name string, c, w *tensor.Tensor, tol float64) {
		t.Helper()
		wc, err := w.To(cpu)
		if err != nil {
			t.Fatal(err)
		}
		a, b := tensor.MustSlice[float32](c), tensor.MustSlice[float32](wc)
		var md float64
		bad := -1
		for i := range a {
			d := math.Abs(float64(a[i] - b[i]))
			if d > md || math.IsNaN(d) {
				md = d
				if bad < 0 && (d > tol || math.IsNaN(d)) {
					bad = i
				}
			}
		}
		if bad >= 0 {
			t.Errorf("%s: max |diff| %.3g; first bad index %d of %d (cpu %v, webgpu %v)", name, md, bad, len(a), a[bad], b[bad])
		} else {
			t.Logf("%s: max |diff| %.3g over %d values", name, md, len(a))
		}
	}
	wc, wg := both(wData, V, D)
	xc, xg := both(xData, T, D)
	// LM head: x · Wᵀ via a transposed view, as models do.
	wcT, _ := wc.Transpose(0, 1)
	wgT, _ := wg.Transpose(0, 1)
	lc, err := ops.MatMul(xc, wcT)
	if err != nil {
		t.Fatal(err)
	}
	lg, err := ops.MatMul(xg, wgT)
	if err != nil {
		t.Fatal(err)
	}
	cmp("lm head matmul", lc, lg, 1e-3)
	// Embedding lookup.
	ids := []int64{464, 3139, 286, 4881, 318, 50256, 0}
	ic, _ := tensor.New(ids, []int{len(ids)}, cpu)
	ig, _ := ic.To(g)
	ec, err := ops.IndexSelect(wc, ic)
	if err != nil {
		t.Fatal(err)
	}
	eg, err := ops.IndexSelect(wg, ig)
	if err != nil {
		t.Fatal(err)
	}
	cmp("embedding index_select", ec, eg, 0)
	_ = dtype.F32
}

// tanh and GELU stay finite for large arguments (the builtin tanh returns
// NaN past ~±44 on some D3D12 drivers; GPT-2's MLP reaches that range).
func TestTanhGELULargeInputs(t *testing.T) {
	g, err := device.WebGPU(0)
	if err != nil {
		t.Skip(err)
	}
	xs := []float32{-200, -90, -45, -20, -3, -0.5, 0, 0.5, 3, 20, 45, 90, 200}
	x, _ := tensor.New(xs, []int{len(xs)}, device.CPU())
	xg, _ := x.To(g)
	for _, op := range []struct {
		name string
		f    func(*tensor.Tensor) (*tensor.Tensor, error)
	}{{"tanh", ops.Tanh}, {"gelu", ops.Gelu}} {
		yc, err := op.f(x)
		if err != nil {
			t.Fatal(err)
		}
		yg, err := op.f(xg)
		if err != nil {
			t.Fatal(err)
		}
		yg, _ = yg.To(device.CPU())
		a, b := tensor.MustSlice[float32](yc), tensor.MustSlice[float32](yg)
		for i := range a {
			if d := math.Abs(float64(a[i] - b[i])); !(d <= 1e-4*max(1, math.Abs(float64(a[i])))) {
				t.Errorf("%s(%g): cpu %g, webgpu %g", op.name, xs[i], a[i], b[i])
			}
		}
	}
}
