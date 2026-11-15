package training

import (
	"math"
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/tensor"
)

func randParam(tb testing.TB, r *rand.Rand, dims ...int) *tensor.Tensor {
	n := 1
	for _, d := range dims {
		n *= d
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(r.NormFloat64())
	}
	t, err := tensor.New(data, dims, device.CPU())
	if err != nil {
		tb.Fatal(err)
	}
	return t
}

func cloneParams(tb testing.TB, ps []*tensor.Tensor) []*tensor.Tensor {
	out := make([]*tensor.Tensor, len(ps))
	for i, p := range ps {
		c, err := tensor.New(tensor.MustSlice[float32](p), p.Shape().Dims(), device.CPU())
		if err != nil {
			tb.Fatal(err)
		}
		out[i] = c
	}
	return out
}

// TestAdamWFusedCPUBitIdentical: the fused CPU update gives bit-identical
// parameters and moments to the tensor-op path (forceGeneric), with and
// without weight decay and with parameters that skip steps.
func TestAdamWFusedCPUBitIdentical(t *testing.T) {
	for _, wd := range []float32{0, 0.01} {
		r := rand.New(rand.NewPCG(1, uint64(wd*1000)))
		ps := []*tensor.Tensor{randParam(t, r, 33, 17), randParam(t, r, 5), randParam(t, r, 4, 3, 2)}
		qs := cloneParams(t, ps)
		opts := AdamWOptions{LR: 3e-3, WeightDecay: wd}
		fast, err := NewAdamW(ps, opts)
		if err != nil {
			t.Fatal(err)
		}
		ref, err := NewAdamW(qs, opts)
		if err != nil {
			t.Fatal(err)
		}
		ref.forceGeneric = true
		for step := range 12 {
			for i := range ps {
				if (step+i)%4 == 3 {
					ps[i].SetGrad(nil) // this parameter skips the step
					qs[i].SetGrad(nil)
					continue
				}
				g := randParam(t, r, ps[i].Shape().Dims()...)
				ps[i].SetGrad(g)
				gc, _ := tensor.New(tensor.MustSlice[float32](g), g.Shape().Dims(), device.CPU())
				qs[i].SetGrad(gc)
			}
			if err := fast.Step(); err != nil {
				t.Fatal(err)
			}
			if err := ref.Step(); err != nil {
				t.Fatal(err)
			}
		}
		check := func(what string, a, b *tensor.Tensor) {
			x, y := tensor.MustSlice[float32](a), tensor.MustSlice[float32](b)
			for j := range x {
				if math.Float32bits(x[j]) != math.Float32bits(y[j]) {
					t.Fatalf("wd=%g %s[%d]: fused %v, ops %v", wd, what, j, x[j], y[j])
				}
			}
		}
		for i := range ps {
			check("p", ps[i], qs[i])
			check("m", fast.m[i], ref.m[i])
			check("v", fast.v[i], ref.v[i])
		}
	}
}

func BenchmarkAdamWStep(b *testing.B) {
	r := rand.New(rand.NewPCG(2, 2))
	// ~4.2M parameters: two 1024x1024 matrices, a 2048x1024, biases.
	ps := []*tensor.Tensor{randParam(b, r, 1024, 1024), randParam(b, r, 1024), randParam(b, r, 2048, 1024), randParam(b, r, 2048), randParam(b, r, 1024, 1024)}
	opt, err := NewAdamW(ps, AdamWOptions{LR: 1e-3, WeightDecay: 0.01})
	if err != nil {
		b.Fatal(err)
	}
	for _, p := range ps {
		g, _ := tensor.RandN(dtype.F32, p.Shape().Dims(), device.CPU())
		p.SetGrad(g)
	}
	b.ResetTimer()
	for range b.N {
		if err := opt.Step(); err != nil {
			b.Fatal(err)
		}
	}
}
