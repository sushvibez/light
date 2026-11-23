//go:build !nocuda && cgo

package ops_test

import (
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// TestCUDAConv2dPlanReuse runs convolutions of several shapes, strides and
// dtypes interleaved and repeated, each against the CPU: the cuDNN
// descriptors and algorithm are cached per shape and the workspace comes
// from the caching pool, so a stale plan or a workspace handed out twice
// would show here.
func TestCUDAConv2dPlanReuse(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	rng := rand.New(rand.NewPCG(7, 9))
	rnd := func(shape ...int) *tensor.Tensor {
		n := 1
		for _, d := range shape {
			n *= d
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64()) * 0.2
		}
		x, err := tensor.New(v, shape, cpu)
		if err != nil {
			t.Fatal(err)
		}
		return x
	}
	type conv struct {
		n, cin, h, w, cout, k int
		stride, pad           int
		bias                  bool
	}
	cases := []conv{
		{1, 16, 32, 32, 16, 3, 1, 1, true},
		{1, 16, 32, 32, 32, 3, 2, 1, false},
		{2, 8, 24, 40, 12, 1, 1, 0, true},
		{1, 16, 32, 32, 16, 3, 1, 0, true}, // the first shape with another padding
		{1, 32, 64, 64, 32, 3, 1, 1, true},
	}
	to := func(x *tensor.Tensor, dt dtype.DType) *tensor.Tensor {
		g, err := x.To(gpu)
		if err != nil {
			t.Fatal(err)
		}
		if dt != dtype.F32 {
			if g, err = ops.Cast(g, dt); err != nil {
				t.Fatal(err)
			}
		}
		return g
	}
	for round := range 3 {
		for ci, c := range cases {
			x, w := rnd(c.n, c.cin, c.h, c.w), rnd(c.cout, c.cin, c.k, c.k)
			var b *tensor.Tensor
			if c.bias {
				b = rnd(c.cout)
			}
			st, pd := [2]int{c.stride, c.stride}, [2]int{c.pad, c.pad}
			want, err := ops.Conv2d(x, w, b, st, pd)
			if err != nil {
				t.Fatal(err)
			}
			for _, dt := range []dtype.DType{dtype.F32, dtype.BF16} {
				var bg *tensor.Tensor
				if b != nil {
					bg = to(b, dt)
				}
				got, err := ops.Conv2d(to(x, dt), to(w, dt), bg, st, pd)
				if err != nil {
					t.Fatalf("round %d case %d %v: %v", round, ci, dt, err)
				}
				if dt != dtype.F32 {
					if got, err = ops.Cast(got, dtype.F32); err != nil {
						t.Fatal(err)
					}
				}
				tol := 1e-4
				if dt == dtype.BF16 {
					tol = 0.08 // 8-bit mantissa inputs and output over up to 288 products
				}
				if d := maxAbsDiff(t, want, got); d > tol {
					t.Errorf("round %d case %d %v: max diff %g > %g", round, ci, dt, d, tol)
				}
			}
		}
	}
}
