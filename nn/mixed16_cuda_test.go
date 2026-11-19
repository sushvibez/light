//go:build !nocuda && cgo

package nn_test

import (
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// ForwardMixed16 on Mixed16Input's rounding equals Forward on the F32
// input bit for bit (the same cast and GEMM), with and without a bias.
func TestForwardMixed16MatchesForward(t *testing.T) {
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skip(err)
	}
	rng := rand.New(rand.NewPCG(3, 4))
	in, out := 96, 40
	for _, bias := range []bool{false, true} {
		var l *nn.Linear
		if err := nn.WithWeightStorage16Mixed(dtype.BF16, func() (err error) {
			l, err = nn.NewLinearUninit(in, out, bias, dtype.F32, gpu)
			return
		}); err != nil {
			t.Fatal(err)
		}
		fill := func(dst *tensor.Tensor) {
			n := dst.NumElements()
			v := make([]float32, n)
			for i := range v {
				v[i] = float32(rng.NormFloat64())
			}
			src, err := tensor.New(v, dst.Shape().Dims(), device.CPU())
			if err != nil {
				t.Fatal(err)
			}
			if src, err = ops.Cast(src, dst.DType()); err != nil {
				t.Fatal(err)
			}
			if err := dst.CopyFrom(src); err != nil {
				t.Fatal(err)
			}
		}
		fill(l.W)
		if l.B != nil {
			fill(l.B)
		}
		xv := make([]float32, 2*7*in)
		for i := range xv {
			xv[i] = float32(rng.NormFloat64())
		}
		x, err := tensor.New(xv, []int{2, 7, in}, device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		if x, err = x.To(gpu); err != nil {
			t.Fatal(err)
		}
		want, err := l.Forward(x)
		if err != nil {
			t.Fatal(err)
		}
		xr, err := l.Mixed16Input(x)
		if err != nil || xr == nil {
			t.Fatalf("Mixed16Input: %v, %v", xr, err)
		}
		got, err := l.ForwardMixed16(xr)
		if err != nil {
			t.Fatal(err)
		}
		if !got.Shape().Equal(want.Shape()) {
			t.Fatalf("shape %v, want %v", got.Shape(), want.Shape())
		}
		g, _ := got.To(device.CPU())
		w, _ := want.To(device.CPU())
		gs, ws := tensor.MustSlice[float32](g), tensor.MustSlice[float32](w)
		for i := range gs {
			if gs[i] != ws[i] {
				t.Fatalf("bias=%v: element %d: %g, Forward %g", bias, i, gs[i], ws[i])
			}
		}
	}
}
