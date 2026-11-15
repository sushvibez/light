package autograd_test

import (
	"fmt"
	"math"
	"testing"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// matmulGradsRef is the original backward: both transposes materialised,
// both gradients always computed.
func matmulGradsRef(a, b, g *tensor.Tensor) (ga, gb *tensor.Tensor, err error) {
	r := a.Rank()
	bT, _ := b.Transpose(r-2, r-1)
	aT, _ := a.Transpose(r-2, r-1)
	bTc, _ := bT.Contiguous()
	aTc, _ := aT.Contiguous()
	if ga, err = ops.MatMul(g, bTc); err != nil {
		return
	}
	gb, err = ops.MatMul(aTc, g)
	return
}

func randT(tb testing.TB, dims ...int) *tensor.Tensor {
	x, err := tensor.RandN(dtype.F32, dims, device.CPU())
	if err != nil {
		tb.Fatal(err)
	}
	return x
}

// maxRelDiff is max|x−y| relative to max|y| (float32 summation-order
// differences show up on near-cancelling entries, so per-entry relative
// error is not meaningful).
func maxRelDiff(tb testing.TB, x, y *tensor.Tensor) float64 {
	xs, ys := tensor.MustSlice[float32](x), tensor.MustSlice[float32](y)
	var worst, scale float64
	for i := range xs {
		worst = math.Max(worst, math.Abs(float64(xs[i]-ys[i])))
		scale = math.Max(scale, math.Abs(float64(ys[i])))
	}
	return worst / math.Max(scale, 1e-30)
}

// TestMatMulBackwardSkipsAndMatchesRef: a gradient is returned only for an
// operand that needs one, and equals the materialised-transpose reference.
func TestMatMulBackwardSkipsAndMatchesRef(t *testing.T) {
	for _, dims := range [][2][]int{
		{{7, 5}, {5, 3}},
		{{64, 96}, {96, 80}},
		{{4, 33, 17}, {4, 17, 29}},
	} {
		for _, need := range [][2]bool{{true, true}, {true, false}, {false, true}} {
			a, b := randT(t, dims[0]...), randT(t, dims[1]...)
			a.SetRequiresGrad(need[0])
			b.SetRequiresGrad(need[1])
			ctx := &autograd.Context{}
			outs, err := autograd.MatMulFunction{}.Forward(ctx, []*tensor.Tensor{a, b})
			if err != nil {
				t.Fatal(err)
			}
			g := randT(t, outs[0].Shape().Dims()...)
			grads, err := autograd.MatMulFunction{}.Backward(ctx, []*tensor.Tensor{g})
			if err != nil {
				t.Fatal(err)
			}
			refA, refB, err := matmulGradsRef(a, b, g)
			if err != nil {
				t.Fatal(err)
			}
			for i, ref := range []*tensor.Tensor{refA, refB} {
				if !need[i] {
					if grads[i] != nil {
						t.Errorf("%v need=%v: grad %d computed for an operand without grad", dims, need, i)
					}
					continue
				}
				if grads[i] == nil {
					t.Fatalf("%v need=%v: grad %d missing", dims, need, i)
				}
				if d := maxRelDiff(t, grads[i], ref); d > 1e-5 {
					t.Errorf("%v need=%v: grad %d differs from reference by %.3g", dims, need, i, d)
				} else if d != 0 {
					t.Logf("%v grad %d: max relative diff %.3g (summation order)", dims, i, d)
				}
			}
		}
	}
}

func BenchmarkMatMulBackward(b *testing.B) {
	for _, c := range []struct {
		name   string
		da, db []int
		needA  bool
	}{
		{"2d_512", []int{512, 512}, []int{512, 512}, true},
		{"2d_512_wonly", []int{512, 512}, []int{512, 512}, false},
		{"2d_256x1024x1024", []int{256, 1024}, []int{1024, 1024}, true},
		{"3d_8x128x64x128", []int{8, 128, 64}, []int{8, 64, 128}, true},
	} {
		x, w := randT(b, c.da...), randT(b, c.db...)
		x.SetRequiresGrad(c.needA)
		w.SetRequiresGrad(true)
		ctx := &autograd.Context{}
		outs, err := autograd.MatMulFunction{}.Forward(ctx, []*tensor.Tensor{x, w})
		if err != nil {
			b.Fatal(err)
		}
		g := randT(b, outs[0].Shape().Dims()...)
		b.Run(fmt.Sprintf("new/%s", c.name), func(b *testing.B) {
			for range b.N {
				if _, err := (autograd.MatMulFunction{}).Backward(ctx, []*tensor.Tensor{g}); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(fmt.Sprintf("ref/%s", c.name), func(b *testing.B) {
			for range b.N {
				if _, _, err := matmulGradsRef(x, w, g); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
