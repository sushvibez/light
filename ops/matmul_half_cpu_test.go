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

// 16-bit matmuls with nn.Linear's x · Wᵀ layout on the CPU (the decode +
// tiled path): BF16 × BF16, F16 × F16 and F32 × BF16 (mixed: activations
// rounded to BF16 first) against float64 sums of the rounded operands,
// within one rounding of the output.
func TestMatMulHalfLinearLayoutCPU(t *testing.T) {
	cpu := device.CPU()
	rng := rand.New(rand.NewPCG(13, 14))
	const M, K, N = 9, 136, 37
	xs, ws := make([]float32, M*K), make([]float32, N*K)
	for i := range xs {
		xs[i] = float32(rng.NormFloat64())
	}
	for i := range ws {
		ws[i] = float32(rng.NormFloat64()) * 0.1
	}
	for _, c := range []struct {
		name     string
		adt, wdt dtype.DType
		mixed    bool
	}{{"bf16", dtype.BF16, dtype.BF16, false}, {"f16", dtype.F16, dtype.F16, false}, {"f32xbf16", dtype.F32, dtype.BF16, true}} {
		x, _ := tensor.New(xs, []int{M, K}, cpu)
		w, _ := tensor.New(ws, []int{N, K}, cpu)
		if c.adt != dtype.F32 {
			x, _ = ops.Cast(x, c.adt)
		}
		w, _ = ops.Cast(w, c.wdt)
		wT, _ := w.Transpose(0, 1)
		var y *tensor.Tensor
		var err error
		if c.mixed {
			y, err = ops.MatMulMixed(x, wT)
		} else {
			y, err = ops.MatMul(x, wT)
		}
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		// Reference on the rounded operands.
		x16, _ := ops.Cast(x, c.wdt)
		xr, _ := ops.Cast(x16, dtype.F32)
		wr, _ := ops.Cast(w, dtype.F32)
		y32, _ := ops.Cast(y, dtype.F32)
		xa, wa, ya := tensor.MustSlice[float32](xr), tensor.MustSlice[float32](wr), tensor.MustSlice[float32](y32)
		for m := range M {
			for n := range N {
				var want float64
				for k := range K {
					want += float64(xa[m*K+k]) * float64(wa[n*K+k])
				}
				tol := 1e-5 + math.Abs(want)*1e-5
				if y.DType() != dtype.F32 {
					tol = math.Abs(want)*1.0/128 + 1e-4 // one BF16 / F16 rounding of the output
				}
				if d := math.Abs(float64(ya[m*N+n]) - want); d > tol {
					t.Fatalf("%s [%d,%d] = %g, want %g", c.name, m, n, ya[m*N+n], want)
				}
			}
		}
	}
}
