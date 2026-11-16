package cpu

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/parallel"
	"light/internal/registry"
	"light/internal/simd"
	"light/tensor"
)

func init() {
	for _, dt := range []dtype.DType{dtype.F32, dtype.BF16, dtype.F16} {
		registry.Register(registry.Key{Op: "moe_gemm", Dev: device.KindCPU, DType: dt}, moeGemmKernel)
	}
}

func f32Reader(t *tensor.Tensor) (func(i int) float32, error) {
	switch t.DType() {
	case dtype.F32:
		s := asSlice[float32](t)
		return func(i int) float32 { return s[i] }, nil
	case dtype.BF16:
		s := asSlice[uint16](t)
		return func(i int) float32 { return dtype.BFloat16(s[i]).ToF32() }, nil
	case dtype.F16:
		s := asSlice[uint16](t)
		return func(i int) float32 { return dtype.Half(s[i]).ToF32() }, nil
	}
	return nil, fmt.Errorf("moe_gemm: unsupported dtype %v", t.DType())
}

func f32Writer(t *tensor.Tensor) func(i int, v float32) {
	switch t.DType() {
	case dtype.BF16:
		s := asSlice[uint16](t)
		return func(i int, v float32) { s[i] = uint16(dtype.BF16FromF32(v)) }
	case dtype.F16:
		s := asSlice[uint16](t)
		return func(i int, v float32) { s[i] = uint16(dtype.HalfFromF32(v)) }
	}
	s := asSlice[float32](t)
	return func(i int, v float32) { s[i] = v }
}

// moeGemmKernel is the reference for ops.MoEGemm (contract in
// kernels/cuda/src/moe.cu): for every sorted pair i with expert e_i and pair
// id p, out[p] = w(p) · input[row(p)] @ weights[e_i]ᵀ, F32 accumulation.
func moeGemmKernel(ctx *registry.Context) error {
	in, w, pairs, experts, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Inputs[3], ctx.Output
	var topkW []float32
	if len(ctx.Inputs) > 6 && ctx.Inputs[6] != nil {
		topkW = asSlice[float32](ctx.Inputs[6])
	}
	topk := ctx.Attrs["topk"].(int)
	rowPerPair := ctx.Attrs["row_per_pair"].(bool)
	rin, err := f32Reader(in)
	if err != nil {
		return err
	}
	rw, err := f32Reader(w)
	if err != nil {
		return err
	}
	store := f32Writer(out)
	N, K := w.Dim(1), w.Dim(2)
	pairS, expS := asSlice[int64](pairs), asSlice[int64](experts)
	sizeM := out.Dim(0)
	// rowOf is pair p's input row.
	rowOf := func(p int) int {
		if rowPerPair {
			return p
		}
		return p / topk
	}
	if in.DType() == dtype.F32 && w.DType() == dtype.F32 {
		// F32: every (pair, output) dot product in parallel, SIMD.
		xs, ws := asSlice[float32](in), asSlice[float32](w)
		parallel.For(len(pairS)*N, max(1, 16384/(K+1)), func(lo, hi int) {
			for idx := lo; idx < hi; idx++ {
				i, n := idx/N, idx%N
				p := int(pairS[i])
				if p < 0 || p >= sizeM {
					continue
				}
				r, e := rowOf(p), int(expS[i])
				acc := simd.Dot(xs[r*K:(r+1)*K], ws[(e*N+n)*K:(e*N+n+1)*K])
				if topkW != nil {
					acc *= topkW[p]
				}
				store(p*N+n, acc)
			}
		})
		return nil
	}
	row := make([]float32, K)
	for i, p64 := range pairS {
		p := int(p64)
		if p < 0 || p >= sizeM {
			continue
		}
		e := int(expS[i])
		r := rowOf(p)
		for k := range K {
			row[k] = rin(r*K + k)
		}
		scale := float32(1)
		if topkW != nil {
			scale = topkW[p]
		}
		wBase := e * N * K
		for n := range N {
			var acc float32
			for k := range K {
				acc += row[k] * rw(wBase+n*K+k)
			}
			store(p*N+n, acc*scale)
		}
	}
	return nil
}
