package cpu

import (
	"math"

	"light/dtype"
	"light/tensor"
)

// matmulHalfFast runs a matmul whose B operand is a 16-bit (BF16 / F16)
// weight in nn.Linear's x · Wᵀ layout (b[k, j] at k + j·K) through the
// quantized path's machinery: weight rows are decoded to F32 (AVX2 for
// BF16), multiplied with 4 × 3 register tiles, accumulated in F32 and
// rounded once to the output dtype — the same contract as the reference
// loops, with a different summation order. A may be F32 / BF16 / F16;
// roundA rounds F32 activations to the weight's dtype first (the mixed
// path's semantics). It reports false for any other layout.
func matmulHalfFast(a, b, out *tensor.Tensor, roundA bool) bool {
	M, K, N := a.Dim(0), a.Dim(1), b.Dim(1)
	bStr := b.Strides().Slice()
	if M == 0 || N == 0 || K%8 != 0 || bStr[0] != 1 || (bStr[1] != K && N > 1) {
		return false
	}
	bdt := b.DType()
	if bdt != dtype.BF16 && bdt != dtype.F16 {
		return false
	}
	ra, err := readF32(a)
	if err != nil {
		return false
	}
	x := make([]float32, M*K)
	for i := range M {
		for k := range K {
			v := ra(i, k)
			if roundA {
				if bdt == dtype.BF16 {
					v = dtype.BF16FromF32(v).ToF32()
				} else {
					v = dtype.HalfFromF32(v).ToF32()
				}
			}
			x[i*K+k] = v
		}
	}
	w := asSlice[uint16](b)[b.Offset():]
	y := make([]float32, M*N)
	rowsTimesX(x, y, M, K, N, func(n int, dst []float32) {
		src := w[n*K : n*K+K]
		switch {
		case bdt == dtype.BF16 && decodeAVX2:
			bf16RowAVX2(&src[0], &dst[0], K)
		case bdt == dtype.BF16:
			for k, v := range src {
				dst[k] = math.Float32frombits(uint32(v) << 16)
			}
		default:
			for k, v := range src {
				dst[k] = dtype.Half(v).ToF32()
			}
		}
	})
	switch out.DType() {
	case dtype.F32:
		copy(asSlice[float32](out), y)
	case dtype.BF16:
		o := asSlice[uint16](out)
		for i, v := range y {
			o[i] = uint16(dtype.BF16FromF32(v))
		}
	case dtype.F16:
		o := asSlice[uint16](out)
		for i, v := range y {
			o[i] = uint16(dtype.HalfFromF32(v))
		}
	default:
		return false
	}
	return true
}
