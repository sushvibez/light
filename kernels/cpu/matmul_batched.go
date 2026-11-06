package cpu

import (
	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func init() {
	reg := func(dt dtype.DType, k registry.Kernel) {
		registry.Register(registry.Key{Op: "matmul_batched", Dev: device.KindCPU, DType: dt}, k)
	}
	reg(dtype.F32, batchedMatmulKernel[float32])
	reg(dtype.F64, batchedMatmulKernel[float64])
	reg(dtype.F16, batchedMatmulHalfKernel(func(u uint16) float32 { return dtype.Half(u).ToF32() }, func(f float32) uint16 { return uint16(dtype.HalfFromF32(f)) }))
	reg(dtype.BF16, batchedMatmulHalfKernel(func(u uint16) float32 { return dtype.BFloat16(u).ToF32() }, func(f float32) uint16 { return uint16(dtype.BF16FromF32(f)) }))
}

// batchedMatmulKernel implements (B, M, K) × (B, K, N) → (B, M, N) as B
// independent 2-D naive matmuls. Used both as the CPU reference and as the
// CUDA fallback path when cuBLAS batched gemm isn't available.
func batchedMatmulKernel[T dtype.Numeric](ctx *registry.Context) error {
	a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	aS := asSlice[T](a)
	bS := asSlice[T](b)
	outS := asSlice[T](out)

	B := a.Dim(0)
	M := a.Dim(1)
	K := a.Dim(2)
	N := b.Dim(2)

	// Stride-aware so transposed views (kᵀ in attention) and narrowed
	// batches are read in place, as the 2-D kernel and the cuBLAS launcher do.
	aStr, bStr := a.Strides().Slice(), b.Strides().Slice()
	outStride := M * N

	for batch := range B {
		aBase := batch*aStr[0] + a.Offset()
		bBase := batch*bStr[0] + b.Offset()
		oBase := batch * outStride
		for i := range M {
			for j := range N {
				var sum T
				for kk := range K {
					sum += aS[aBase+i*aStr[1]+kk*aStr[2]] * bS[bBase+kk*bStr[1]+j*bStr[2]]
				}
				outS[oBase+i*N+j] = sum
			}
		}
	}
	return nil
}

// batchedMatmulHalfKernel is the F16 / BF16 batched reference: 16-bit
// storage, F32 products and accumulation, one rounding on store (the
// cuBLAS GemmStridedBatchedEx contract).
func batchedMatmulHalfKernel(toF32 func(uint16) float32, fromF32 func(float32) uint16) registry.Kernel {
	return func(ctx *registry.Context) error {
		a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
		aS, bS, outS := asSlice[uint16](a), asSlice[uint16](b), asSlice[uint16](out)
		B, M, K, N := a.Dim(0), a.Dim(1), a.Dim(2), b.Dim(2)
		aStr, bStr := a.Strides().Slice(), b.Strides().Slice()
		for batch := range B {
			aBase := batch*aStr[0] + a.Offset()
			bBase := batch*bStr[0] + b.Offset()
			oBase := batch * M * N
			for i := range M {
				for j := range N {
					var sum float32
					for kk := range K {
						sum += toF32(aS[aBase+i*aStr[1]+kk*aStr[2]]) * toF32(bS[bBase+kk*bStr[1]+j*bStr[2]])
					}
					outS[oBase+i*N+j] = fromF32(sum)
				}
			}
		}
		return nil
	}
}
