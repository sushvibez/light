//go:build !nocuda && cgo

package cuda

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
	"light/tensor"
)

func init() {
	registry.Register(registry.Key{Op: "matmul_batched", Dev: device.KindCUDA, DType: dtype.F32}, batchedMatmulF32CUDA)
	registry.Register(registry.Key{Op: "matmul_batched", Dev: device.KindCUDA, DType: dtype.BF16}, batchedMatmulBF16CUDA)
}

func batchedMatmulBF16CUDA(ctx *registry.Context) error {
	a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	a, b, transB, err := batchedMatmulPrep(a, b)
	if err != nil {
		return err
	}
	ord := out.Device().Ordinal()
	if err := cudart.SetDevice(ord); err != nil {
		return err
	}
	h, err := cudart.GetBlasHandle(ord)
	if err != nil {
		return err
	}
	B := a.Dim(0)
	M := a.Dim(1)
	K := a.Dim(2)
	N := b.Dim(2)
	gemm := cudart.GemmExBF16BatchedRowMajor
	if transB {
		gemm = cudart.GemmExBF16BatchedRowMajorTransB
	}
	return gemm(h, M, N, K, a.Storage().CUDAPtr(), b.Storage().CUDAPtr(), out.Storage().CUDAPtr(), B)
}

func batchedMatmulF32CUDA(ctx *registry.Context) error {
	a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	a, b, transB, err := batchedMatmulPrep(a, b)
	if err != nil {
		return err
	}
	ord := out.Device().Ordinal()
	if err := cudart.SetDevice(ord); err != nil {
		return err
	}
	h, err := cudart.GetBlasHandle(ord)
	if err != nil {
		return err
	}
	B := a.Dim(0)
	M := a.Dim(1)
	K := a.Dim(2)
	N := b.Dim(2)
	gemm := cudart.SgemmStridedBatchedRowMajor
	if transB {
		gemm = cudart.SgemmStridedBatchedRowMajorTransB
	}
	return gemm(h, M, N, K, a.Storage().CUDAPtr(), b.Storage().CUDAPtr(), out.Storage().CUDAPtr(), B)
}

// batchedMatmulPrep makes both operands cuBLAS-ready. A transpose view of a
// contiguous (count, N, K) block as b (attention's kᵀ against the key cache)
// is passed through with transB set, so the cache is never copied.
func batchedMatmulPrep(a, b *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, bool, error) {
	if a.Rank() != 3 || b.Rank() != 3 {
		return nil, nil, false, fmt.Errorf("cuda matmul_batched: need rank-3, got %d and %d", a.Rank(), b.Rank())
	}
	var err error
	if !a.IsContiguous() || a.Offset() != 0 {
		a, err = a.Contiguous()
		if err != nil {
			return nil, nil, false, err
		}
	}
	if isTransposedBlock(b) {
		return a, b, true, nil
	}
	if !b.IsContiguous() || b.Offset() != 0 {
		b, err = b.Contiguous()
		if err != nil {
			return nil, nil, false, err
		}
	}
	return a, b, false, nil
}

// isTransposedBlock reports whether b (count, K, N) is the transpose view of
// a contiguous (count, N, K) block at offset 0: strides (N·K, 1, K).
func isTransposedBlock(b *tensor.Tensor) bool {
	if b.Offset() != 0 || b.IsContiguous() {
		return false
	}
	K, N := b.Dim(1), b.Dim(2)
	st := b.Strides().Slice()
	return st[0] == N*K && st[1] == 1 && st[2] == K && K > 1 && N > 1
}
