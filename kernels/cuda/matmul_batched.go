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
	a, b, err := batchedMatmulPrep(a, b)
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
	if err := cudart.GemmExBF16BatchedRowMajor(h, M, N, K,
		a.Storage().CUDAPtr(),
		b.Storage().CUDAPtr(),
		out.Storage().CUDAPtr(),
		B,
	); err != nil {
		return err
	}
	return nil
}

func batchedMatmulF32CUDA(ctx *registry.Context) error {
	a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	a, b, err := batchedMatmulPrep(a, b)
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
	if err := cudart.SgemmStridedBatchedRowMajor(h, M, N, K,
		a.Storage().CUDAPtr(),
		b.Storage().CUDAPtr(),
		out.Storage().CUDAPtr(),
		B,
	); err != nil {
		return err
	}
	return nil
}

func batchedMatmulPrep(a, b *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	if a.Rank() != 3 || b.Rank() != 3 {
		return nil, nil, fmt.Errorf("cuda matmul_batched: need rank-3, got %d and %d", a.Rank(), b.Rank())
	}
	var err error
	if !a.IsContiguous() || a.Offset() != 0 {
		a, err = a.Contiguous()
		if err != nil {
			return nil, nil, err
		}
	}
	if !b.IsContiguous() || b.Offset() != 0 {
		b, err = b.Contiguous()
		if err != nil {
			return nil, nil, err
		}
	}
	return a, b, nil
}
