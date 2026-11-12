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
	registry.Register(registry.Key{Op: "matmul_batched", Dev: device.KindCUDA, DType: dtype.F32}, batchedMatmulLauncher(cudart.DataF32))
	registry.Register(registry.Key{Op: "matmul_batched", Dev: device.KindCUDA, DType: dtype.BF16}, batchedMatmulLauncher(cudart.DataBF16))
	registry.Register(registry.Key{Op: "matmul_batched", Dev: device.KindCUDA, DType: dtype.F16}, batchedMatmulLauncher(cudart.DataF16))
}

// batchedMatmulLauncher runs (count, M, K) × (count, K, N) → (count, M, N)
// on cuBLAS's strided-batched GEMM. Either operand may be any view whose
// slices are row-major blocks with a leading dimension and slice stride,
// or the transpose of one: attention's kᵀ against the key cache, and the
// narrowed, batch-and-head-folded views of a fixed-capacity KV cache, all
// read in place. Other layouts are contiguified first.
func batchedMatmulLauncher(dt cudart.DataType) registry.Kernel {
	return func(ctx *registry.Context) error {
		a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
		if a.Rank() != 3 || b.Rank() != 3 {
			return fmt.Errorf("cuda matmul_batched: need rank-3, got %d and %d", a.Rank(), b.Rank())
		}
		ord := out.Device().Ordinal()
		if err := cudart.SetDevice(ord); err != nil {
			return err
		}
		h, err := cudart.GetBlasHandle(ord)
		if err != nil {
			return err
		}
		opA, err := batchedOperand(a)
		if err != nil {
			return err
		}
		opB, err := batchedOperand(b)
		if err != nil {
			return err
		}
		return cudart.GemmStridedBatched(h, a.Dim(1), b.Dim(2), a.Dim(2), opA, opB, out.Storage().CUDAPtr(), dt, a.Dim(0))
	}
}

// batchedOperand describes a rank-3 view (count, R, C) to cuBLAS, copying
// it to a contiguous block only when neither its rows nor its columns are
// packed.
func batchedOperand(t *tensor.Tensor) (cudart.BatchedOperand, error) {
	R, Cc := t.Dim(1), t.Dim(2)
	st := t.Strides().Slice()
	es := t.DType().Size()
	base := t.Storage().CUDAPtr() + uintptr(t.Offset()*es)
	switch {
	case st[2] == 1 && st[1] >= Cc:
		return cudart.BatchedOperand{Ptr: base, LD: st[1], Stride: st[0]}, nil
	case st[1] == 1 && st[2] >= R && R > 1 && Cc > 1:
		return cudart.BatchedOperand{Ptr: base, LD: st[2], Stride: st[0], Trans: true}, nil
	}
	c, err := t.Contiguous()
	if err != nil {
		return cudart.BatchedOperand{}, err
	}
	return cudart.BatchedOperand{Ptr: c.Storage().CUDAPtr(), LD: Cc, Stride: R * Cc}, nil
}
