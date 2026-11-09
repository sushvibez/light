//go:build !nocuda && cgo

package cuda

import (
	"fmt"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
	"light/tensor"
)

// TopKMaxFused is the largest k the single-block top-k kernel handles;
// ops.TopK falls back to the full sort above it.
const TopKMaxFused = 64

func init() {
	registry.Register(registry.Key{Op: "topk", Dev: device.KindCUDA, DType: dtype.F32}, topkRowF32)
}

// topkRowF32: Inputs[0] (rows, cols) F32 contiguous, Output (rows, k) F32
// values, Attrs: "k" int, "largest" bool, "idx" *tensor.Tensor (rows, k)
// I64 on the same device.
func topkRowF32(ctx *registry.Context) error {
	a, vals := ctx.Inputs[0], ctx.Output
	k := ctx.Attrs["k"].(int)
	largest := int32(0)
	if ctx.Attrs["largest"].(bool) {
		largest = 1
	}
	idx, _ := ctx.Attrs["idx"].(*tensor.Tensor)
	if idx == nil || k > TopKMaxFused {
		return fmt.Errorf("cuda topk: need an I64 idx tensor and k <= %d (got %d)", TopKMaxFused, k)
	}
	cols := int64(a.Dim(-1))
	rows := int64(a.NumElements()) / cols
	if err := cudart.SetDevice(vals.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("topk_row_f32")
	if err != nil {
		return err
	}
	aPtr, vPtr, iPtr := a.Storage().CUDAPtr(), vals.Storage().CUDAPtr(), idx.Storage().CUDAPtr()
	k32 := int32(k)
	args := []unsafe.Pointer{
		unsafe.Pointer(&aPtr), unsafe.Pointer(&vPtr), unsafe.Pointer(&iPtr),
		unsafe.Pointer(&rows), unsafe.Pointer(&cols), unsafe.Pointer(&k32), unsafe.Pointer(&largest),
	}
	return cudart.LaunchKernel(fn, uint32(rows), 1, 1, 256, 1, 1, 0, 0, args)
}

// argmaxRowF32 is the block-parallel last-axis argmax (one 256-thread block
// per row, first index on ties).
func argmaxRowF32(ctx *registry.Context) error {
	a, out := ctx.Inputs[0], ctx.Output
	if axes, _ := ctx.Attrs["axes"].([]int); !isLastAxisContiguous(axes, a.Rank()) {
		return fmt.Errorf("cuda argmax: only last-axis supported")
	}
	cols := int64(a.Dim(-1))
	rows := int64(a.NumElements()) / cols
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("argmax_row_f32")
	if err != nil {
		return err
	}
	aPtr, oPtr := a.Storage().CUDAPtr(), out.Storage().CUDAPtr()
	args := []unsafe.Pointer{unsafe.Pointer(&aPtr), unsafe.Pointer(&oPtr), unsafe.Pointer(&rows), unsafe.Pointer(&cols)}
	return cudart.LaunchKernel(fn, uint32(rows), 1, 1, 256, 1, 1, 0, 0, args)
}
