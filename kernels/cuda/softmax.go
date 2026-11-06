//go:build !nocuda

package cuda

import (
	"fmt"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
)

func init() {
	registry.Register(registry.Key{Op: "softmax", Dev: device.KindCUDA, DType: dtype.F32}, softmaxF32)
	registry.Register(registry.Key{Op: "softmax", Dev: device.KindCUDA, DType: dtype.F64}, softmaxF64)
	registry.Register(registry.Key{Op: "softmax", Dev: device.KindCUDA, DType: dtype.BF16}, softmaxBF16)
}

// softmaxRowsCols derives (rows, cols) for a last-axis softmax kernel launch.
func softmaxRowsCols(dims []int, axis int) (uint64, uint64) {
	cols := uint64(dims[axis])
	rows := uint64(1)
	for i, d := range dims {
		if i == axis {
			continue
		}
		rows *= uint64(d)
	}
	return rows, cols
}

// softmaxF32 runs the last-axis softmax kernel. Input must be contiguous,
// and the Attrs["axis"] must reference the last axis.
func softmaxF32(ctx *registry.Context) error {
	a, out := ctx.Inputs[0], ctx.Output
	axis, _ := ctx.Attrs["axis"].(int)
	rank := a.Rank()
	if axis != rank-1 {
		return fmt.Errorf("cuda softmax: only last-axis softmax supported (axis=%d, rank=%d)", axis, rank)
	}
	if !a.IsContiguous() || a.Offset() != 0 {
		return fmt.Errorf("cuda softmax: non-contiguous input not yet supported")
	}
	rows, cols := softmaxRowsCols(a.Shape().Dims(), axis)

	ord := out.Device().Ordinal()
	if err := cudart.SetDevice(ord); err != nil {
		return err
	}
	fn, err := GetKernel("softmax_f32")
	if err != nil {
		return err
	}
	aPtr := a.Storage().CUDAPtr()
	outPtr := out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&aPtr),
		unsafe.Pointer(&outPtr),
		unsafe.Pointer(&rows),
		unsafe.Pointer(&cols),
	}
	const block = 256
	// shared memory = 256 floats
	const sharedBytes = block * 4
	err = cudart.LaunchKernel(fn, uint32(rows), 1, 1, block, 1, 1, sharedBytes, 0, args)
	if err != nil {
		return err
	}
	return nil
}

func softmaxBF16(ctx *registry.Context) error {
	a, out := ctx.Inputs[0], ctx.Output
	axis, _ := ctx.Attrs["axis"].(int)
	rank := a.Rank()
	if axis != rank-1 {
		return fmt.Errorf("cuda softmax_bf16: only last-axis softmax supported")
	}
	if !a.IsContiguous() || a.Offset() != 0 {
		return fmt.Errorf("cuda softmax_bf16: non-contiguous input not yet supported")
	}
	rows, cols := softmaxRowsCols(a.Shape().Dims(), axis)
	ord := out.Device().Ordinal()
	if err := cudart.SetDevice(ord); err != nil {
		return err
	}
	fn, err := GetKernel("softmax_bf16")
	if err != nil {
		return err
	}
	aPtr := a.Storage().CUDAPtr()
	outPtr := out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&aPtr),
		unsafe.Pointer(&outPtr),
		unsafe.Pointer(&rows),
		unsafe.Pointer(&cols),
	}
	const block = 256
	const sharedBytes = block * 4 // F32 reductions
	err = cudart.LaunchKernel(fn, uint32(rows), 1, 1, block, 1, 1, sharedBytes, 0, args)
	if err != nil {
		return err
	}
	return nil
}

func softmaxF64(ctx *registry.Context) error {
	a, out := ctx.Inputs[0], ctx.Output
	axis, _ := ctx.Attrs["axis"].(int)
	rank := a.Rank()
	if axis != rank-1 {
		return fmt.Errorf("cuda softmax: only last-axis softmax supported")
	}
	if !a.IsContiguous() || a.Offset() != 0 {
		return fmt.Errorf("cuda softmax: non-contiguous input not yet supported")
	}
	rows, cols := softmaxRowsCols(a.Shape().Dims(), axis)

	ord := out.Device().Ordinal()
	if err := cudart.SetDevice(ord); err != nil {
		return err
	}
	fn, err := GetKernel("softmax_f64")
	if err != nil {
		return err
	}
	aPtr := a.Storage().CUDAPtr()
	outPtr := out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&aPtr),
		unsafe.Pointer(&outPtr),
		unsafe.Pointer(&rows),
		unsafe.Pointer(&cols),
	}
	const block = 256
	const sharedBytes = block * 8
	err = cudart.LaunchKernel(fn, uint32(rows), 1, 1, block, 1, 1, sharedBytes, 0, args)
	if err != nil {
		return err
	}
	return nil
}
