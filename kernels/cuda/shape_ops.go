//go:build !nocuda && cgo

package cuda

import (
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
	"light/tensor"
)

func init() {
	registry.Register(registry.Key{Op: "pad", Dev: device.KindCUDA, DType: dtype.F32}, padF32CUDA)
	registry.Register(registry.Key{Op: "flip", Dev: device.KindCUDA, DType: dtype.F32}, flipF32CUDA)
	registry.Register(registry.Key{Op: "roll", Dev: device.KindCUDA, DType: dtype.F32}, rollF32CUDA)
	// "sort" has no CUDA kernel on purpose: ops.Sort round-trips through the
	// host (documented there).
}

// splitAroundDim mirrors the CPU helper: (outer, size, inner) around dim.
func splitAroundDim(dims []int, dim int) (outer, size, inner uint64) {
	outer, inner = 1, 1
	for i := 0; i < dim; i++ {
		outer *= uint64(dims[i])
	}
	for i := dim + 1; i < len(dims); i++ {
		inner *= uint64(dims[i])
	}
	return outer, uint64(dims[dim]), inner
}

func padF32CUDA(ctx *registry.Context) error {
	in, out := ctx.Inputs[0], ctx.Output
	dim := ctx.Attrs["dim"].(int)
	left := int64(ctx.Attrs["left"].(int))
	outer, size, inner := splitAroundDim(in.Shape().Dims(), dim)
	outSize := uint64(out.Dim(dim))
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("pad_f32")
	if err != nil {
		return err
	}
	inPtr := in.Storage().CUDAPtr()
	outPtr := out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&inPtr), unsafe.Pointer(&outPtr),
		unsafe.Pointer(&outer), unsafe.Pointer(&size), unsafe.Pointer(&outSize), unsafe.Pointer(&inner),
		unsafe.Pointer(&left),
	}
	if err := Launch1D(fn, outer*outSize*inner, args); err != nil {
		return err
	}
	return nil
}

// flipF32CUDA flips one dim per launch; multi-dim flips chain through a
// scratch tensor (flip is an involution per dim, so order is irrelevant).
func flipF32CUDA(ctx *registry.Context) error {
	in, out := ctx.Inputs[0], ctx.Output
	dims := ctx.Attrs["dims"].([]int)
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	if len(dims) == 0 {
		return cudart.MemcpyDtoD(out.Storage().CUDAPtr(), in.Storage().CUDAPtr(), uintptr(dtype.F32.ByteSize(in.NumElements())))
	}
	fn, err := GetKernel("flip_f32")
	if err != nil {
		return err
	}
	shapeDims := in.Shape().Dims()
	src := in
	for i, d := range dims {
		dst := out
		if i != len(dims)-1 {
			// Intermediate result: alternate between out and a scratch so we
			// never read and write the same buffer in one launch.
			if dst, err = tensor.Zeros(dtype.F32, shapeDims, out.Device()); err != nil {
				return err
			}
		}
		outer, size, inner := splitAroundDim(shapeDims, d)
		srcPtr := src.Storage().CUDAPtr()
		dstPtr := dst.Storage().CUDAPtr()
		args := []unsafe.Pointer{
			unsafe.Pointer(&srcPtr), unsafe.Pointer(&dstPtr),
			unsafe.Pointer(&outer), unsafe.Pointer(&size), unsafe.Pointer(&inner),
		}
		if err := Launch1D(fn, outer*size*inner, args); err != nil {
			return err
		}
		src = dst
	}
	return nil
}

func rollF32CUDA(ctx *registry.Context) error {
	in, out := ctx.Inputs[0], ctx.Output
	dim := ctx.Attrs["dim"].(int)
	shift := ctx.Attrs["shift"].(int)
	outer, size, inner := splitAroundDim(in.Shape().Dims(), dim)
	if size == 0 {
		return nil
	}
	s := int64(size)
	norm := uint64(((int64(shift) % s) + s) % s)
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("roll_f32")
	if err != nil {
		return err
	}
	inPtr := in.Storage().CUDAPtr()
	outPtr := out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&inPtr), unsafe.Pointer(&outPtr),
		unsafe.Pointer(&outer), unsafe.Pointer(&size), unsafe.Pointer(&inner), unsafe.Pointer(&norm),
	}
	if err := Launch1D(fn, outer*size*inner, args); err != nil {
		return err
	}
	return nil
}
