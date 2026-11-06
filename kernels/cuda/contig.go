//go:build !nocuda && cgo

package cuda

import (
	"fmt"
	"unsafe"

	"light/dtype"
	"light/internal/cudart"
	"light/tensor"
)

func init() {
	tensor.RegisterCUDAContig(copyStridedToContig)
}

// copyStridedToContig implements CUDA-backed tensor.Contiguous: src has
// arbitrary strides; dst is contiguous, same shape and dtype, freshly
// allocated.
func copyStridedToContig(dst, src *tensor.Tensor) error {
	if !dst.IsContiguous() || dst.Offset() != 0 {
		return fmt.Errorf("cuda Contiguous: dst must be contiguous")
	}
	if !dst.Shape().Equal(src.Shape()) {
		return fmt.Errorf("cuda Contiguous: shape mismatch")
	}
	if dst.DType() != src.DType() {
		return fmt.Errorf("cuda Contiguous: dtype mismatch")
	}

	dims := src.Shape().Dims()
	str := src.Strides().Slice()
	rank := int32(len(dims))

	// Pack meta: [dims..., strides...] as int32.
	meta := make([]int32, 2*len(dims))
	for i, d := range dims {
		meta[i] = int32(d)
		meta[len(dims)+i] = int32(str[i])
	}

	if err := cudart.SetDevice(dst.Device().Ordinal()); err != nil {
		return err
	}
	metaDev, err := uploadMeta(dst.Device().Ordinal(), meta)
	if err != nil {
		return err
	}

	// Resolve kernel by dtype.
	kname, ok := map[dtype.DType]string{
		dtype.F32:    "copy_strided_f32",
		dtype.F64:    "copy_strided_f64",
		dtype.I32:    "copy_strided_i32",
		dtype.I64:    "copy_strided_i64",
		dtype.F16:    "copy_strided_f16",
		dtype.BF16:   "copy_strided_bf16",
		dtype.U8:     "copy_strided_u8",
		dtype.I8:     "copy_strided_i8",
		dtype.F8E4M3: "copy_strided_u8",
	}[src.DType()]
	if !ok {
		return fmt.Errorf("cuda Contiguous: unsupported dtype %v", src.DType())
	}
	fn, err := GetKernel(kname)
	if err != nil {
		return err
	}

	// Adjust source pointer for any element offset — in bytes.
	srcPtr := src.Storage().CUDAPtr()
	elemSize := src.DType().Size()
	if elemSize == 0 {
		return fmt.Errorf("cuda Contiguous: quantized dtypes not supported")
	}
	srcPtr += uintptr(src.Offset() * elemSize)

	dstPtr := dst.Storage().CUDAPtr()
	n := uint64(src.NumElements())

	if err := cudart.SetDevice(dst.Device().Ordinal()); err != nil {
		return err
	}
	args := []unsafe.Pointer{
		unsafe.Pointer(&srcPtr),
		unsafe.Pointer(&dstPtr),
		unsafe.Pointer(&n),
		unsafe.Pointer(&rank),
		unsafe.Pointer(&metaDev),
	}
	if err := Launch1D(fn, n, args); err != nil {
		return err
	}
	return nil
}
