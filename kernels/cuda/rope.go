//go:build !nocuda

package cuda

import (
	"unsafe"

	"light/internal/cudart"
	"light/tensor"
)

// ApplyRoPE runs the rope_f32 kernel on a CUDA tensor. x is (..., seqLen,
// headDim); positions is (seqLen,) I64. Returns `out` freshly allocated.
func ApplyRoPE(x, positions *tensor.Tensor, base float32) (*tensor.Tensor, error) {
	xC, err := x.Contiguous()
	if err != nil {
		return nil, err
	}
	posC, err := positions.Contiguous()
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(x.DType(), x.Shape().Dims(), x.Device())
	if err != nil {
		return nil, err
	}
	headDim := uint64(x.Dim(-1))
	seqLen := uint64(x.Dim(-2))
	rows := uint64(x.NumElements()) / headDim
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return nil, err
	}
	fn, err := GetKernel("rope_f32")
	if err != nil {
		return nil, err
	}
	xPtr := xC.Storage().CUDAPtr()
	posPtr := posC.Storage().CUDAPtr()
	outPtr := out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&xPtr),
		unsafe.Pointer(&posPtr),
		unsafe.Pointer(&outPtr),
		unsafe.Pointer(&rows),
		unsafe.Pointer(&seqLen),
		unsafe.Pointer(&headDim),
		unsafe.Pointer(&base),
	}
	halfDim := headDim / 2
	total := rows * halfDim
	if err := Launch1D(fn, total, args); err != nil {
		return nil, err
	}
	return out, nil
}
