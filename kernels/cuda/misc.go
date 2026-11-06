//go:build !nocuda && cgo

package cuda

import (
	"unsafe"

	"light/internal/cudart"
	"light/tensor"
)

// ApplyDropoutF32 writes out = x * scale where the hash(seed, i) exceeds p,
// else 0. Caller holds the RNG seed to control determinism.
func ApplyDropoutF32(x *tensor.Tensor, p, scale float32, seed uint64) (*tensor.Tensor, error) {
	xC, err := x.Contiguous()
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(x.DType(), x.Shape().Dims(), x.Device())
	if err != nil {
		return nil, err
	}
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return nil, err
	}
	fn, err := GetKernel("dropout_f32")
	if err != nil {
		return nil, err
	}
	xPtr := xC.Storage().CUDAPtr()
	outPtr := out.Storage().CUDAPtr()
	n := uint64(x.NumElements())
	args := []unsafe.Pointer{
		unsafe.Pointer(&xPtr),
		unsafe.Pointer(&outPtr),
		unsafe.Pointer(&n),
		unsafe.Pointer(&p),
		unsafe.Pointer(&scale),
		unsafe.Pointer(&seed),
	}
	if err := Launch1D(fn, n, args); err != nil {
		return nil, err
	}
	return out, nil
}

// ApplyTrilF32 zeroes the elements of a CUDA F32 tensor above the k-th
// diagonal of its trailing two axes. Caller is expected to pass a freshly
// allocated / copied output.
func ApplyTrilF32(data *tensor.Tensor, k int) error {
	return runTriangleKernel(data, k, "tril_f32")
}

// ApplyTriuF32 zeroes the elements below the k-th diagonal.
func ApplyTriuF32(data *tensor.Tensor, k int) error {
	return runTriangleKernel(data, k, "triu_f32")
}

func runTriangleKernel(data *tensor.Tensor, k int, kname string) error {
	rank := data.Rank()
	dims := data.Shape().Dims()
	rows := uint64(dims[rank-2])
	cols := uint64(dims[rank-1])
	batch := uint64(data.NumElements()) / (rows * cols)
	if err := cudart.SetDevice(data.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel(kname)
	if err != nil {
		return err
	}
	dataPtr := data.Storage().CUDAPtr()
	kI := int32(k)
	args := []unsafe.Pointer{
		unsafe.Pointer(&dataPtr),
		unsafe.Pointer(&batch),
		unsafe.Pointer(&rows),
		unsafe.Pointer(&cols),
		unsafe.Pointer(&kI),
	}
	n := batch * rows * cols
	if err := Launch1D(fn, n, args); err != nil {
		return err
	}
	return nil
}
