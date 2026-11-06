//go:build !nocuda

package tensor

import (
	"light/device"
	"light/internal/cudart"
	"light/shape"
	"light/storage"
)

// toCUDA copies a CPU tensor to a CUDA device.
func (t *Tensor) toCUDA(dev device.Device) (*Tensor, error) {
	src := t
	if !src.IsContiguous() || src.offset != 0 {
		c, err := src.Contiguous()
		if err != nil {
			return nil, err
		}
		src = c
	}
	s, err := storage.NewCUDA(dev, src.DType(), src.NumElements())
	if err != nil {
		return nil, err
	}
	if err := cudart.SetDevice(dev.Ordinal()); err != nil {
		return nil, err
	}
	// Slice CPUBytes to exactly the tensor's logical byte length — a
	// `Narrow(0, 0, n)` view shares the parent storage, so the underlying
	// byte slice can be much larger than what the destination CUDA buffer
	// holds.
	nbytes := src.DType().ByteSize(src.NumElements())
	srcBytes := src.Storage().CPUBytes()
	if nbytes > len(srcBytes) {
		nbytes = len(srcBytes)
	}
	if err := cudart.MemcpyHtoD(s.CUDAPtr(), srcBytes[:nbytes]); err != nil {
		return nil, err
	}
	return newTensor(s, src.shape, shape.DefaultStrides(src.shape), 0), nil
}

// toCPU copies a CUDA tensor down to the CPU. If the source has strides that
// differ from default row-major, it's first Contiguous()-ified on the device
// (via the CUDA strided-copy kernel), then memcpy'd to host.
func (t *Tensor) toCPU(_ device.Device) (*Tensor, error) {
	src := t
	if !src.IsContiguous() || src.offset != 0 {
		c, err := src.Contiguous()
		if err != nil {
			return nil, err
		}
		src = c
	}
	s := storage.NewCPU(src.DType(), src.NumElements())
	if err := cudart.SetDevice(src.Device().Ordinal()); err != nil {
		return nil, err
	}
	if err := cudart.MemcpyDtoH(s.CPUBytes(), src.Storage().CUDAPtr()); err != nil {
		return nil, err
	}
	return newTensor(s, src.shape, shape.DefaultStrides(src.shape), 0), nil
}
