//go:build !nowebgpu

package tensor

import (
	"light/device"
	"light/internal/wgpudev"
	"light/shape"
	"light/storage"
)

// toWebGPU uploads a CPU tensor to the WebGPU device via queue.WriteBuffer.
func (t *Tensor) toWebGPU(dev device.Device) (*Tensor, error) {
	src := t
	if !src.IsContiguous() || src.offset != 0 {
		c, err := src.Contiguous()
		if err != nil {
			return nil, err
		}
		src = c
	}
	// A Narrow view shares its parent's storage, so slice the byte buffer
	// to exactly the logical length.
	nbytes := src.DType().ByteSize(src.NumElements())
	srcBytes := src.Storage().CPUBytes()
	if nbytes > len(srcBytes) {
		nbytes = len(srcBytes)
	}
	s, err := storage.NewWebGPUFromBytes(dev, src.DType(), src.NumElements(), srcBytes[:nbytes])
	if err != nil {
		return nil, err
	}
	return newTensor(s, src.shape, shape.DefaultStrides(src.shape), 0), nil
}

// webGPUToCPU downloads a WebGPU tensor through a MAP_READ staging buffer.
// Strided sources are contiguified on the device first (via the registered
// WebGPU strided-copy kernel).
func (t *Tensor) webGPUToCPU(_ device.Device) (*Tensor, error) {
	src := t
	if !src.IsContiguous() || src.offset != 0 {
		c, err := src.Contiguous()
		if err != nil {
			return nil, err
		}
		src = c
	}
	d, err := wgpudev.Get()
	if err != nil {
		return nil, err
	}
	s := storage.NewCPU(src.DType(), src.NumElements())
	n := len(s.CPUBytes())
	if n > 0 {
		data, err := d.Read(src.Storage().WebGPUBuffer().Buf, 0, n)
		if err != nil {
			return nil, err
		}
		copy(s.CPUBytes(), data)
	}
	return newTensor(s, src.shape, shape.DefaultStrides(src.shape), 0), nil
}
