//go:build !nowebgpu

package storage

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/wgpudev"
)

// NewWebGPU allocates a zero-filled device buffer for nelems elements of
// dtype dt on the given WebGPU device. The buffer is returned to the
// wgpudev pool when the refcount reaches zero.
func NewWebGPU(dev device.Device, dt dtype.DType, nelems int) (*Storage, error) {
	return newWebGPU(dev, dt, nelems, nil)
}

// NewWebGPUFromBytes allocates a device buffer and uploads data into it
// (no zero-fill pass). len(data) must equal dt.ByteSize(nelems).
func NewWebGPUFromBytes(dev device.Device, dt dtype.DType, nelems int, data []byte) (*Storage, error) {
	if want := dt.ByteSize(nelems); len(data) != want {
		return nil, fmt.Errorf("storage.NewWebGPUFromBytes: got %d bytes, want %d", len(data), want)
	}
	return newWebGPU(dev, dt, nelems, data)
}

func newWebGPU(dev device.Device, dt dtype.DType, nelems int, data []byte) (*Storage, error) {
	if dev.Kind() != device.KindWebGPU {
		panic("storage.NewWebGPU: device is not WebGPU")
	}
	d, err := wgpudev.Get()
	if err != nil {
		return nil, err
	}
	bytes := dt.ByteSize(nelems)
	buf, err := d.Alloc(uint64(bytes), data == nil)
	if err != nil {
		return nil, err
	}
	if len(data) > 0 {
		if err := d.Write(buf.Buf, 0, data); err != nil {
			d.Free(buf)
			return nil, err
		}
	}
	return &Storage{
		device:    dev,
		dtype:     dt,
		nelems:    nelems,
		bytes:     bytes,
		gpuHandle: buf,
		release:   func() { d.Free(buf) },
	}, nil
}

// WebGPUBuffer returns the pooled device buffer backing this Storage.
// Panics if the storage is not WebGPU-backed.
func (s *Storage) WebGPUBuffer() *wgpudev.Buffer {
	if s.device.Kind() != device.KindWebGPU {
		panic("storage.WebGPUBuffer: not a WebGPU storage")
	}
	return s.gpuHandle.(*wgpudev.Buffer)
}
