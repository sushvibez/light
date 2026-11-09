//go:build !nowebgpu

package tensor

import (
	"light/device"
	"light/dtype"
	"light/internal/wgpudev"
	"light/storage"
)

func allocWebGPUStorage(dt dtype.DType, nelems int, dev device.Device) (*storage.Storage, error) {
	return storage.NewWebGPU(dev, dt, nelems)
}

// TrimWebGPU waits for the queued WebGPU work and releases the idle pooled
// buffers: after a pass that replaces many large tensors (narrowing weights
// at load), the replaced buffers' memory goes back to the driver instead of
// the pool. A no-op without an initialized WebGPU device.
func TrimWebGPU() {
	d, err := wgpudev.Get()
	if err != nil {
		return
	}
	d.Synchronize()
	d.PoolClear()
}

// WebGPUMaxTensorBytes is the largest tensor the WebGPU device can hold in
// one bound buffer (the smaller of the buffer and storage-binding caps), or
// 0 when no WebGPU device is up.
func WebGPUMaxTensorBytes() int64 {
	d, err := wgpudev.Get()
	if err != nil {
		return 0
	}
	n := d.MaxBufferSize
	if d.MaxStorageBufferBindingSize < n {
		n = d.MaxStorageBufferBindingSize
	}
	return int64(n)
}
