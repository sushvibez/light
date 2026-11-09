//go:build nowebgpu

package tensor

import (
	"light/device"
	"light/dtype"
	"light/storage"
)

func allocWebGPUStorage(dt dtype.DType, nelems int, dev device.Device) (*storage.Storage, error) {
	return nil, errNoWebGPU
}

// TrimWebGPU is a no-op in builds without WebGPU.
func TrimWebGPU() {}

// WebGPUMaxTensorBytes is 0 in builds without WebGPU.
func WebGPUMaxTensorBytes() int64 { return 0 }
