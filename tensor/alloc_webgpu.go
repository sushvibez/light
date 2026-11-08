//go:build !nowebgpu

package tensor

import (
	"light/device"
	"light/dtype"
	"light/storage"
)

func allocWebGPUStorage(dt dtype.DType, nelems int, dev device.Device) (*storage.Storage, error) {
	return storage.NewWebGPU(dev, dt, nelems)
}
