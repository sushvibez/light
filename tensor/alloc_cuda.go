//go:build !nocuda && cgo

package tensor

import (
	"light/device"
	"light/dtype"
	"light/storage"
)

func allocCUDAStorage(dt dtype.DType, nelems int, dev device.Device) (*storage.Storage, error) {
	return storage.NewCUDA(dev, dt, nelems)
}
