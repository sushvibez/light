package tensor

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/storage"
)

// allocStorage builds a backing Storage for the given device. Dispatches to
// CPU, CUDA or WebGPU based on the device kind; CUDA is the default and is
// excluded only when building with `-tags nocuda`; WebGPU likewise unless
// building with `-tags nowebgpu`.
func allocStorage(dt dtype.DType, nelems int, dev device.Device) (*storage.Storage, error) {
	switch dev.Kind() {
	case device.KindCPU:
		return storage.NewCPU(dt, nelems), nil
	case device.KindCUDA:
		return allocCUDAStorage(dt, nelems, dev)
	case device.KindWebGPU:
		return allocWebGPUStorage(dt, nelems, dev)
	default:
		return nil, fmt.Errorf("tensor.allocStorage: unsupported device %v", dev)
	}
}
