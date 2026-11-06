package tensor

import (
	"fmt"

	"light/device"
)

// To returns a tensor with the same values on the target device. If the
// source is already on that device, returns the receiver unchanged. Cross-
// device copies allocate new storage and memcpy the data. Non-contiguous
// source tensors are contiguified on the source device first.
//
// Supported transitions: CPU<->CUDA and CPU<->WebGPU. GPU<->GPU moves go
// through the CPU explicitly.
func (t *Tensor) To(dev device.Device) (*Tensor, error) {
	if dev.SameAs(t.Device()) {
		return t, nil
	}
	from := t.Device().Kind()
	switch {
	case dev.Kind() == device.KindCUDA && from == device.KindCPU:
		return t.toCUDA(dev)
	case dev.Kind() == device.KindCPU && from == device.KindCUDA:
		return t.toCPU(dev)
	case dev.Kind() == device.KindWebGPU && from == device.KindCPU:
		return t.toWebGPU(dev)
	case dev.Kind() == device.KindCPU && from == device.KindWebGPU:
		return t.webGPUToCPU(dev)
	default:
		return nil, fmt.Errorf("tensor.To: unsupported transition %v → %v", t.Device(), dev)
	}
}
