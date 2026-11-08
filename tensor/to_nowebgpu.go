//go:build nowebgpu

package tensor

import (
	"light/device"
)

func (t *Tensor) toWebGPU(_ device.Device) (*Tensor, error)    { return nil, errNoWebGPU }
func (t *Tensor) webGPUToCPU(_ device.Device) (*Tensor, error) { return nil, errNoWebGPU }
