//go:build nocuda

package tensor

import (
	"light/device"
)

func (t *Tensor) toCUDA(_ device.Device) (*Tensor, error) { return nil, errNoCUDA }
func (t *Tensor) toCPU(_ device.Device) (*Tensor, error)  { return nil, errNoCUDA }
