package nn

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// InstanceNorm2d normalises each (n, c) spatial plane of an (N, C, H, W)
// input over its H*W elements and applies a per-channel affine (Gamma,
// Beta). It is GroupNorm with Groups = C and shares that kernel.
type InstanceNorm2d struct {
	Gamma *tensor.Tensor
	Beta  *tensor.Tensor
	Eps   float64
}

// NewInstanceNorm2d initialises gamma to ones and beta to zeros (F32).
func NewInstanceNorm2d(channels int, eps float64, dev device.Device) (*InstanceNorm2d, error) {
	return NewInstanceNorm2dWithDType(channels, eps, dtype.F32, dev)
}

// NewInstanceNorm2dWithDType allocates gamma/beta at the given dtype (the
// fused groupnorm kernel requires them to match the input dtype).
func NewInstanceNorm2dWithDType(channels int, eps float64, wdt dtype.DType, dev device.Device) (*InstanceNorm2d, error) {
	if channels <= 0 {
		return nil, fmt.Errorf("nn.InstanceNorm2d: channels must be positive (got %d)", channels)
	}
	g, err := tensor.Ones(wdt, []int{channels}, dev)
	if err != nil {
		return nil, err
	}
	b, err := tensor.Zeros(wdt, []int{channels}, dev)
	if err != nil {
		return nil, err
	}
	return &InstanceNorm2d{Gamma: g, Beta: b, Eps: eps}, nil
}

// Forward applies instance normalisation to x (N, C, H, W).
func (n *InstanceNorm2d) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	return ops.InstanceNorm2d(x, n.Gamma, n.Beta, n.Eps)
}

// Parameters returns gamma and beta.
func (n *InstanceNorm2d) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{n.Gamma, n.Beta}
}
