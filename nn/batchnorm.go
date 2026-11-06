package nn

import (
	"fmt"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// BatchNorm2d normalises over the (N, H, W) axes per channel.
//
// Parameters: Weight (γ) and Bias (β), each of shape (C,). Running mean/var
// buffers are maintained out-of-band and updated only when Training == true.
type BatchNorm2d struct {
	Weight      *tensor.Tensor // gamma
	Bias        *tensor.Tensor // beta
	RunningMean *tensor.Tensor
	RunningVar  *tensor.Tensor

	channels int
	eps      float32
	momentum float32
	Training bool // set to false during evaluation to use running stats
}

// NewBatchNorm2d initialises γ=1, β=0, running_mean=0, running_var=1.
func NewBatchNorm2d(channels int, eps, momentum float32, dev device.Device) (*BatchNorm2d, error) {
	if channels <= 0 {
		return nil, fmt.Errorf("nn.BatchNorm2d: channels must be positive")
	}
	if eps == 0 {
		eps = 1e-5
	}
	if momentum == 0 {
		momentum = 0.1
	}
	gamma, err := tensor.Ones(dtype.F32, []int{channels}, dev)
	if err != nil {
		return nil, err
	}
	beta, err := tensor.Zeros(dtype.F32, []int{channels}, dev)
	if err != nil {
		return nil, err
	}
	rm, err := tensor.Zeros(dtype.F32, []int{channels}, dev)
	if err != nil {
		return nil, err
	}
	rv, err := tensor.Ones(dtype.F32, []int{channels}, dev)
	if err != nil {
		return nil, err
	}
	return &BatchNorm2d{
		Weight: gamma, Bias: beta, RunningMean: rm, RunningVar: rv,
		channels: channels, eps: eps, momentum: momentum, Training: true,
	}, nil
}

func (b *BatchNorm2d) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	y, newRM, newRV, err := autograd.ApplyBatchNorm2d(
		x, b.Weight, b.Bias, b.RunningMean, b.RunningVar,
		ops.BatchNormOptions{Train: b.Training, Momentum: b.momentum, Eps: b.eps},
	)
	if err != nil {
		return nil, err
	}
	// Update running stats in place on the module.
	b.RunningMean = newRM
	b.RunningVar = newRV
	return y, nil
}

func (b *BatchNorm2d) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{b.Weight, b.Bias}
}

// SetTraining toggles whether the module uses batch statistics (true) or
// running statistics (false). Standard PyTorch idiom: model.eval() ≡
// m.SetTraining(false) on every BN layer.
func (b *BatchNorm2d) SetTraining(v bool) { b.Training = v }
