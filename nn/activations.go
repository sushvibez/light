package nn

import (
	"fmt"

	"light/autograd"
	"light/device"
	"light/tensor"
)

// ReLU is the rectified-linear-unit activation module: y = max(x, 0).
type ReLU struct{}

func (ReLU) Forward(x *tensor.Tensor) (*tensor.Tensor, error) { return autograd.ReLU(x) }
func (ReLU) Parameters() []*tensor.Tensor                     { return nil }

// SiLU is the sigmoid-linear-unit (a.k.a. Swish): y = x * sigmoid(x).
type SiLU struct{}

func (SiLU) Forward(x *tensor.Tensor) (*tensor.Tensor, error) { return autograd.SiLU(x) }
func (SiLU) Parameters() []*tensor.Tensor                     { return nil }

// Sigmoid activation.
type Sigmoid struct{}

func (Sigmoid) Forward(x *tensor.Tensor) (*tensor.Tensor, error) { return autograd.Sigmoid(x) }
func (Sigmoid) Parameters() []*tensor.Tensor                     { return nil }

// Tanh activation.
type Tanh struct{}

func (Tanh) Forward(x *tensor.Tensor) (*tensor.Tensor, error) { return autograd.Tanh(x) }
func (Tanh) Parameters() []*tensor.Tensor                     { return nil }

// GELU (tanh approximation, GPT-2 / HF "gelu_new").
type GELU struct{}

func (GELU) Forward(x *tensor.Tensor) (*tensor.Tensor, error) { return autograd.GELU(x) }
func (GELU) Parameters() []*tensor.Tensor                     { return nil }

// ReLU6 is min(max(x, 0), 6).
type ReLU6 struct{}

func (ReLU6) Forward(x *tensor.Tensor) (*tensor.Tensor, error) { return autograd.ReLU6(x) }
func (ReLU6) Parameters() []*tensor.Tensor                     { return nil }

// ELU is x for x > 0, Alpha·(eˣ-1) otherwise (Alpha 0 means 1).
type ELU struct{ Alpha float64 }

func (e ELU) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	a := e.Alpha
	if a == 0 {
		a = 1
	}
	return autograd.ELU(x, a)
}
func (ELU) Parameters() []*tensor.Tensor { return nil }

// LeakyReLU is x for x > 0, Slope·x otherwise (Slope 0 means 0.01).
type LeakyReLU struct{ Slope float64 }

func (l LeakyReLU) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	s := l.Slope
	if s == 0 {
		s = 0.01
	}
	return autograd.LeakyReLU(x, s)
}
func (LeakyReLU) Parameters() []*tensor.Tensor { return nil }

// Mish is x·tanh(softplus(x)).
type Mish struct{}

func (Mish) Forward(x *tensor.Tensor) (*tensor.Tensor, error) { return autograd.Mish(x) }
func (Mish) Parameters() []*tensor.Tensor                     { return nil }

// HardSwish is x·hard_sigmoid(x) (MobileNetV3).
type HardSwish struct{}

func (HardSwish) Forward(x *tensor.Tensor) (*tensor.Tensor, error) { return autograd.HardSwish(x) }
func (HardSwish) Parameters() []*tensor.Tensor                     { return nil }

// HardSigmoid is clamp((x+3)/6, 0, 1).
type HardSigmoid struct{}

func (HardSigmoid) Forward(x *tensor.Tensor) (*tensor.Tensor, error) { return autograd.HardSigmoid(x) }
func (HardSigmoid) Parameters() []*tensor.Tensor                     { return nil }

// PReLU is relu(x) + Weight·min(x, 0) with a learnable slope: one value
// shared across the tensor, or one per channel of NCHW input.
type PReLU struct {
	Weight *tensor.Tensor // (1,) or (C, 1, 1)
}

// NewPReLU creates the module with every slope initialised to 0.25
// (PyTorch default). channels 0 or 1 gives a single shared slope; otherwise
// the weight is (channels, 1, 1) for NCHW inputs.
func NewPReLU(channels int, dev device.Device) (*PReLU, error) {
	dims := []int{1}
	if channels > 1 {
		dims = []int{channels, 1, 1}
	}
	w, err := tensor.Full[float32](0.25, dims, dev)
	if err != nil {
		return nil, err
	}
	w.SetRequiresGrad(true)
	return &PReLU{Weight: w}, nil
}

func (p *PReLU) Forward(x *tensor.Tensor) (*tensor.Tensor, error) { return autograd.PReLU(x, p.Weight) }
func (p *PReLU) Parameters() []*tensor.Tensor                     { return []*tensor.Tensor{p.Weight} }

// SwiGLU splits the last axis in half and returns silu(a)·b (Candle
// `Activation::Swiglu`).
type SwiGLU struct{}

func (SwiGLU) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	d := x.Dim(-1)
	if d%2 != 0 {
		return nil, fmt.Errorf("SwiGLU: last dim %d must be even", d)
	}
	a, err := autograd.Narrow(x, -1, 0, d/2)
	if err != nil {
		return nil, err
	}
	b, err := autograd.Narrow(x, -1, d/2, d/2)
	if err != nil {
		return nil, err
	}
	if a, err = autograd.SiLU(a); err != nil {
		return nil, err
	}
	return autograd.Mul(a, b)
}
func (SwiGLU) Parameters() []*tensor.Tensor { return nil }
