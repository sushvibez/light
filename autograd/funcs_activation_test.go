package autograd_test

import (
	"testing"

	"light/autograd"
	"light/device"
	"light/tensor"
)

func TestGradCheckELUMishHardActivations(t *testing.T) {
	cases := []struct {
		name string
		fn   func(*tensor.Tensor) (*tensor.Tensor, error)
	}{
		{"exp", autograd.Exp},
		{"log", func(x *tensor.Tensor) (*tensor.Tensor, error) {
			// keep the argument positive: log(x + 2)
			two, _ := tensor.Full[float32](2, []int{1}, x.Device())
			y, err := autograd.Add(x, two)
			if err != nil {
				return nil, err
			}
			return autograd.Log(y)
		}},
		{"abs", autograd.Abs},
		{"sqr", autograd.Sqr},
		{"elu", func(x *tensor.Tensor) (*tensor.Tensor, error) { return autograd.ELU(x, 0.7) }},
		{"leaky_relu", func(x *tensor.Tensor) (*tensor.Tensor, error) { return autograd.LeakyReLU(x, 0.1) }},
		{"mish", autograd.Mish},
		{"hard_swish", autograd.HardSwish},
		{"hard_sigmoid", autograd.HardSigmoid},
		{"clamp", func(x *tensor.Tensor) (*tensor.Tensor, error) { return autograd.Clamp(x, -0.5, 1.0) }},
		{"narrow", func(x *tensor.Tensor) (*tensor.Tensor, error) { return autograd.Narrow(x, 0, 1, 2) }},
		{"concat", func(x *tensor.Tensor) (*tensor.Tensor, error) {
			y, err := autograd.Sqr(x)
			if err != nil {
				return nil, err
			}
			return autograd.Concat([]*tensor.Tensor{x, y}, 0)
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			// Points away from the kinks (0, ±3, clamp bounds).
			x, _ := tensor.New([]float32{-2.3, -0.3, 0.5, 1.2, -0.9, 2.1}, []int{6}, device.CPU())
			compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
				y, err := c.fn(leaves[0])
				if err != nil {
					return nil, err
				}
				return autograd.Sum(y)
			}
			if err := autograd.GradCheck(compute, []*tensor.Tensor{x}, 1e-3, 2e-3); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestGradCheckMaximumMinimumPReLU(t *testing.T) {
	a, _ := tensor.New([]float32{-1, 2, 0.5, -0.2}, []int{2, 2}, device.CPU())
	b, _ := tensor.New([]float32{0.3, -1}, []int{2}, device.CPU()) // broadcast
	for _, c := range []struct {
		name string
		fn   func(a, b *tensor.Tensor) (*tensor.Tensor, error)
	}{{"maximum", autograd.Maximum}, {"minimum", autograd.Minimum}} {
		t.Run(c.name, func(t *testing.T) {
			compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
				y, err := c.fn(leaves[0], leaves[1])
				if err != nil {
					return nil, err
				}
				return autograd.Sum(y)
			}
			if err := autograd.GradCheck(compute, []*tensor.Tensor{a, b}, 1e-3, 1e-3); err != nil {
				t.Fatal(err)
			}
		})
	}
	w, _ := tensor.New([]float32{0.25, 0.5}, []int{2, 1}, device.CPU()) // per-row slope
	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := autograd.PReLU(leaves[0], leaves[1])
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	if err := autograd.GradCheck(compute, []*tensor.Tensor{a, w}, 1e-3, 1e-3); err != nil {
		t.Fatal(err)
	}
}
