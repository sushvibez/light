package autograd_test

import (
	"testing"

	"light/autograd"
	"light/device"
	"light/ops"
	"light/tensor"
)

func TestGradCheckConv2d(t *testing.T) {
	x, _ := tensor.New([]float32{
		// N=1, Cin=1, H=4, W=4
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}, []int{1, 1, 4, 4}, device.CPU())
	w, _ := tensor.New([]float32{
		// Cout=2, Cin=1, kH=3, kW=3
		1, 0, -1,
		1, 0, -1,
		1, 0, -1,
		0.5, 0.5, 0.5,
		0, 0, 0,
		-0.5, -0.5, -0.5,
	}, []int{2, 1, 3, 3}, device.CPU())
	b, _ := tensor.New([]float32{0.1, -0.1}, []int{2}, device.CPU())

	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := autograd.Conv2d(leaves[0], leaves[1], leaves[2], [2]int{1, 1}, [2]int{0, 0})
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	if err := autograd.GradCheck(compute, []*tensor.Tensor{x, w, b}, 1e-2, 5e-3); err != nil {
		t.Fatal(err)
	}
}

func TestGradCheckConv2dStridePadding(t *testing.T) {
	// Stride=2, padding=1. 1×1×6×6 input, 2×1×3×3 kernel.
	xData := make([]float32, 36)
	for i := range xData {
		xData[i] = float32(i)*0.1 - 1.5
	}
	x, _ := tensor.New(xData, []int{1, 1, 6, 6}, device.CPU())
	w, _ := tensor.New([]float32{
		0.1, 0.2, 0.3,
		-0.1, 0.0, 0.1,
		-0.3, -0.2, -0.1,
		0.5, -0.5, 0.5,
		-0.5, 0.5, -0.5,
		0.5, -0.5, 0.5,
	}, []int{2, 1, 3, 3}, device.CPU())
	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := autograd.Conv2d(leaves[0], leaves[1], nil, [2]int{2, 2}, [2]int{1, 1})
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	if err := autograd.GradCheck(compute, []*tensor.Tensor{x, w}, 1e-2, 5e-3); err != nil {
		t.Fatal(err)
	}
}

func TestGradCheckMaxPool2d(t *testing.T) {
	x, _ := tensor.New([]float32{
		// deliberately distinct values so argmax is stable under perturbation
		0.1, 0.3, 0.2, 0.4,
		0.7, 0.6, 0.5, 0.8,
		0.15, 0.35, 0.25, 0.45,
		0.75, 0.65, 0.55, 0.85,
	}, []int{1, 1, 4, 4}, device.CPU())
	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := autograd.MaxPool2d(leaves[0], [2]int{2, 2}, [2]int{2, 2}, [2]int{0, 0})
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	if err := autograd.GradCheck(compute, []*tensor.Tensor{x}, 1e-3, 1e-3); err != nil {
		t.Fatal(err)
	}
}

func TestGradCheckAvgPool2d(t *testing.T) {
	x, _ := tensor.New([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}, []int{1, 1, 4, 4}, device.CPU())
	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := autograd.AvgPool2d(leaves[0], [2]int{2, 2}, [2]int{2, 2}, [2]int{0, 0})
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	if err := autograd.GradCheck(compute, []*tensor.Tensor{x}, 1e-2, 5e-3); err != nil {
		t.Fatal(err)
	}
}

// BatchNorm2d backward requires the training-mode forward (with batch stats
// and cached x_hat/sigma). GradCheck needs a closure that re-runs the
// forward each call.
func TestGradCheckBatchNorm2d(t *testing.T) {
	xData := []float32{
		0.5, -1, 2, 0.1,
		1.5, 0.3, -0.5, 0.2,
		-0.2, 0.4, 0.8, 1.0,
		0.9, -0.7, 0.6, -0.1,
	}
	x, _ := tensor.New(xData, []int{2, 2, 2, 2}, device.CPU())
	gamma, _ := tensor.New([]float32{1.0, 0.5}, []int{2}, device.CPU())
	beta, _ := tensor.New([]float32{0.0, 0.1}, []int{2}, device.CPU())

	// Running stats (unused at training time, but the API requires them).
	rm, _ := tensor.Zeros(0 /*F32, handled*/, []int{2}, device.CPU())
	rv, _ := tensor.Zeros(0, []int{2}, device.CPU())
	// Need F32 explicitly.
	rm, _ = tensor.New([]float32{0, 0}, []int{2}, device.CPU())
	rv, _ = tensor.New([]float32{1, 1}, []int{2}, device.CPU())

	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, _, _, err := autograd.ApplyBatchNorm2d(leaves[0], leaves[1], leaves[2], rm, rv,
			ops.BatchNormOptions{Train: true, Momentum: 0.1, Eps: 1e-5})
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	if err := autograd.GradCheck(compute, []*tensor.Tensor{x, gamma, beta}, 1e-2, 5e-3); err != nil {
		t.Fatal(err)
	}
}
