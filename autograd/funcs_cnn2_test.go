package autograd_test

import (
	"testing"

	"light/autograd"
	"light/device"
	"light/tensor"
)

func TestGradCheckAdaptiveAvgPool2d(t *testing.T) {
	x, _ := tensor.New([]float32{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
		13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24,
	}, []int{1, 1, 4, 6}, device.CPU())
	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := autograd.AdaptiveAvgPool2d(leaves[0], 2, 3)
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	if err := autograd.GradCheck(compute, []*tensor.Tensor{x}, 1e-3, 5e-3); err != nil {
		t.Fatal(err)
	}
}

func TestGradCheckAdaptiveAvgPool2d1x1(t *testing.T) {
	// Global average pool: common in ResNet.
	x, _ := tensor.New([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}, []int{1, 1, 4, 4}, device.CPU())
	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := autograd.AdaptiveAvgPool2d(leaves[0], 1, 1)
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	if err := autograd.GradCheck(compute, []*tensor.Tensor{x}, 1e-3, 1e-3); err != nil {
		t.Fatal(err)
	}
}

func TestGradCheckConv2dDepthwise(t *testing.T) {
	// Depthwise: Cin=Cout=3, groups=3. Each output channel depends on exactly
	// one input channel + its corresponding filter.
	xData := make([]float32, 3*4*4)
	for i := range xData {
		xData[i] = float32(i)*0.05 - 1
	}
	x, _ := tensor.New(xData, []int{1, 3, 4, 4}, device.CPU())
	// Weight shape (Cout, Cin/groups, kH, kW) = (3, 1, 3, 3).
	wData := make([]float32, 3*1*3*3)
	for i := range wData {
		wData[i] = float32((i%5)-2) * 0.1
	}
	w, _ := tensor.New(wData, []int{3, 1, 3, 3}, device.CPU())
	b, _ := tensor.New([]float32{0.1, -0.1, 0}, []int{3}, device.CPU())

	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := autograd.Conv2dGrouped(leaves[0], leaves[1], leaves[2], [2]int{1, 1}, [2]int{1, 1}, 3)
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	if err := autograd.GradCheck(compute, []*tensor.Tensor{x, w, b}, 1e-2, 1e-2); err != nil {
		t.Fatal(err)
	}
}
