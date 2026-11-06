//go:build !nocuda

package autograd_test

import (
	"math"
	"testing"

	"light/autograd"
	"light/device"
	"light/tensor"
)

// parityGrad runs the forward + backward on both CPU and CUDA and compares
// the computed gradients elementwise. Only F32 and assumes `compute` writes
// the same forward graph on both devices.
func parityGrad(
	t *testing.T,
	gpu device.Device,
	shapes [][]int,
	cpuLeaves []*tensor.Tensor,
	compute func(leaves []*tensor.Tensor) (*tensor.Tensor, error),
	atol float64,
) {
	t.Helper()
	// CPU backward
	for _, l := range cpuLeaves {
		l.SetRequiresGrad(true)
		l.SetGrad(nil)
	}
	lossCPU, err := compute(cpuLeaves)
	if err != nil {
		t.Fatalf("cpu forward: %v", err)
	}
	if err := autograd.Backward(lossCPU, nil); err != nil {
		t.Fatalf("cpu backward: %v", err)
	}
	cpuGrads := make([][]float32, len(cpuLeaves))
	for i, l := range cpuLeaves {
		cont, _ := l.Grad().Contiguous()
		cpuGrads[i] = tensor.MustSlice[float32](cont)
	}

	// Move leaves to GPU, rerun.
	gpuLeaves := make([]*tensor.Tensor, len(cpuLeaves))
	for i, l := range cpuLeaves {
		g, err := l.To(gpu)
		if err != nil {
			t.Fatalf("to gpu: %v", err)
		}
		g.SetRequiresGrad(true)
		g.SetGrad(nil)
		gpuLeaves[i] = g
	}
	lossGPU, err := compute(gpuLeaves)
	if err != nil {
		t.Fatalf("gpu forward: %v", err)
	}
	if err := autograd.Backward(lossGPU, nil); err != nil {
		t.Fatalf("gpu backward: %v", err)
	}
	for i, l := range gpuLeaves {
		gCPU, err := l.Grad().To(device.CPU())
		if err != nil {
			t.Fatalf("gpu grad to cpu: %v", err)
		}
		cont, _ := gCPU.Contiguous()
		got := tensor.MustSlice[float32](cont)
		for k := range got {
			if math.Abs(float64(got[k]-cpuGrads[i][k])) > atol {
				t.Errorf("leaf %d index %d: cpu=%v gpu=%v", i, k, cpuGrads[i][k], got[k])
				break
			}
		}
	}
	_ = shapes
}

func TestCUDAConv2dBackwardParity(t *testing.T) {
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	x, _ := tensor.New(rangeF32(48), []int{2, 1, 4, 6}, device.CPU())
	w, _ := tensor.New(rangeF32(18), []int{2, 1, 3, 3}, device.CPU())
	b, _ := tensor.New([]float32{0.1, -0.2}, []int{2}, device.CPU())

	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := autograd.Conv2d(leaves[0], leaves[1], leaves[2], [2]int{1, 1}, [2]int{1, 1})
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	parityGrad(t, gpu, nil, []*tensor.Tensor{x, w, b}, compute, 1e-3)
}

func TestCUDAMaxPool2dBackwardParity(t *testing.T) {
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	x, _ := tensor.New([]float32{
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
	parityGrad(t, gpu, nil, []*tensor.Tensor{x}, compute, 1e-5)
}

func TestCUDAAvgPool2dBackwardParity(t *testing.T) {
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	x, _ := tensor.New(rangeF32(16), []int{1, 1, 4, 4}, device.CPU())
	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := autograd.AvgPool2d(leaves[0], [2]int{2, 2}, [2]int{2, 2}, [2]int{0, 0})
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	parityGrad(t, gpu, nil, []*tensor.Tensor{x}, compute, 1e-5)
}

func rangeF32(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(i)*0.1 - float32(n)*0.05
	}
	return out
}
