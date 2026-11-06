package autograd_test

import (
	"math"
	"testing"

	"light/autograd"
	"light/device"
	"light/tensor"
)

func TestDropoutEvalIsIdentity(t *testing.T) {
	x, _ := tensor.New([]float32{1, 2, 3, 4, 5}, []int{5}, device.CPU())
	y, err := autograd.Dropout(x, 0.5, false /*training*/)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](y)
	want := []float32{1, 2, 3, 4, 5}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Errorf("eval mode should be identity, got %v", got)
		}
	}
}

func TestDropoutTrainingDropsSome(t *testing.T) {
	// Large-ish input with p=0.5; expect roughly half zeros (probabilistic test).
	n := 10000
	data := make([]float32, n)
	for i := range data {
		data[i] = 1
	}
	x, _ := tensor.New(data, []int{n}, device.CPU())
	x.SetRequiresGrad(true)
	y, err := autograd.Dropout(x, 0.5, true)
	if err != nil {
		t.Fatal(err)
	}
	vals := tensor.MustSlice[float32](y)
	zeros := 0
	for _, v := range vals {
		if v == 0 {
			zeros++
		}
	}
	// Expect ~5000 ± 5σ = ~250.
	if zeros < 4500 || zeros > 5500 {
		t.Errorf("expected ~5000 zeros with p=0.5, got %d", zeros)
	}
}

func TestGradCheckDropoutEval(t *testing.T) {
	// In eval mode, the gradient must pass through unchanged.
	x, _ := tensor.New([]float32{0.2, -0.3, 1.5, 0.7}, []int{4}, device.CPU())
	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := autograd.Dropout(leaves[0], 0.5, false)
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	if err := autograd.GradCheck(compute, []*tensor.Tensor{x}, 1e-3, 1e-4); err != nil {
		t.Fatal(err)
	}
}
