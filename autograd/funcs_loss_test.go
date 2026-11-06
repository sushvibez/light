package autograd_test

import (
	"math"
	"testing"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/tensor"
)

func TestGradCheckSoftmax(t *testing.T) {
	x, _ := tensor.New([]float32{0.2, -0.5, 1.3, 0.1}, []int{2, 2}, device.CPU())
	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := autograd.Softmax(leaves[0], 1)
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	if err := autograd.GradCheck(compute, []*tensor.Tensor{x}, 1e-3, 5e-3); err != nil {
		t.Fatal(err)
	}
}

func TestGradCheckLogSoftmax(t *testing.T) {
	x, _ := tensor.New([]float32{0.2, -0.5, 1.3, 0.1, 0.7, -0.2}, []int{2, 3}, device.CPU())
	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := autograd.LogSoftmax(leaves[0], 1)
		if err != nil {
			return nil, err
		}
		// Sum over negatives to make the result finite for any input.
		neg, err := autograd.Neg(y)
		if err != nil {
			return nil, err
		}
		return autograd.Sum(neg)
	}
	if err := autograd.GradCheck(compute, []*tensor.Tensor{x}, 1e-3, 5e-3); err != nil {
		t.Fatal(err)
	}
}

func TestCrossEntropyLossForward(t *testing.T) {
	// Single-row test: with logits [1, 2, 3], target = 2.
	// softmax = exp / sum_exp; -log(softmax[2]) is the loss.
	logits, _ := tensor.New([]float32{1, 2, 3}, []int{1, 3}, device.CPU())
	target, _ := tensor.New([]int64{2}, []int{1}, device.CPU())
	loss, err := autograd.CrossEntropyLoss(logits, target)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](loss)[0]
	// -log(e^3 / (e^1 + e^2 + e^3))
	denom := math.Exp(1) + math.Exp(2) + math.Exp(3)
	want := float32(-math.Log(math.Exp(3) / denom))
	if math.Abs(float64(got-want)) > 1e-4 {
		t.Errorf("got %v want %v", got, want)
	}
}

func TestCrossEntropyLossBackward(t *testing.T) {
	// Two-row check against finite differences.
	logits, _ := tensor.New([]float32{0.1, -0.2, 0.5, 1.0, 0.3, -0.1}, []int{2, 3}, device.CPU())
	target, _ := tensor.New([]int64{1, 2}, []int{2}, device.CPU())

	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		return autograd.CrossEntropyLoss(leaves[0], target)
	}
	if err := autograd.GradCheck(compute, []*tensor.Tensor{logits}, 1e-3, 1e-3); err != nil {
		t.Fatal(err)
	}
}

func TestCrossEntropyLossTargetValidation(t *testing.T) {
	logits, _ := tensor.New([]float32{0.1, 0.2, 0.3, 0.4}, []int{2, 2}, device.CPU())
	badTarget, _ := tensor.New([]int64{0, 5}, []int{2}, device.CPU())
	if _, err := autograd.CrossEntropyLoss(logits, badTarget); err == nil {
		t.Error("expected error for out-of-range target")
	}
	// Wrong dtype for target.
	f32Target, _ := tensor.New([]float32{0, 1}, []int{2}, device.CPU())
	if _, err := autograd.CrossEntropyLoss(logits, f32Target); err == nil {
		t.Error("expected error for F32 target")
	}
	_ = dtype.I64
}

func TestLossesValuesAndGrads(t *testing.T) {
	pred, _ := tensor.New([]float32{0.5, -1.0, 2.0, 0.1}, []int{4}, device.CPU())
	target, _ := tensor.New([]float32{0, 1, 1, 0}, []int{4}, device.CPU())
	scalar := func(x *tensor.Tensor) float64 { return float64(tensor.MustSlice[float32](x)[0]) }

	mse, err := autograd.MSELoss(pred, target)
	if err != nil {
		t.Fatal(err)
	}
	// ((0.5)² + (2)² + 1² + (0.1)²)/4 = (0.25+4+1+0.01)/4
	if want := (0.25 + 4 + 1 + 0.01) / 4; math.Abs(scalar(mse)-want) > 1e-6 {
		t.Errorf("mse %g want %g", scalar(mse), want)
	}

	bce, err := autograd.BCEWithLogitsLoss(pred, target)
	if err != nil {
		t.Fatal(err)
	}
	var want float64
	for i, x := range []float64{0.5, -1, 2, 0.1} {
		tt := []float64{0, 1, 1, 0}[i]
		want += math.Max(x, 0) - x*tt + math.Log1p(math.Exp(-math.Abs(x)))
	}
	want /= 4
	if math.Abs(scalar(bce)-want) > 1e-6 {
		t.Errorf("bce %g want %g", scalar(bce), want)
	}

	hub, err := autograd.HuberLoss(pred, target, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	// d = [0.5, -2, 1, 0.1]: 0.125, 1·(2-0.5)=1.5, 0.5, 0.005 → mean
	if want := (0.125 + 1.5 + 0.5 + 0.005) / 4; math.Abs(scalar(hub)-want) > 1e-6 {
		t.Errorf("huber %g want %g", scalar(hub), want)
	}

	logits, _ := tensor.New([]float32{1, 2, 0.5, -1, 0, 3}, []int{2, 3}, device.CPU())
	tgt, _ := tensor.New([]int64{1, 2}, []int{2}, device.CPU())
	logp, err := autograd.LogSoftmax(logits, 1)
	if err != nil {
		t.Fatal(err)
	}
	nll, err := autograd.NLLLoss(logp, tgt)
	if err != nil {
		t.Fatal(err)
	}
	ce, err := autograd.CrossEntropyLoss(logits, tgt)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(scalar(nll)-scalar(ce)) > 1e-6 {
		t.Errorf("nll %g != cross_entropy %g", scalar(nll), scalar(ce))
	}

	for _, c := range []struct {
		name string
		fn   func(p *tensor.Tensor) (*tensor.Tensor, error)
	}{
		{"mse", func(p *tensor.Tensor) (*tensor.Tensor, error) { return autograd.MSELoss(p, target) }},
		{"bce", func(p *tensor.Tensor) (*tensor.Tensor, error) { return autograd.BCEWithLogitsLoss(p, target) }},
		{"huber", func(p *tensor.Tensor) (*tensor.Tensor, error) { return autograd.HuberLoss(p, target, 1.0) }},
	} {
		compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) { return c.fn(leaves[0]) }
		if err := autograd.GradCheck(compute, []*tensor.Tensor{pred}, 1e-3, 1e-3); err != nil {
			t.Errorf("%s: %v", c.name, err)
		}
	}
	computeNLL := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		lp, err := autograd.LogSoftmax(leaves[0], 1)
		if err != nil {
			return nil, err
		}
		return autograd.NLLLoss(lp, tgt)
	}
	if err := autograd.GradCheck(computeNLL, []*tensor.Tensor{logits}, 1e-3, 1e-3); err != nil {
		t.Errorf("nll: %v", err)
	}
}
