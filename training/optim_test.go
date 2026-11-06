package training_test

import (
	"math"
	"testing"

	"light/autograd"
	"light/device"
	"light/nn"
	"light/tensor"
	"light/training"
)

// TestSGDBasicUpdate: one SGD step should move the weight exactly lr * grad.
func TestSGDBasicUpdate(t *testing.T) {
	w, _ := tensor.New([]float32{1.0, 2.0, 3.0}, []int{3}, device.CPU())
	w.SetRequiresGrad(true)
	// Install a gradient of [0.1, 0.2, 0.3].
	g, _ := tensor.New([]float32{0.1, 0.2, 0.3}, []int{3}, device.CPU())
	w.SetGrad(g)

	opt, err := training.NewSGD([]*tensor.Tensor{w}, training.SGDOptions{LR: 0.5})
	if err != nil {
		t.Fatal(err)
	}
	if err := opt.Step(); err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](w)
	want := []float32{1.0 - 0.5*0.1, 2.0 - 0.5*0.2, 3.0 - 0.5*0.3}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Errorf("[%d] got %v want %v", i, got[i], want[i])
		}
	}
}

// TestSGDZeroGrad nulls out grad references so the next backward starts clean.
func TestSGDZeroGrad(t *testing.T) {
	w, _ := tensor.New([]float32{1.0, 2.0}, []int{2}, device.CPU())
	w.SetRequiresGrad(true)
	g, _ := tensor.New([]float32{0.5, 0.5}, []int{2}, device.CPU())
	w.SetGrad(g)

	opt, _ := training.NewSGD([]*tensor.Tensor{w}, training.SGDOptions{LR: 0.1})
	opt.ZeroGrad()
	if w.Grad() != nil {
		t.Errorf("expected nil grad after ZeroGrad, got %v", w.Grad())
	}
}

// TestSGDMomentum: with μ > 0 the velocity accumulates across steps.
func TestSGDMomentum(t *testing.T) {
	w, _ := tensor.New([]float32{1.0}, []int{1}, device.CPU())
	w.SetRequiresGrad(true)

	opt, _ := training.NewSGD([]*tensor.Tensor{w}, training.SGDOptions{LR: 0.1, Momentum: 0.9})

	// Step 1: grad = 1 → v = 1 → w = 1 - 0.1*1 = 0.9
	g, _ := tensor.New([]float32{1.0}, []int{1}, device.CPU())
	w.SetGrad(g)
	if err := opt.Step(); err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](w)[0]
	if math.Abs(float64(got-0.9)) > 1e-6 {
		t.Errorf("step1: got %v want 0.9", got)
	}
	// Step 2: grad = 1 → v = 0.9 + 1 = 1.9 → w = 0.9 - 0.1*1.9 = 0.71
	w.SetGrad(g)
	if err := opt.Step(); err != nil {
		t.Fatal(err)
	}
	got = tensor.MustSlice[float32](w)[0]
	if math.Abs(float64(got-0.71)) > 1e-4 {
		t.Errorf("step2: got %v want 0.71", got)
	}
}

// TestSGDWeightDecay: effective grad includes wd * p.
func TestSGDWeightDecay(t *testing.T) {
	w, _ := tensor.New([]float32{2.0}, []int{1}, device.CPU())
	w.SetRequiresGrad(true)
	g, _ := tensor.New([]float32{0.0}, []int{1}, device.CPU())
	w.SetGrad(g)
	// lr=0.1, wd=0.1: eff_grad = 0 + 0.1 * 2 = 0.2; w = 2 - 0.1*0.2 = 1.98
	opt, _ := training.NewSGD([]*tensor.Tensor{w}, training.SGDOptions{LR: 0.1, WeightDecay: 0.1})
	if err := opt.Step(); err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](w)[0]
	if math.Abs(float64(got-1.98)) > 1e-5 {
		t.Errorf("got %v want 1.98", got)
	}
}

// End-to-end: train a 2-layer nn.Sequential MLP on y = x^2 via AdamW.
// AdamW should reach much lower loss than SGD for the same number of steps.
func TestAdamWTrainsMLP(t *testing.T) {
	cpu := device.CPU()
	n := 32
	xData := make([]float32, n)
	yData := make([]float32, n)
	for i := range xData {
		xData[i] = -1 + 2*float32(i)/float32(n-1)
		yData[i] = xData[i] * xData[i]
	}
	x, _ := tensor.New(xData, []int{n, 1}, cpu)
	y, _ := tensor.New(yData, []int{n, 1}, cpu)

	l1, _ := nn.NewLinear(1, 16, true, cpu)
	l2, _ := nn.NewLinear(16, 1, true, cpu)
	model := nn.NewSequential(l1, nn.Tanh{}, l2)

	opt, err := training.NewAdamW(model.Parameters(), training.AdamWOptions{LR: 0.05})
	if err != nil {
		t.Fatal(err)
	}
	var first, last float32
	for step := range 500 {
		opt.ZeroGrad()
		pred, _ := model.Forward(x)
		diff, _ := autograd.Sub(pred, y)
		sq, _ := autograd.Mul(diff, diff)
		loss, err := autograd.Mean(sq)
		if err != nil {
			t.Fatal(err)
		}
		if err := autograd.Backward(loss, nil); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(); err != nil {
			t.Fatal(err)
		}
		cur := tensor.MustSlice[float32](loss)[0]
		if step == 0 {
			first = cur
		}
		last = cur
	}
	if last > 0.05*first {
		t.Errorf("AdamW loss insufficient: first=%v last=%v", first, last)
	}
	if math.IsNaN(float64(last)) || math.IsInf(float64(last), 0) {
		t.Errorf("AdamW diverged: %v", last)
	}
}

// End-to-end: train a 2-layer nn.Sequential MLP on y = x^2 via SGD. Loss
// must decrease substantially.
func TestSGDTrainsMLP(t *testing.T) {
	cpu := device.CPU()
	// Synthetic regression: y = x^2 over [-1, 1].
	n := 32
	xData := make([]float32, n)
	yData := make([]float32, n)
	for i := range xData {
		xData[i] = -1 + 2*float32(i)/float32(n-1)
		yData[i] = xData[i] * xData[i]
	}
	x, _ := tensor.New(xData, []int{n, 1}, cpu)
	y, _ := tensor.New(yData, []int{n, 1}, cpu)

	l1, _ := nn.NewLinear(1, 16, true, cpu)
	l2, _ := nn.NewLinear(16, 1, true, cpu)
	model := nn.NewSequential(l1, nn.Tanh{}, l2)

	opt, err := training.NewSGD(model.Parameters(), training.SGDOptions{LR: 0.1})
	if err != nil {
		t.Fatal(err)
	}

	var first, last float32
	for step := range 500 {
		opt.ZeroGrad()
		pred, err := model.Forward(x)
		if err != nil {
			t.Fatal(err)
		}
		diff, _ := autograd.Sub(pred, y)
		sq, _ := autograd.Mul(diff, diff)
		loss, err := autograd.Mean(sq)
		if err != nil {
			t.Fatal(err)
		}
		if err := autograd.Backward(loss, nil); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(); err != nil {
			t.Fatal(err)
		}
		cur := tensor.MustSlice[float32](loss)[0]
		if step == 0 {
			first = cur
		}
		last = cur
	}
	if last > 0.2*first {
		t.Errorf("SGD did not reduce loss enough: first=%v last=%v", first, last)
	}
	if math.IsNaN(float64(last)) || math.IsInf(float64(last), 0) {
		t.Errorf("loss diverged: %v", last)
	}
}
