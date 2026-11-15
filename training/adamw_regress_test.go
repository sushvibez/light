package training_test

import (
	"math"
	"testing"

	"light/device"
	"light/tensor"
	"light/training"
)

// R8: bias correction follows each parameter's own update count. A
// parameter first updated on the optimizer's second step takes a plain
// first AdamW step: p - lr·sign(g) (≈ 0.9 here), not the over-corrected
// ≈ 0.9256 a global step count gives. Entirely empty steps are no-ops.
func TestAdamWPerParameterStep(t *testing.T) {
	a, _ := tensor.New([]float32{1}, []int{1}, device.CPU())
	b, _ := tensor.New([]float32{1}, []int{1}, device.CPU())
	opt, err := training.NewAdamW([]*tensor.Tensor{a, b}, training.AdamWOptions{LR: 0.1})
	if err != nil {
		t.Fatal(err)
	}
	one := func() *tensor.Tensor { g, _ := tensor.New([]float32{1}, []int{1}, device.CPU()); return g }

	if err := opt.Step(); err != nil { // no grads at all
		t.Fatal(err)
	}
	a.SetGrad(one())
	if err := opt.Step(); err != nil {
		t.Fatal(err)
	}
	opt.ZeroGrad()
	if err := opt.Step(); err != nil { // empty again
		t.Fatal(err)
	}
	b.SetGrad(one())
	if err := opt.Step(); err != nil {
		t.Fatal(err)
	}
	for name, p := range map[string]*tensor.Tensor{"a": a, "b": b} {
		if got := tensor.MustSlice[float32](p)[0]; math.Abs(float64(got)-0.9) > 1e-5 {
			t.Errorf("%s = %v want ≈0.9", name, got)
		}
	}
}

// Q1: AdamW rejects non-finite or out-of-range hyperparameters.
func TestAdamWValidatesHyperparameters(t *testing.T) {
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	for name, o := range map[string]training.AdamWOptions{
		"lr<0":    {LR: -1},
		"lr NaN":  {LR: nan},
		"lr Inf":  {LR: inf},
		"eps<0":   {Eps: -1e-8},
		"eps NaN": {Eps: nan},
		"beta1=1": {Beta1: 1},
		"beta1<0": {Beta1: -0.1},
		"beta2>1": {Beta2: 1.5},
		"wd<0":    {WeightDecay: -0.01},
		"wd NaN":  {WeightDecay: nan},
	} {
		p, _ := tensor.New([]float32{1}, []int{1}, device.CPU())
		if _, err := training.NewAdamW([]*tensor.Tensor{p}, o); err == nil {
			t.Errorf("%s: want error", name)
		}
	}
	p, _ := tensor.New([]float32{1}, []int{1}, device.CPU())
	if _, err := training.NewAdamW([]*tensor.Tensor{p}, training.AdamWOptions{LR: 1e-3, WeightDecay: 0.01}); err != nil {
		t.Errorf("valid options rejected: %v", err)
	}
}
