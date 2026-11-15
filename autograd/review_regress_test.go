package autograd_test

import (
	"math"
	"strings"
	"testing"

	"light/autograd"
	"light/device"
	"light/tensor"
)

// R4: the BCE-with-logits gradient at a zero logit is σ(0) - t = ∓0.5, not
// the -1 / 0 that separate ReLU and Abs subgradients produce.
func TestBCEWithLogitsZeroLogitGrad(t *testing.T) {
	for _, c := range []struct{ target, want float32 }{{1, -0.5}, {0, 0.5}} {
		x, _ := tensor.New([]float32{0}, []int{1}, device.CPU())
		x.SetRequiresGrad(true)
		tgt, _ := tensor.New([]float32{c.target}, []int{1}, device.CPU())
		loss, err := autograd.BCEWithLogitsLoss(x, tgt)
		if err != nil {
			t.Fatal(err)
		}
		if got := tensor.MustSlice[float32](loss)[0]; math.Abs(float64(got)-math.Log(2)) > 1e-6 {
			t.Errorf("target %v: loss %v want ln2", c.target, got)
		}
		if err := autograd.Backward(loss, nil); err != nil {
			t.Fatal(err)
		}
		if got := tensor.MustSlice[float32](x.Grad())[0]; math.Abs(float64(got-c.want)) > 1e-6 {
			t.Errorf("target %v: grad %v want %v", c.target, got, c.want)
		}
	}
}

// R4: the target gradient (-x/N) and a broadcast target still check out.
func TestBCEWithLogitsGradCheckTargetAndBroadcast(t *testing.T) {
	x, _ := tensor.New([]float32{0, 0.7, -1.2, 0, 2, -0.3}, []int{2, 3}, device.CPU())
	tg, _ := tensor.New([]float32{1, 0, 0.3}, []int{3}, device.CPU())
	compute := func(l []*tensor.Tensor) (*tensor.Tensor, error) {
		return autograd.BCEWithLogitsLoss(l[0], l[1])
	}
	if err := autograd.GradCheck(compute, []*tensor.Tensor{x, tg}, 1e-3, 1e-3); err != nil {
		t.Fatal(err)
	}
}

// R9: empty axes mean every axis; bad axes are errors, not panics.
func TestAxisMeanAxes(t *testing.T) {
	x, _ := tensor.New([]float32{1, 2, 3, 4}, []int{2, 2}, device.CPU())
	m, err := autograd.AxisMean(x, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if got := tensor.MustSlice[float32](m); len(got) != 1 || got[0] != 2.5 {
		t.Errorf("AxisMean(nil) = %v want [2.5]", got)
	}
	m, err = autograd.AxisMean(x, []int{-1}, false)
	if err != nil {
		t.Fatal(err)
	}
	if got := tensor.MustSlice[float32](m); got[0] != 1.5 || got[1] != 3.5 {
		t.Errorf("AxisMean(-1) = %v want [1.5 3.5]", got)
	}
	for _, axes := range [][]int{{2}, {-3}, {0, 0}, {1, -1}} {
		if _, err := autograd.AxisMean(x, axes, false); err == nil {
			t.Errorf("AxisMean(%v): want error", axes)
		}
	}
}

// R10: F64 reductions and activations run their backward at F64.
func TestF64Backward(t *testing.T) {
	cases := []struct {
		name string
		fn   func(*tensor.Tensor) (*tensor.Tensor, error)
		want func(x float64) float64
	}{
		{"Sum", autograd.Sum, func(float64) float64 { return 1 }},
		{"Mean", autograd.Mean, func(float64) float64 { return 0.25 }},
		{"AxisSum", func(x *tensor.Tensor) (*tensor.Tensor, error) {
			s, err := autograd.AxisSum(x, []int{1}, false)
			if err != nil {
				return nil, err
			}
			return autograd.Sum(s)
		}, func(float64) float64 { return 1 }},
		{"AxisMean", func(x *tensor.Tensor) (*tensor.Tensor, error) {
			s, err := autograd.AxisMean(x, nil, false)
			if err != nil {
				return nil, err
			}
			return autograd.Sum(s)
		}, func(float64) float64 { return 0.25 }},
		{"Sigmoid", func(x *tensor.Tensor) (*tensor.Tensor, error) {
			y, err := autograd.Sigmoid(x)
			if err != nil {
				return nil, err
			}
			return autograd.Sum(y)
		}, func(x float64) float64 { s := 1 / (1 + math.Exp(-x)); return s * (1 - s) }},
		{"Tanh", func(x *tensor.Tensor) (*tensor.Tensor, error) {
			y, err := autograd.Tanh(x)
			if err != nil {
				return nil, err
			}
			return autograd.Sum(y)
		}, func(x float64) float64 { th := math.Tanh(x); return 1 - th*th }},
	}
	vals := []float64{0.5, -1, 2, 0.25}
	for _, c := range cases {
		x, _ := tensor.New(append([]float64(nil), vals...), []int{2, 2}, device.CPU())
		x.SetRequiresGrad(true)
		y, err := c.fn(x)
		if err != nil {
			t.Fatalf("%s: forward: %v", c.name, err)
		}
		if err := autograd.Backward(y, nil); err != nil {
			t.Fatalf("%s: backward: %v", c.name, err)
		}
		g := tensor.MustSlice[float64](x.Grad())
		for i, v := range vals {
			if math.Abs(g[i]-c.want(v)) > 1e-12 {
				t.Errorf("%s: grad[%d] = %v want %v", c.name, i, g[i], c.want(v))
			}
		}
	}
}

// R11: NLLLoss rejects targets of the wrong length, rank, or dtype.
func TestNLLLossTargetValidation(t *testing.T) {
	logp, _ := tensor.New([]float32{-1, -2, -3, -4}, []int{2, 2}, device.CPU())
	short, _ := tensor.New([]int64{0}, []int{1}, device.CPU())
	long, _ := tensor.New([]int64{0, 1, 1}, []int{3}, device.CPU())
	f32, _ := tensor.New([]float32{0, 1}, []int{2}, device.CPU())
	rank2, _ := tensor.New([]int64{0, 1}, []int{2, 1}, device.CPU())
	oob, _ := tensor.New([]int64{0, 2}, []int{2}, device.CPU())
	for name, tg := range map[string]*tensor.Tensor{"short": short, "long": long, "f32": f32, "rank2": rank2, "oob": oob} {
		if _, err := autograd.NLLLoss(logp, tg); err == nil {
			t.Errorf("%s target: want error", name)
		}
	}
	ok, _ := tensor.New([]int64{0, 1}, []int{2}, device.CPU())
	l, err := autograd.NLLLoss(logp, ok)
	if err != nil {
		t.Fatal(err)
	}
	if got := tensor.MustSlice[float32](l)[0]; got != 2.5 {
		t.Errorf("loss %v want 2.5", got)
	}
}

// R14: a supplied gradOutput must match the root's shape and dtype.
func TestBackwardRejectsMismatchedGradOutput(t *testing.T) {
	leaf, _ := tensor.New([]float32{1, 2}, []int{2}, device.CPU())
	leaf.SetRequiresGrad(true)
	bad, _ := tensor.New([]float32{1}, []int{1}, device.CPU())
	if err := autograd.Backward(leaf, bad); err == nil {
		t.Error("leaf root: want shape error")
	}
	if leaf.Grad() != nil {
		t.Error("leaf root: grad stored despite error")
	}

	x, _ := tensor.New([]float32{1, 2}, []int{2}, device.CPU())
	x.SetRequiresGrad(true)
	y, err := autograd.Sqr(x)
	if err != nil {
		t.Fatal(err)
	}
	if err := autograd.Backward(y, bad); err == nil {
		t.Error("graph root: want shape error")
	}
	bad64, _ := tensor.New([]float64{1, 1}, []int{2}, device.CPU())
	if err := autograd.Backward(y, bad64); err == nil {
		t.Error("graph root: want dtype error")
	}
	good, _ := tensor.New([]float32{1, 1}, []int{2}, device.CPU())
	if err := autograd.Backward(y, good); err != nil {
		t.Fatal(err)
	}
	if g := tensor.MustSlice[float32](x.Grad()); g[0] != 2 || g[1] != 4 {
		t.Errorf("grad %v want [2 4]", g)
	}
}

// R15: overwriting a saved input between forward and backward is reported,
// whether the write goes through the tensor itself or a view of its storage.
func TestBackwardDetectsInPlaceMutation(t *testing.T) {
	three, _ := tensor.New([]float32{3}, []int{1}, device.CPU())

	x, _ := tensor.New([]float32{2}, []int{1}, device.CPU())
	x.SetRequiresGrad(true)
	y, err := autograd.Sqr(x)
	if err != nil {
		t.Fatal(err)
	}
	if err := x.CopyFrom(three); err != nil {
		t.Fatal(err)
	}
	if err := autograd.Backward(y, nil); err == nil || !strings.Contains(err.Error(), "modified in place") {
		t.Errorf("direct write: got %v, want in-place modification error", err)
	}

	x2, _ := tensor.New([]float32{2}, []int{1}, device.CPU())
	x2.SetRequiresGrad(true)
	y2, err := autograd.Sqr(x2)
	if err != nil {
		t.Fatal(err)
	}
	view, err := x2.Reshape(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if view.Storage() != x2.Storage() {
		t.Fatal("Reshape did not alias storage; test needs a view")
	}
	three2, _ := tensor.New([]float32{3}, []int{1, 1}, device.CPU())
	if err := view.Assign(three2); err != nil {
		t.Fatal(err)
	}
	if err := autograd.Backward(y2, nil); err == nil {
		t.Error("aliased write: want in-place modification error")
	}

	// Unmodified graphs still differentiate normally.
	x3, _ := tensor.New([]float32{2}, []int{1}, device.CPU())
	x3.SetRequiresGrad(true)
	y3, err := autograd.Sqr(x3)
	if err != nil {
		t.Fatal(err)
	}
	if err := autograd.Backward(y3, nil); err != nil {
		t.Fatal(err)
	}
	if g := tensor.MustSlice[float32](x3.Grad())[0]; g != 4 {
		t.Errorf("grad %v want 4", g)
	}
}
