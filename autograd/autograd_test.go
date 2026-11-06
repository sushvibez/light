package autograd_test

import (
	"math"
	"testing"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// Simple end-to-end gradient propagation: y = a * b + c; dy/da = b, dy/db = a, dy/dc = 1.
func TestAddMulBackward(t *testing.T) {
	a, _ := tensor.New([]float32{2, 3, 4}, []int{3}, device.CPU())
	b, _ := tensor.New([]float32{5, 6, 7}, []int{3}, device.CPU())
	c, _ := tensor.New([]float32{10, 20, 30}, []int{3}, device.CPU())
	a.SetRequiresGrad(true)
	b.SetRequiresGrad(true)
	c.SetRequiresGrad(true)

	prod, err := autograd.Mul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	y, err := autograd.Add(prod, c)
	if err != nil {
		t.Fatal(err)
	}
	// Reduce to scalar so Backward can default its gradient to ones.
	s, err := autograd.Sum(y)
	if err != nil {
		t.Fatal(err)
	}
	if err := autograd.Backward(s, nil); err != nil {
		t.Fatal(err)
	}

	// With loss = sum(a*b + c), dL/da = b, dL/db = a, dL/dc = 1.
	wantGA := []float32{5, 6, 7}
	wantGB := []float32{2, 3, 4}
	wantGC := []float32{1, 1, 1}
	checkGrad(t, "a", a.Grad(), wantGA)
	checkGrad(t, "b", b.Grad(), wantGB)
	checkGrad(t, "c", c.Grad(), wantGC)
}

// 2-layer MLP forward + gradcheck. Uses autograd exclusively for the forward
// graph, then verifies analytic gradients against finite differences.
func TestGradCheckMLP(t *testing.T) {
	// Network: y = relu(x @ W1 + b1) @ W2 + b2; loss = mean((y - target)^2)
	x, _ := tensor.New([]float32{
		0.5, -1.0, 2.0,
		-0.5, 0.25, 1.5,
	}, []int{2, 3}, device.CPU())

	W1, _ := tensor.New([]float32{
		0.1, -0.2, 0.3, 0.4,
		0.5, 0.6, -0.7, 0.8,
		-0.9, 1.0, 0.1, -0.2,
	}, []int{3, 4}, device.CPU())
	b1, _ := tensor.New([]float32{0.1, -0.1, 0.05, 0.2}, []int{4}, device.CPU())

	W2, _ := tensor.New([]float32{
		0.3, -0.4,
		-0.5, 0.6,
		0.7, -0.8,
		0.9, 0.1,
	}, []int{4, 2}, device.CPU())
	b2, _ := tensor.New([]float32{0.05, -0.05}, []int{2}, device.CPU())

	target, _ := tensor.New([]float32{1, -1, 0.5, 0.25}, []int{2, 2}, device.CPU())

	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		W1, b1, W2, b2 := leaves[0], leaves[1], leaves[2], leaves[3]
		z1, err := autograd.MatMul(x, W1)
		if err != nil {
			return nil, err
		}
		h1, err := autograd.Add(z1, b1) // b1 broadcasts: [4] + [2,4]
		if err != nil {
			return nil, err
		}
		a1, err := autograd.ReLU(h1)
		if err != nil {
			return nil, err
		}
		z2, err := autograd.MatMul(a1, W2)
		if err != nil {
			return nil, err
		}
		y, err := autograd.Add(z2, b2)
		if err != nil {
			return nil, err
		}
		diff, err := autograd.Sub(y, target)
		if err != nil {
			return nil, err
		}
		sq, err := autograd.Mul(diff, diff)
		if err != nil {
			return nil, err
		}
		return autograd.Mean(sq)
	}

	if err := autograd.GradCheck(compute, []*tensor.Tensor{W1, b1, W2, b2}, 1e-2, 5e-3); err != nil {
		t.Fatal(err)
	}
}

// Broadcast-aware gradient: y = a + b with a shape [3,4] and b shape [4]. dL/db
// should sum gradients along axis 0 so its shape matches.
func TestAddBroadcastBackward(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, []int{3, 4}, device.CPU())
	b, _ := tensor.New([]float32{10, 20, 30, 40}, []int{4}, device.CPU())
	a.SetRequiresGrad(true)
	b.SetRequiresGrad(true)

	y, err := autograd.Add(a, b)
	if err != nil {
		t.Fatal(err)
	}
	s, err := autograd.Sum(y)
	if err != nil {
		t.Fatal(err)
	}
	if err := autograd.Backward(s, nil); err != nil {
		t.Fatal(err)
	}

	// dL/da = ones(3,4); dL/db = (3, 3, 3, 3) (summed across batch axis).
	gb := tensor.MustSlice[float32](b.Grad())
	if !approxEqual(gb, []float32{3, 3, 3, 3}) {
		t.Errorf("grad b = %v, want [3 3 3 3]", gb)
	}
	ga := tensor.MustSlice[float32](a.Grad())
	wantGA := make([]float32, 12)
	for i := range wantGA {
		wantGA[i] = 1
	}
	if !approxEqual(ga, wantGA) {
		t.Errorf("grad a = %v, want all 1s", ga)
	}
}

// Non-grad inputs should not cause Backward to fail, and their .Grad() should
// stay nil.
func TestBackwardIgnoresNonRequiresGrad(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	b, _ := tensor.New([]float32{4, 5, 6}, []int{3}, device.CPU())
	// Only a requires grad.
	a.SetRequiresGrad(true)

	prod, _ := autograd.Mul(a, b)
	s, _ := autograd.Sum(prod)
	if err := autograd.Backward(s, nil); err != nil {
		t.Fatal(err)
	}
	ga := tensor.MustSlice[float32](a.Grad())
	if !approxEqual(ga, []float32{4, 5, 6}) {
		t.Errorf("grad a = %v, want [4 5 6]", ga)
	}
	if b.Grad() != nil {
		t.Error("b.Grad() should be nil — b did not requires_grad")
	}
}

// Activations: gradcheck each one individually against its analytical formula.
func TestGradCheckActivations(t *testing.T) {
	cases := []struct {
		name string
		fn   func(*tensor.Tensor) (*tensor.Tensor, error)
	}{
		{"relu", autograd.ReLU},
		{"sigmoid", autograd.Sigmoid},
		{"tanh", autograd.Tanh},
		{"silu", autograd.SiLU},
		{"gelu", autograd.GELU},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			x, _ := tensor.New([]float32{-0.3, 0.5, 1.2, -0.9}, []int{4}, device.CPU())
			compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
				y, err := c.fn(leaves[0])
				if err != nil {
					return nil, err
				}
				return autograd.Sum(y)
			}
			if err := autograd.GradCheck(compute, []*tensor.Tensor{x}, 1e-3, 5e-4); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// Sanity: ensure non-autograd ops still work unmolested.
func TestOpsUnchangedByAutograd(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2}, []int{2}, device.CPU())
	b, _ := tensor.New([]float32{10, 20}, []int{2}, device.CPU())
	// Call ops.Add directly (not autograd.Add) — no graph should be built.
	c, err := ops.Add(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if c.GradFn() != nil {
		t.Error("ops.Add should not attach a gradFn")
	}
	_ = dtype.F32 // keep import stable
}

// ---- helpers ----

func checkGrad(t *testing.T, name string, g *tensor.Tensor, want []float32) {
	t.Helper()
	if g == nil {
		t.Fatalf("%s.Grad() is nil", name)
	}
	got := tensor.MustSlice[float32](g)
	if !approxEqual(got, want) {
		t.Errorf("%s grad = %v, want %v", name, got, want)
	}
}

func approxEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > 1e-4 {
			return false
		}
	}
	return true
}
