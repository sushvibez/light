package nn_test

import (
	"testing"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/tensor"
)

func TestMHABasicShape(t *testing.T) {
	m, err := nn.NewMultiHeadAttention(8, 2, true, false, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	x, _ := tensor.Rand(dtype.F32, []int{4, 8}, device.CPU())
	y, err := m.Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	if y.Rank() != 2 || y.Dim(0) != 4 || y.Dim(1) != 8 {
		t.Errorf("shape = %v", y.Shape())
	}
	if m.NumHeads() != 2 || m.HeadDim() != 4 || m.Hidden() != 8 || !m.Causal() {
		t.Errorf("accessors off")
	}
}

func TestMHAInvalid(t *testing.T) {
	if _, err := nn.NewMultiHeadAttention(10, 3, false, false, device.CPU()); err == nil {
		t.Error("expected divisibility error")
	}
	if _, err := nn.NewMultiHeadAttention(-1, 1, false, false, device.CPU()); err == nil {
		t.Error("expected positive error")
	}
	m, _ := nn.NewMultiHeadAttention(8, 2, false, false, device.CPU())
	bad1, _ := tensor.Zeros(dtype.F32, []int{8}, device.CPU())
	if _, err := m.Forward(bad1); err == nil {
		t.Error("expected rank error")
	}
	bad2, _ := tensor.Zeros(dtype.F32, []int{4, 16}, device.CPU())
	if _, err := m.Forward(bad2); err == nil {
		t.Error("expected hidden-mismatch error")
	}
}

func TestMHACausalMaskAndCache(t *testing.T) {
	m, _ := nn.NewMultiHeadAttention(8, 2, true, false, device.CPU())
	x, _ := tensor.Rand(dtype.F32, []int{4, 8}, device.CPU())
	if _, err := m.Forward(x); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Forward(x); err != nil {
		t.Fatal(err)
	} // cache hit
	x2, _ := tensor.Rand(dtype.F32, []int{6, 8}, device.CPU())
	if _, err := m.Forward(x2); err != nil {
		t.Fatal(err)
	} // cache miss
	if len(m.Parameters()) != 4 {
		t.Errorf("params = %d", len(m.Parameters()))
	}
}

func TestMHANonCausal(t *testing.T) {
	m, _ := nn.NewMultiHeadAttention(8, 2, false, false, device.CPU())
	if m.Causal() {
		t.Error("Causal() should be false")
	}
	x, _ := tensor.Rand(dtype.F32, []int{4, 8}, device.CPU())
	if _, err := m.Forward(x); err != nil {
		t.Fatal(err)
	}
}

func TestMHAMultipleHeadDims(t *testing.T) {
	// Different (hidden, numHeads) combos.
	for _, c := range []struct{ hidden, heads int }{
		{4, 1}, {4, 2}, {4, 4}, {12, 3}, {16, 4},
	} {
		m, err := nn.NewMultiHeadAttention(c.hidden, c.heads, false, false, device.CPU())
		if err != nil {
			t.Errorf("(%d,%d) err: %v", c.hidden, c.heads, err)
			continue
		}
		x, _ := tensor.Rand(dtype.F32, []int{3, c.hidden}, device.CPU())
		y, err := m.Forward(x)
		if err != nil {
			t.Errorf("(%d,%d) forward err: %v", c.hidden, c.heads, err)
			continue
		}
		if y.Dim(0) != 3 || y.Dim(1) != c.hidden {
			t.Errorf("(%d,%d) shape = %v", c.hidden, c.heads, y.Shape())
		}
	}
}
