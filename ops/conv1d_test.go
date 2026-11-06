package ops_test

import (
	"slices"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

func TestConv1dIdentity(t *testing.T) {
	// 1×1 kernel passes input through. input: (1, 1, 5), weight: (1, 1, 1) = [1].
	in, _ := tensor.New([]float32{1, 2, 3, 4, 5}, []int{1, 1, 5}, device.CPU())
	w, _ := tensor.New([]float32{1}, []int{1, 1, 1}, device.CPU())
	out, err := ops.Conv1d(in, w, nil, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(tensor.MustSlice[float32](out), []float32{1, 2, 3, 4, 5}) {
		t.Errorf("got %v", tensor.MustSlice[float32](out))
	}
}

func TestConv1dSum3(t *testing.T) {
	// Kernel of size 3 all-ones on input [1..5] with stride=1, padding=0 → output
	// length = (5 - 3)/1 + 1 = 3. Values: [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12].
	in, _ := tensor.New([]float32{1, 2, 3, 4, 5}, []int{1, 1, 5}, device.CPU())
	w, _ := tensor.Ones(dtype.F32, []int{1, 1, 3}, device.CPU())
	out, _ := ops.Conv1d(in, w, nil, 1, 0)
	if !slices.Equal(tensor.MustSlice[float32](out), []float32{6, 9, 12}) {
		t.Errorf("got %v", tensor.MustSlice[float32](out))
	}
}

func TestConv1dErrors(t *testing.T) {
	// Wrong rank
	in, _ := tensor.Zeros(dtype.F32, []int{1, 3}, device.CPU())
	w, _ := tensor.Zeros(dtype.F32, []int{1, 1, 1}, device.CPU())
	if _, err := ops.Conv1d(in, w, nil, 1, 0); err == nil {
		t.Error("expected rank error")
	}
	// Cin mismatch
	in3, _ := tensor.Zeros(dtype.F32, []int{1, 2, 5}, device.CPU())
	wBad, _ := tensor.Zeros(dtype.F32, []int{1, 5, 3}, device.CPU())
	if _, err := ops.Conv1d(in3, wBad, nil, 1, 0); err == nil {
		t.Error("expected Cin mismatch")
	}
}

func TestUpsampleNearest(t *testing.T) {
	// (1, 1, 2, 2) input scaled by (2, 2) → (1, 1, 4, 4).
	in, _ := tensor.New([]float32{1, 2, 3, 4}, []int{1, 1, 2, 2}, device.CPU())
	out, err := ops.UpsampleNearest(in, [2]int{2, 2})
	if err != nil {
		t.Fatal(err)
	}
	if out.Dim(2) != 4 || out.Dim(3) != 4 {
		t.Errorf("shape = %v", out.Shape())
	}
	// Each input value should be replicated 2x2 times.
	want := []float32{
		1, 1, 2, 2,
		1, 1, 2, 2,
		3, 3, 4, 4,
		3, 3, 4, 4,
	}
	if !slices.Equal(tensor.MustSlice[float32](out), want) {
		t.Errorf("got %v, want %v", tensor.MustSlice[float32](out), want)
	}
}
