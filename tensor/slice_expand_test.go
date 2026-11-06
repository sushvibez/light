package tensor_test

import (
	"slices"
	"testing"

	"light/device"
	"light/dtype"
	"light/tensor"
)

func TestSlice2D(t *testing.T) {
	// (3, 4) → slice [1:3, 1:3] → (2, 2)
	tt, _ := tensor.New([]float32{
		0, 1, 2, 3,
		4, 5, 6, 7,
		8, 9, 10, 11,
	}, []int{3, 4}, device.CPU())
	s, err := tt.Slice([]int{1, 1}, []int{3, 3})
	if err != nil {
		t.Fatal(err)
	}
	if s.Dim(0) != 2 || s.Dim(1) != 2 {
		t.Errorf("shape = %v", s.Shape())
	}
	c, _ := s.Contiguous()
	want := []float32{5, 6, 9, 10}
	if !slices.Equal(tensor.MustSlice[float32](c), want) {
		t.Errorf("got %v, want %v", tensor.MustSlice[float32](c), want)
	}
}

func TestSliceNoOpAxis(t *testing.T) {
	tt, _ := tensor.New([]float32{0, 1, 2, 3}, []int{2, 2}, device.CPU())
	// 0..0 on first axis means "no-op", take all; 0..1 on second gives col 0.
	s, err := tt.Slice([]int{0, 0}, []int{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if s.Dim(0) != 2 || s.Dim(1) != 1 {
		t.Errorf("shape = %v, want (2, 1)", s.Shape())
	}
	c, _ := s.Contiguous()
	if !slices.Equal(tensor.MustSlice[float32](c), []float32{0, 2}) {
		t.Errorf("got %v", tensor.MustSlice[float32](c))
	}
}

func TestSliceErrors(t *testing.T) {
	tt, _ := tensor.New([]float32{0, 1, 2, 3}, []int{2, 2}, device.CPU())
	if _, err := tt.Slice([]int{0}, []int{1}); err == nil {
		t.Error("expected rank-mismatch error")
	}
	if _, err := tt.Slice([]int{1, 0}, []int{0, 0}); err == nil {
		t.Error("expected stop<=start error")
	}
}

func TestExpand(t *testing.T) {
	// (1, 3) → expand to (4, 3)
	tt, _ := tensor.New([]float32{10, 20, 30}, []int{1, 3}, device.CPU())
	e, err := tt.Expand(4, 3)
	if err != nil {
		t.Fatal(err)
	}
	if e.Dim(0) != 4 || e.Dim(1) != 3 {
		t.Errorf("shape = %v", e.Shape())
	}
	c, _ := e.Contiguous()
	want := []float32{10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30}
	if !slices.Equal(tensor.MustSlice[float32](c), want) {
		t.Errorf("got %v, want %v", tensor.MustSlice[float32](c), want)
	}
}

func TestExpandKeepDim(t *testing.T) {
	// -1 keeps current size
	tt, _ := tensor.Zeros(dtype.F32, []int{1, 5}, device.CPU())
	e, err := tt.Expand(3, -1)
	if err != nil {
		t.Fatal(err)
	}
	if e.Dim(0) != 3 || e.Dim(1) != 5 {
		t.Errorf("shape = %v", e.Shape())
	}
}

func TestExpandErrors(t *testing.T) {
	tt, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	if _, err := tt.Expand(3, 4); err == nil {
		t.Error("expected rank mismatch error")
	}
	tt2, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	if _, err := tt2.Expand(5); err == nil {
		t.Error("expected size-mismatch error (3 cannot expand to 5)")
	}
}
