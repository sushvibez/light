package ops_test

import (
	"math"
	"slices"
	"testing"

	"light/device"
	"light/ops"
	"light/tensor"
)

func TestSumAll(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device.CPU())
	got, err := ops.Sum(a, ops.ReduceOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if got.Rank() != 0 {
		t.Errorf("expected scalar, got shape %v", got.Shape())
	}
	s := tensor.MustSlice[float32](got)
	if s[0] != 21 {
		t.Errorf("got %v, want 21", s[0])
	}
}

func TestSumAxis(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device.CPU())
	// Sum axis 0 → shape (3): [1+4, 2+5, 3+6]
	got, err := ops.Sum(a, ops.ReduceOptions{Axes: []int{0}})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(tensor.MustSlice[float32](got), []float32{5, 7, 9}) {
		t.Errorf("got %v", tensor.MustSlice[float32](got))
	}

	// Sum axis 1 → shape (2): [1+2+3, 4+5+6]
	got, _ = ops.Sum(a, ops.ReduceOptions{Axes: []int{1}})
	if !slices.Equal(tensor.MustSlice[float32](got), []float32{6, 15}) {
		t.Errorf("got %v", tensor.MustSlice[float32](got))
	}
}

func TestSumKeepDims(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device.CPU())
	got, err := ops.Sum(a, ops.ReduceOptions{Axes: []int{1}, KeepDims: true})
	if err != nil {
		t.Fatal(err)
	}
	if got.Rank() != 2 || got.Dim(0) != 2 || got.Dim(1) != 1 {
		t.Errorf("keepdims shape = %v", got.Shape())
	}
}

func TestMean(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3, 4}, []int{4}, device.CPU())
	got, err := ops.Mean(a, ops.ReduceOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if tensor.MustSlice[float32](got)[0] != 2.5 {
		t.Errorf("mean = %v, want 2.5", tensor.MustSlice[float32](got)[0])
	}
}

func TestMaxMin(t *testing.T) {
	a, _ := tensor.New([]float32{3, 1, 4, 1, 5, 9, 2, 6}, []int{2, 4}, device.CPU())
	got, _ := ops.Max(a, ops.ReduceOptions{Axes: []int{1}})
	if !slices.Equal(tensor.MustSlice[float32](got), []float32{4, 9}) {
		t.Errorf("max got %v", tensor.MustSlice[float32](got))
	}
	got, _ = ops.Min(a, ops.ReduceOptions{Axes: []int{1}})
	if !slices.Equal(tensor.MustSlice[float32](got), []float32{1, 2}) {
		t.Errorf("min got %v", tensor.MustSlice[float32](got))
	}
}

func TestSumNegativeAxis(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device.CPU())
	got, err := ops.Sum(a, ops.ReduceOptions{Axes: []int{-1}})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(tensor.MustSlice[float32](got), []float32{6, 15}) {
		t.Errorf("got %v", tensor.MustSlice[float32](got))
	}
}

func TestReduceErrors(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	if _, err := ops.Sum(a, ops.ReduceOptions{Axes: []int{5}}); err == nil {
		t.Error("expected out-of-range error")
	}
	if _, err := ops.Sum(a, ops.ReduceOptions{Axes: []int{0, 0}}); err == nil {
		t.Error("expected duplicate error")
	}
}

func TestSumI32(t *testing.T) {
	a, _ := tensor.New([]int32{1, 2, 3, 4}, []int{4}, device.CPU())
	got, _ := ops.Sum(a, ops.ReduceOptions{})
	if tensor.MustSlice[int32](got)[0] != 10 {
		t.Errorf("got %v", tensor.MustSlice[int32](got)[0])
	}
}

func TestSoftmax(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	got, err := ops.Softmax(a, 0)
	if err != nil {
		t.Fatal(err)
	}
	s := tensor.MustSlice[float32](got)
	// Expected: exp(1),exp(2),exp(3) normalized.
	e1 := math.Exp(1)
	e2 := math.Exp(2)
	e3 := math.Exp(3)
	total := e1 + e2 + e3
	want := []float64{e1 / total, e2 / total, e3 / total}
	sum := float64(0)
	for i, v := range s {
		if math.Abs(float64(v)-want[i]) > 1e-5 {
			t.Errorf("softmax[%d] = %v, want %v", i, v, want[i])
		}
		sum += float64(v)
	}
	if math.Abs(sum-1.0) > 1e-5 {
		t.Errorf("softmax sum = %v, want 1.0", sum)
	}
}

func TestSoftmax2D(t *testing.T) {
	// Along last axis: each row normalizes independently.
	a, _ := tensor.New([]float32{1, 2, 3, 1, 2, 3}, []int{2, 3}, device.CPU())
	got, err := ops.Softmax(a, -1)
	if err != nil {
		t.Fatal(err)
	}
	s := tensor.MustSlice[float32](got)
	// Both rows should match.
	for i := range 3 {
		if math.Abs(float64(s[i]-s[i+3])) > 1e-6 {
			t.Errorf("rows should match: %v", s)
		}
	}
	// Each row sums to 1.
	for i := range 2 {
		sum := s[i*3] + s[i*3+1] + s[i*3+2]
		if math.Abs(float64(sum)-1.0) > 1e-5 {
			t.Errorf("row %d sum = %v", i, sum)
		}
	}
}

func TestSoftmaxNumericStability(t *testing.T) {
	// Very large values should not overflow the internal exp.
	a, _ := tensor.New([]float32{1000, 1001, 1002}, []int{3}, device.CPU())
	got, err := ops.Softmax(a, 0)
	if err != nil {
		t.Fatal(err)
	}
	s := tensor.MustSlice[float32](got)
	sum := float32(0)
	for _, v := range s {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("softmax produced NaN/Inf: %v", s)
		}
		sum += v
	}
	if math.Abs(float64(sum)-1.0) > 1e-4 {
		t.Errorf("sum = %v", sum)
	}
}
