package ops_test

import (
	"math"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

func TestCastF32ToF64(t *testing.T) {
	a, _ := tensor.New([]float32{1.5, 2.5, 3.5}, []int{3}, device.CPU())
	b, err := ops.Cast(a, dtype.F64)
	if err != nil {
		t.Fatal(err)
	}
	if b.DType() != dtype.F64 {
		t.Errorf("dtype = %v", b.DType())
	}
	vals := tensor.MustSlice[float64](b)
	want := []float64{1.5, 2.5, 3.5}
	for i := range want {
		if math.Abs(vals[i]-want[i]) > 1e-6 {
			t.Errorf("[%d] got %v want %v", i, vals[i], want[i])
		}
	}
}

func TestCastF32ToI32(t *testing.T) {
	a, _ := tensor.New([]float32{1.9, -2.3, 0.7, 5.5}, []int{4}, device.CPU())
	b, err := ops.Cast(a, dtype.I32)
	if err != nil {
		t.Fatal(err)
	}
	vals := tensor.MustSlice[int32](b)
	// Truncation toward zero (Go default)
	want := []int32{1, -2, 0, 5}
	if !slices.Equal(vals, want) {
		t.Errorf("got %v, want %v", vals, want)
	}
}

func TestCastF32ToF16(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3, 4}, []int{4}, device.CPU())
	b, err := ops.Cast(a, dtype.F16)
	if err != nil {
		t.Fatal(err)
	}
	hs, err := tensor.ToHalfSlice(b)
	if err != nil {
		t.Fatal(err)
	}
	for i, h := range hs {
		if math.Abs(float64(h.ToF32())-float64(i+1)) > 1e-2 {
			t.Errorf("[%d] = %v, want %v", i, h.ToF32(), float32(i+1))
		}
	}
}

func TestCastNoOp(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	b, err := ops.Cast(a, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(tensor.MustSlice[float32](b), []float32{1, 2, 3}) {
		t.Error("no-op cast should preserve values")
	}
}

func TestLogSoftmax(t *testing.T) {
	// ls(x) should be row-normalised such that sum(exp(ls)) = 1.
	a, _ := tensor.New([]float32{1, 2, 3, 1, 2, 3}, []int{2, 3}, device.CPU())
	y, err := ops.LogSoftmax(a, -1)
	if err != nil {
		t.Fatal(err)
	}
	vals := tensor.MustSlice[float32](y)
	for row := range 2 {
		sumExp := float64(0)
		for j := range 3 {
			sumExp += math.Exp(float64(vals[row*3+j]))
		}
		if math.Abs(sumExp-1.0) > 1e-5 {
			t.Errorf("row %d: sum(exp(ls)) = %v, want 1.0", row, sumExp)
		}
	}
}

func TestLogSoftmaxStability(t *testing.T) {
	// Very large values shouldn't produce NaN/Inf.
	a, _ := tensor.New([]float32{1000, 1001, 1002}, []int{3}, device.CPU())
	y, _ := ops.LogSoftmax(a, 0)
	vals := tensor.MustSlice[float32](y)
	for _, v := range vals {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("got NaN/Inf: %v", vals)
			break
		}
	}
}

func TestArgmax(t *testing.T) {
	a, _ := tensor.New([]float32{1, 5, 2, 9, 3, 4, 8, 7, 0}, []int{3, 3}, device.CPU())
	// argmax along axis 1 → per-row max index: [1, 0, 0]
	y, err := ops.Argmax(a, 1, false)
	if err != nil {
		t.Fatal(err)
	}
	if y.DType() != dtype.I64 {
		t.Errorf("dtype = %v", y.DType())
	}
	vals := tensor.MustSlice[int64](y)
	want := []int64{1, 0, 0}
	if !slices.Equal(vals, want) {
		t.Errorf("got %v, want %v", vals, want)
	}
}

func TestArgmin(t *testing.T) {
	a, _ := tensor.New([]float32{1, 5, 2, 9, 3, 4}, []int{2, 3}, device.CPU())
	y, _ := ops.Argmin(a, -1, false)
	vals := tensor.MustSlice[int64](y)
	want := []int64{0, 1}
	if !slices.Equal(vals, want) {
		t.Errorf("got %v want %v", vals, want)
	}
}

func TestArgmaxKeepdims(t *testing.T) {
	a, _ := tensor.New([]float32{1, 5, 2, 9, 3, 4}, []int{2, 3}, device.CPU())
	y, _ := ops.Argmax(a, 1, true)
	if y.Rank() != 2 || y.Dim(0) != 2 || y.Dim(1) != 1 {
		t.Errorf("keepdims shape = %v", y.Shape())
	}
}

func TestConcatAxis0(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3, 4}, []int{2, 2}, device.CPU())
	b, _ := tensor.New([]float32{5, 6, 7, 8, 9, 10}, []int{3, 2}, device.CPU())
	c, err := ops.Concat([]*tensor.Tensor{a, b}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if c.Dim(0) != 5 || c.Dim(1) != 2 {
		t.Errorf("shape = %v", c.Shape())
	}
	want := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	if !slices.Equal(tensor.MustSlice[float32](c), want) {
		t.Errorf("got %v", tensor.MustSlice[float32](c))
	}
}

func TestConcatAxis1(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3, 4}, []int{2, 2}, device.CPU())
	b, _ := tensor.New([]float32{10, 20, 30, 40, 50, 60}, []int{2, 3}, device.CPU())
	c, err := ops.Concat([]*tensor.Tensor{a, b}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if c.Dim(0) != 2 || c.Dim(1) != 5 {
		t.Errorf("shape = %v", c.Shape())
	}
	// Row 0: a[0] + b[0] = [1,2,10,20,30]; Row 1: [3,4,40,50,60]
	want := []float32{1, 2, 10, 20, 30, 3, 4, 40, 50, 60}
	if !slices.Equal(tensor.MustSlice[float32](c), want) {
		t.Errorf("got %v, want %v", tensor.MustSlice[float32](c), want)
	}
}

func TestConcatErrors(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3, 4}, []int{2, 2}, device.CPU())
	b, _ := tensor.New([]float32{1, 2}, []int{2}, device.CPU()) // wrong rank
	if _, err := ops.Concat([]*tensor.Tensor{a, b}, 0); err == nil {
		t.Error("expected rank mismatch")
	}
	// Empty
	if _, err := ops.Concat(nil, 0); err == nil {
		t.Error("expected empty-list error")
	}
	// dtype mismatch
	bbad, _ := tensor.New([]float64{1, 2, 3, 4}, []int{2, 2}, device.CPU())
	if _, err := ops.Concat([]*tensor.Tensor{a, bbad}, 0); err == nil {
		t.Error("expected dtype mismatch")
	}
	// non-cat-axis dim mismatch
	bbad2, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{3, 2}, device.CPU())
	if _, err := ops.Concat([]*tensor.Tensor{a, bbad2}, 1); err == nil {
		t.Error("expected non-cat-axis mismatch")
	}
}

func TestStack(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2}, []int{2}, device.CPU())
	b, _ := tensor.New([]float32{3, 4}, []int{2}, device.CPU())
	c, _ := tensor.New([]float32{5, 6}, []int{2}, device.CPU())

	// Stack along axis 0 → (3, 2)
	s0, err := ops.Stack([]*tensor.Tensor{a, b, c}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if s0.Dim(0) != 3 || s0.Dim(1) != 2 {
		t.Errorf("axis=0 shape = %v", s0.Shape())
	}
	want := []float32{1, 2, 3, 4, 5, 6}
	if !slices.Equal(tensor.MustSlice[float32](s0), want) {
		t.Errorf("axis=0 got %v", tensor.MustSlice[float32](s0))
	}
	// Stack along axis 1 → (2, 3)
	s1, _ := ops.Stack([]*tensor.Tensor{a, b, c}, 1)
	if s1.Dim(0) != 2 || s1.Dim(1) != 3 {
		t.Errorf("axis=1 shape = %v", s1.Shape())
	}
	// row 0 = [a[0], b[0], c[0]] = [1, 3, 5]; row 1 = [2, 4, 6]
	want1 := []float32{1, 3, 5, 2, 4, 6}
	if !slices.Equal(tensor.MustSlice[float32](s1), want1) {
		t.Errorf("axis=1 got %v", tensor.MustSlice[float32](s1))
	}
}

func TestStackErrors(t *testing.T) {
	if _, err := ops.Stack(nil, 0); err == nil {
		t.Error("expected empty error")
	}
	a, _ := tensor.New([]float32{1, 2}, []int{2}, device.CPU())
	if _, err := ops.Stack([]*tensor.Tensor{a}, 5); err == nil {
		t.Error("expected axis out of range")
	}
}

func TestRandN(t *testing.T) {
	tt, err := tensor.RandN(dtype.F32, []int{10000}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	vals := tensor.MustSlice[float32](tt)
	// Mean should be near 0, variance near 1.
	var sum, sum2 float64
	for _, v := range vals {
		sum += float64(v)
		sum2 += float64(v) * float64(v)
	}
	n := float64(len(vals))
	mean := sum / n
	variance := sum2/n - mean*mean
	if math.Abs(mean) > 0.05 {
		t.Errorf("mean = %v, expected near 0", mean)
	}
	if math.Abs(variance-1.0) > 0.1 {
		t.Errorf("variance = %v, expected near 1", variance)
	}
}

func TestRandNUnsupportedDType(t *testing.T) {
	if _, err := tensor.RandN(dtype.I32, []int{5}, device.CPU()); err == nil {
		t.Error("expected unsupported dtype error")
	}
}

func TestRandNF64(t *testing.T) {
	tt, err := tensor.RandN(dtype.F64, []int{100}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if tt.DType() != dtype.F64 {
		t.Errorf("dtype = %v", tt.DType())
	}
}
