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

func TestLinspace(t *testing.T) {
	out, err := tensor.Linspace(0, 10, 5, dtype.F32, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{0, 2.5, 5, 7.5, 10}
	vals := tensor.MustSlice[float32](out)
	for i := range want {
		if math.Abs(float64(vals[i]-want[i])) > 1e-5 {
			t.Errorf("[%d] = %v, want %v", i, vals[i], want[i])
		}
	}
}

func TestLinspaceEdgeCases(t *testing.T) {
	// n=1 → single value at start.
	out, _ := tensor.Linspace(5, 10, 1, dtype.F32, device.CPU())
	if tensor.MustSlice[float32](out)[0] != 5 {
		t.Error("n=1 should give start")
	}
	// n=0 → empty.
	out, _ = tensor.Linspace(0, 10, 0, dtype.F32, device.CPU())
	if out.NumElements() != 0 {
		t.Error("n=0 should be empty")
	}
	// Unsupported dtype.
	if _, err := tensor.Linspace(0, 1, 3, dtype.I32, device.CPU()); err == nil {
		t.Error("expected dtype error")
	}
}

func TestEye(t *testing.T) {
	e, err := tensor.Eye(3, dtype.F32, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 0, 0, 0, 1, 0, 0, 0, 1}
	if !slices.Equal(tensor.MustSlice[float32](e), want) {
		t.Errorf("got %v", tensor.MustSlice[float32](e))
	}
}

func TestTrilTriu(t *testing.T) {
	// (3, 3) matrix of ones.
	a, _ := tensor.Ones(dtype.F32, []int{3, 3}, device.CPU())
	lower, err := tensor.Tril(a, 0)
	if err != nil {
		t.Fatal(err)
	}
	// Tril keeps j <= i.
	want := []float32{1, 0, 0, 1, 1, 0, 1, 1, 1}
	if !slices.Equal(tensor.MustSlice[float32](lower), want) {
		t.Errorf("Tril: %v", tensor.MustSlice[float32](lower))
	}
	upper, _ := tensor.Triu(a, 0)
	want = []float32{1, 1, 1, 0, 1, 1, 0, 0, 1}
	if !slices.Equal(tensor.MustSlice[float32](upper), want) {
		t.Errorf("Triu: %v", tensor.MustSlice[float32](upper))
	}
}

func TestMeshgrid(t *testing.T) {
	x, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	y, _ := tensor.New([]float32{10, 20}, []int{2}, device.CPU())
	gx, gy, err := tensor.Meshgrid(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if gx.Dim(0) != 2 || gx.Dim(1) != 3 {
		t.Errorf("gx shape = %v", gx.Shape())
	}
	// gx should be [[1,2,3],[1,2,3]]; gy should be [[10,10,10],[20,20,20]]
	if !slices.Equal(tensor.MustSlice[float32](gx), []float32{1, 2, 3, 1, 2, 3}) {
		t.Errorf("gx = %v", tensor.MustSlice[float32](gx))
	}
	if !slices.Equal(tensor.MustSlice[float32](gy), []float32{10, 10, 10, 20, 20, 20}) {
		t.Errorf("gy = %v", tensor.MustSlice[float32](gy))
	}
}

func TestCumsum(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3, 4, 5}, []int{5}, device.CPU())
	out, err := ops.Cumsum(a, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 3, 6, 10, 15}
	if !slices.Equal(tensor.MustSlice[float32](out), want) {
		t.Errorf("got %v", tensor.MustSlice[float32](out))
	}
}

func TestCumsumAxis1(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device.CPU())
	out, _ := ops.Cumsum(a, 1)
	want := []float32{1, 3, 6, 4, 9, 15}
	if !slices.Equal(tensor.MustSlice[float32](out), want) {
		t.Errorf("got %v", tensor.MustSlice[float32](out))
	}
}

func TestCumprod(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3, 4}, []int{4}, device.CPU())
	out, _ := ops.Cumprod(a, 0)
	want := []float32{1, 2, 6, 24}
	if !slices.Equal(tensor.MustSlice[float32](out), want) {
		t.Errorf("got %v", tensor.MustSlice[float32](out))
	}
}

func TestTopK(t *testing.T) {
	a, _ := tensor.New([]float32{3, 1, 4, 1, 5, 9, 2, 6}, []int{8}, device.CPU())
	vals, idx, err := ops.TopK(a, 3, true)
	if err != nil {
		t.Fatal(err)
	}
	// Top-3 largest: 9, 6, 5 at indices 5, 7, 4.
	wantV := []float32{9, 6, 5}
	wantI := []int64{5, 7, 4}
	if !slices.Equal(tensor.MustSlice[float32](vals), wantV) {
		t.Errorf("vals = %v, want %v", tensor.MustSlice[float32](vals), wantV)
	}
	if !slices.Equal(tensor.MustSlice[int64](idx), wantI) {
		t.Errorf("idx = %v, want %v", tensor.MustSlice[int64](idx), wantI)
	}
}

func TestTopKSmallest(t *testing.T) {
	a, _ := tensor.New([]float32{3, 1, 4, 1, 5}, []int{5}, device.CPU())
	vals, _, err := ops.TopK(a, 2, false)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(tensor.MustSlice[float32](vals), []float32{1, 1}) {
		t.Errorf("got %v, want [1, 1]", tensor.MustSlice[float32](vals))
	}
}

func TestTopK2D(t *testing.T) {
	a, _ := tensor.New([]float32{3, 1, 4, 1, 5, 9, 2, 6}, []int{2, 4}, device.CPU())
	vals, idx, _ := ops.TopK(a, 2, true)
	if vals.Dim(0) != 2 || vals.Dim(1) != 2 {
		t.Errorf("shape = %v", vals.Shape())
	}
	// Row 0 top 2: 4, 3 at [2, 0]. Row 1 top 2: 9, 6 at [1, 3].
	wantV := []float32{4, 3, 9, 6}
	wantI := []int64{2, 0, 1, 3}
	if !slices.Equal(tensor.MustSlice[float32](vals), wantV) {
		t.Errorf("vals = %v", tensor.MustSlice[float32](vals))
	}
	if !slices.Equal(tensor.MustSlice[int64](idx), wantI) {
		t.Errorf("idx = %v", tensor.MustSlice[int64](idx))
	}
}

func TestDropoutTraining(t *testing.T) {
	// With p=0, dropout is identity.
	a, _ := tensor.Ones(dtype.F32, []int{1000}, device.CPU())
	out, _ := ops.Dropout(a, 0, true)
	if out != a {
		t.Error("p=0 should be identity")
	}
	// With p=0.5 and training=true, approximately half zeroed.
	out, err := ops.Dropout(a, 0.5, true)
	if err != nil {
		t.Fatal(err)
	}
	vals := tensor.MustSlice[float32](out)
	zeros := 0
	for _, v := range vals {
		if v == 0 {
			zeros++
		} else if math.Abs(float64(v)-2.0) > 1e-5 {
			t.Errorf("non-zero value should be 2.0 (1/(1-0.5)): %v", v)
		}
	}
	if zeros < 400 || zeros > 600 {
		t.Errorf("expected ~500 zeros, got %d", zeros)
	}
}

func TestDropoutEval(t *testing.T) {
	a, _ := tensor.Ones(dtype.F32, []int{10}, device.CPU())
	out, _ := ops.Dropout(a, 0.5, false)
	if out != a {
		t.Error("training=false should return input unchanged")
	}
}

func TestRoPE(t *testing.T) {
	// (1 batch, seqLen=2, headDim=4) with simple input.
	x, _ := tensor.New([]float32{
		1, 0, 0, 0, // pos 0, x = [1,0,0,0]
		1, 0, 0, 0, // pos 1, x = [1,0,0,0]
	}, []int{2, 4}, device.CPU())
	pos, _ := tensor.New([]int64{0, 1}, []int{2}, device.CPU())
	out, err := ops.RoPE(x, pos, 10000)
	if err != nil {
		t.Fatal(err)
	}
	vals := tensor.MustSlice[float32](out)
	// At pos=0, all thetas are 0 → cos=1, sin=0. Output == input.
	for i := range 4 {
		if math.Abs(float64(vals[i]-[]float32{1, 0, 0, 0}[i])) > 1e-5 {
			t.Errorf("pos=0 [%d] = %v", i, vals[i])
		}
	}
	// At pos=1, theta_0 = 1*10000^0 = 1. Rotate-half pair (0, 2):
	// dim 0 = cos(1), dim 2 = sin(1); dims 1 and 3 (pair 1, both zero) stay 0.
	if math.Abs(float64(vals[4])-math.Cos(1)) > 1e-5 {
		t.Errorf("pos=1 dim 0 = %v, want cos(1)=%v", vals[4], math.Cos(1))
	}
	if math.Abs(float64(vals[6])-math.Sin(1)) > 1e-5 {
		t.Errorf("pos=1 dim 2 = %v, want sin(1)=%v", vals[6], math.Sin(1))
	}
	if vals[5] != 0 || vals[7] != 0 {
		t.Errorf("pos=1 dims 1,3 = %v, %v, want 0", vals[5], vals[7])
	}
}

func TestRoPEErrors(t *testing.T) {
	x, _ := tensor.Zeros(dtype.F32, []int{2, 3}, device.CPU())
	pos, _ := tensor.New([]int64{0, 1}, []int{2}, device.CPU())
	if _, err := ops.RoPE(x, pos, 10000); err == nil {
		t.Error("expected odd-headDim error")
	}
}
