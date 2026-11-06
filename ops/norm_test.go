package ops_test

import (
	"math"
	"testing"

	"light/device"
	"light/ops"
	"light/tensor"
)

func TestRMSNormBasic(t *testing.T) {
	// x = [1,2,3]; sumSq = 14; mean = 14/3 = 4.667; rms = sqrt(4.667+1e-6) ≈ 2.160
	// y = [1/2.160, 2/2.160, 3/2.160] * weight (all-ones) ≈ [0.463, 0.926, 1.389]
	x, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	w, _ := tensor.Ones(2 /*dtype.F32*/, []int{3}, device.CPU())
	y, err := ops.RMSNorm(x, w, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](y)
	// rms = sqrt(14/3 + 1e-6)
	rms := float32(math.Sqrt(14.0/3.0 + 1e-6))
	want := []float32{1 / rms, 2 / rms, 3 / rms}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Errorf("[%d] got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestRMSNormWithWeight(t *testing.T) {
	// weight scales each dim
	x, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	w, _ := tensor.New([]float32{1, 0, 2}, []int{3}, device.CPU())
	y, _ := ops.RMSNorm(x, w, 0)
	got := tensor.MustSlice[float32](y)
	if got[1] != 0 {
		t.Errorf("weight=0 element should zero: %v", got)
	}
	// got[0]*1 and got[2]*2 should preserve the 1:6 ratio
	// Actually x[0]*w[0]/rms vs x[2]*w[2]/rms = 1/rms vs 6/rms
	if math.Abs(float64(got[2]/got[0])-6.0) > 1e-4 {
		t.Errorf("ratio wrong: got %v / %v", got[2], got[0])
	}
}

func TestLayerNormBasic(t *testing.T) {
	// After layer-norm, each row has mean 0 and std 1 (ignoring epsilon).
	x, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device.CPU())
	g, _ := tensor.Ones(2, []int{3}, device.CPU())
	b, _ := tensor.Zeros(2, []int{3}, device.CPU())
	y, err := ops.LayerNorm(x, g, b, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](y)
	// Row 0: [1,2,3] → mean 2, var = 2/3 (population var), std = sqrt(2/3)
	// Normalised: [(1-2)/std, 0, 1/std] ≈ [-1.224, 0, 1.224]
	std := float32(math.Sqrt(2.0/3.0 + 1e-5))
	want := []float32{-1 / std, 0, 1 / std, -1 / std, 0, 1 / std}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-3 {
			t.Errorf("[%d] got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestIndexSelect(t *testing.T) {
	// src: (3, 2) with row 0 = [1,2], row 1 = [3,4], row 2 = [5,6]
	src, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{3, 2}, device.CPU())
	idx, _ := tensor.New([]int64{2, 0, 2}, []int{3}, device.CPU())
	out, err := ops.IndexSelect(src, idx)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dim(0) != 3 || out.Dim(1) != 2 {
		t.Errorf("shape: %v", out.Shape())
	}
	want := []float32{5, 6, 1, 2, 5, 6}
	got := tensor.MustSlice[float32](out)
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("[%d] got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestIndexSelectOutOfRange(t *testing.T) {
	src, _ := tensor.New([]float32{1, 2, 3, 4}, []int{2, 2}, device.CPU())
	idx, _ := tensor.New([]int64{0, 5}, []int{2}, device.CPU())
	if _, err := ops.IndexSelect(src, idx); err == nil {
		t.Error("expected out-of-range error")
	}
}

func TestIndexSelectBadRank(t *testing.T) {
	src, _ := tensor.New([]float32{1, 2, 3, 4}, []int{2, 2}, device.CPU())
	idx, _ := tensor.New([]int64{0, 1, 0, 1}, []int{2, 2}, device.CPU()) // not 1-D
	if _, err := ops.IndexSelect(src, idx); err == nil {
		t.Error("expected rank error")
	}
	// scalar src
	srcScalar, _ := tensor.Zeros(2, nil, device.CPU())
	if _, err := ops.IndexSelect(srcScalar, idx); err == nil {
		t.Error("expected src rank error")
	}
}

func TestRMSNormErrors(t *testing.T) {
	x, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	wBad, _ := tensor.New([]float32{1, 2}, []int{2}, device.CPU())
	if _, err := ops.RMSNorm(x, wBad, 1e-5); err == nil {
		t.Error("expected shape mismatch")
	}
	wBadDt, _ := tensor.New([]float64{1, 2, 3}, []int{3}, device.CPU())
	if _, err := ops.RMSNorm(x, wBadDt, 1e-5); err == nil {
		t.Error("expected dtype mismatch")
	}
}

func TestLayerNormErrors(t *testing.T) {
	x, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	gBad, _ := tensor.New([]float32{1, 2}, []int{2}, device.CPU())
	b, _ := tensor.New([]float32{0, 0, 0}, []int{3}, device.CPU())
	if _, err := ops.LayerNorm(x, gBad, b, 1e-5); err == nil {
		t.Error("expected gamma shape mismatch")
	}
}
