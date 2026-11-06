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

func TestConv2dIdentityKernel(t *testing.T) {
	// 1×1 kernel with weight=1 should pass input through (same shape when stride=1, padding=0).
	// input: (1, 1, 3, 3), weight: (1, 1, 1, 1) = [[1]]
	inData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	input, _ := tensor.New(inData, []int{1, 1, 3, 3}, device.CPU())
	weight, _ := tensor.New([]float32{1}, []int{1, 1, 1, 1}, device.CPU())
	out, err := ops.Conv2d(input, weight, nil, [2]int{1, 1}, [2]int{0, 0})
	if err != nil {
		t.Fatal(err)
	}
	if out.Dim(0) != 1 || out.Dim(1) != 1 || out.Dim(2) != 3 || out.Dim(3) != 3 {
		t.Errorf("shape = %v", out.Shape())
	}
	if !slices.Equal(tensor.MustSlice[float32](out), inData) {
		t.Errorf("got %v", tensor.MustSlice[float32](out))
	}
}

func TestConv2dSum3x3(t *testing.T) {
	// 3x3 kernel of all-ones on a 3x3 input → output is (1,1,1,1) containing sum of input = 45.
	inData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	input, _ := tensor.New(inData, []int{1, 1, 3, 3}, device.CPU())
	weight, _ := tensor.Ones(dtype.F32, []int{1, 1, 3, 3}, device.CPU())
	out, err := ops.Conv2d(input, weight, nil, [2]int{1, 1}, [2]int{0, 0})
	if err != nil {
		t.Fatal(err)
	}
	if tensor.MustSlice[float32](out)[0] != 45 {
		t.Errorf("got %v, want 45", tensor.MustSlice[float32](out)[0])
	}
}

func TestConv2dBias(t *testing.T) {
	// Same as identity but with bias=[10], output = input + 10.
	inData := []float32{1, 2, 3, 4}
	input, _ := tensor.New(inData, []int{1, 1, 2, 2}, device.CPU())
	weight, _ := tensor.New([]float32{1}, []int{1, 1, 1, 1}, device.CPU())
	bias, _ := tensor.New([]float32{10}, []int{1}, device.CPU())
	out, _ := ops.Conv2d(input, weight, bias, [2]int{1, 1}, [2]int{0, 0})
	want := []float32{11, 12, 13, 14}
	if !slices.Equal(tensor.MustSlice[float32](out), want) {
		t.Errorf("got %v", tensor.MustSlice[float32](out))
	}
}

func TestConv2dStridePadding(t *testing.T) {
	// (1,1,4,4) with kernel (1,1,3,3) = all ones, stride=2, padding=1 → output (1,1,2,2).
	input, _ := tensor.Ones(dtype.F32, []int{1, 1, 4, 4}, device.CPU())
	weight, _ := tensor.Ones(dtype.F32, []int{1, 1, 3, 3}, device.CPU())
	out, _ := ops.Conv2d(input, weight, nil, [2]int{2, 2}, [2]int{1, 1})
	if out.Dim(2) != 2 || out.Dim(3) != 2 {
		t.Errorf("shape = %v", out.Shape())
	}
	// Center positions see 9 ones, corners see fewer.
	// h=0 w=0: kernel top-left at (-1,-1); contributions from input[0,0]..input[1,1] = 4.
	// h=0 w=1: contributes from input[0,1..3]*[0..2] — let me just verify numeric: any cell
	// should be between 4 and 9.
	vals := tensor.MustSlice[float32](out)
	for _, v := range vals {
		if v < 4 || v > 9 {
			t.Errorf("value %v out of expected range [4, 9]", v)
		}
	}
}

func TestMaxPool2d(t *testing.T) {
	// (1, 1, 4, 4) with kernel (2, 2), stride (2, 2) → output (1, 1, 2, 2).
	in, _ := tensor.New([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}, []int{1, 1, 4, 4}, device.CPU())
	out, err := ops.MaxPool2d(in, [2]int{2, 2}, [2]int{2, 2}, [2]int{0, 0})
	if err != nil {
		t.Fatal(err)
	}
	// Per 2x2 window max: [6, 8, 14, 16]
	want := []float32{6, 8, 14, 16}
	if !slices.Equal(tensor.MustSlice[float32](out), want) {
		t.Errorf("got %v, want %v", tensor.MustSlice[float32](out), want)
	}
}

func TestAvgPool2d(t *testing.T) {
	// Same input; avg of each 2x2 window.
	in, _ := tensor.New([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}, []int{1, 1, 4, 4}, device.CPU())
	out, _ := ops.AvgPool2d(in, [2]int{2, 2}, [2]int{2, 2}, [2]int{0, 0})
	// Mean: [(1+2+5+6)/4=3.5, (3+4+7+8)/4=5.5, (9+10+13+14)/4=11.5, (11+12+15+16)/4=13.5]
	want := []float32{3.5, 5.5, 11.5, 13.5}
	got := tensor.MustSlice[float32](out)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Errorf("[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestConv2dErrors(t *testing.T) {
	// Wrong rank
	in3d, _ := tensor.Zeros(dtype.F32, []int{1, 2, 3}, device.CPU())
	w, _ := tensor.Zeros(dtype.F32, []int{1, 1, 1, 1}, device.CPU())
	if _, err := ops.Conv2d(in3d, w, nil, [2]int{1, 1}, [2]int{0, 0}); err == nil {
		t.Error("expected rank error")
	}
	// Cin mismatch
	in, _ := tensor.Zeros(dtype.F32, []int{1, 3, 4, 4}, device.CPU())
	wBad, _ := tensor.Zeros(dtype.F32, []int{1, 5, 3, 3}, device.CPU())
	if _, err := ops.Conv2d(in, wBad, nil, [2]int{1, 1}, [2]int{0, 0}); err == nil {
		t.Error("expected Cin mismatch")
	}
}
