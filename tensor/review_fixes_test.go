package tensor_test

import (
	"math"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// CopyFrom between contiguous prefix views must touch only the logical
// elements, not the rest of either parent allocation (R2).
func TestCopyFromPrefixViews(t *testing.T) {
	dstParent, _ := tensor.New([]float32{1, 2, 3, 4}, []int{4}, device.CPU())
	srcParent, _ := tensor.New([]float32{7, 8, 9, 10}, []int{4}, device.CPU())
	dst, _ := dstParent.Narrow(0, 0, 2)
	src, _ := srcParent.Narrow(0, 0, 2)
	if err := dst.CopyFrom(src); err != nil {
		t.Fatal(err)
	}
	if got := tensor.MustSlice[float32](dstParent); !slices.Equal(got, []float32{7, 8, 3, 4}) {
		t.Errorf("dst parent = %v, want [7 8 3 4]", got)
	}
	// A short destination over a full-size source, and vice versa.
	big, _ := tensor.New([]float32{5, 6}, []int{2}, device.CPU())
	if err := dst.CopyFrom(big); err != nil {
		t.Fatal(err)
	}
	if got := tensor.MustSlice[float32](dstParent); !slices.Equal(got, []float32{5, 6, 3, 4}) {
		t.Errorf("dst parent = %v, want [5 6 3 4]", got)
	}
	full, _ := tensor.New([]float32{0, 0}, []int{2}, device.CPU())
	if err := full.CopyFrom(src); err != nil {
		t.Fatal(err)
	}
	if got := tensor.MustSlice[float32](full); !slices.Equal(got, []float32{7, 8}) {
		t.Errorf("full = %v, want [7 8]", got)
	}
}

// Empty tensors are valid: Ones and the other CPU fills must return them
// rather than index an empty buffer (R12).
func TestFillsOnEmptyTensors(t *testing.T) {
	for _, dims := range [][]int{{0}, {0, 3}, {3, 0}, {2, 0, 4}} {
		for _, dt := range []dtype.DType{dtype.F32, dtype.F64, dtype.I32, dtype.I64, dtype.U8, dtype.I8, dtype.F16, dtype.BF16} {
			o, err := tensor.Ones(dt, dims, device.CPU())
			if err != nil {
				t.Fatalf("Ones(%v, %v): %v", dt, dims, err)
			}
			if o.NumElements() != 0 || !slices.Equal(o.Shape().Dims(), dims) {
				t.Errorf("Ones(%v, %v) gave %v", dt, dims, o.Shape())
			}
		}
		for _, dt := range []dtype.DType{dtype.F32, dtype.F64} {
			if _, err := tensor.Rand(dt, dims, device.CPU()); err != nil {
				t.Errorf("Rand(%v, %v): %v", dt, dims, err)
			}
		}
	}
	for _, dt := range []dtype.DType{dtype.F32, dtype.F64} {
		if a, err := tensor.Arange(1, 1, 1, dt, device.CPU()); err != nil || a.NumElements() != 0 {
			t.Errorf("Arange(1,1,1,%v) = %v, %v", dt, a, err)
		}
		if e, err := tensor.Eye(0, dt, device.CPU()); err != nil || e.NumElements() != 0 {
			t.Errorf("Eye(0,%v) = %v, %v", dt, e, err)
		}
	}
	x, _ := tensor.Zeros(dtype.F32, []int{0}, device.CPU())
	y, _ := tensor.New([]float32{1, 2}, []int{2}, device.CPU())
	if gx, gy, err := tensor.Meshgrid(x, y); err != nil || gx.NumElements() != 0 || gy.NumElements() != 0 {
		t.Errorf("Meshgrid(empty, y) = %v, %v, %v", gx, gy, err)
	}
}

// AsStrided bounds arithmetic must not wrap (R25): a huge stride on a small
// storage, overflow in the multiplication, and in the summation.
func TestAsStridedOverflowRejected(t *testing.T) {
	x, _ := tensor.New([]float32{1, 2, 3, 4}, []int{4}, device.CPU())
	bad := []struct {
		dims, strides []int
		offset        int
	}{
		{[]int{3}, []int{math.MaxInt}, 0},                                     // (d-1)*stride wraps
		{[]int{2, 2}, []int{math.MaxInt / 2, math.MaxInt/2 + 3}, 0},           // the sum wraps
		{[]int{2}, []int{math.MaxInt}, 1},                                     // offset + product wraps
		{[]int{math.MaxInt, math.MaxInt}, []int{0, 0}, 0},                     // element count wraps
		{[]int{1}, []int{1}, 4},                                               // offset past storage
		{[]int{2}, []int{3}, 1},                                               // plain out of range
		{[]int{math.MaxInt/2 + 2, 3}, []int{1, math.MaxInt / 3}, math.MaxInt}, // offset itself huge
	}
	for _, c := range bad {
		if _, err := x.AsStrided(c.dims, c.strides, c.offset); err == nil {
			t.Errorf("AsStrided(%v, %v, %d) accepted", c.dims, c.strides, c.offset)
		}
	}
	good := []struct {
		dims, strides []int
		offset        int
	}{
		{[]int{2, 2}, []int{2, 1}, 0},
		{[]int{2}, []int{3}, 0},
		{[]int{4, 5}, []int{1, 0}, 0},
		{[]int{0, 7}, []int{math.MaxInt, math.MaxInt}, 0}, // empty view
		{[]int{1}, []int{math.MaxInt}, 3},
	}
	for _, c := range good {
		if _, err := x.AsStrided(c.dims, c.strides, c.offset); err != nil {
			t.Errorf("AsStrided(%v, %v, %d): %v", c.dims, c.strides, c.offset, err)
		}
	}
}
