package ops_test

import (
	"slices"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// Review R23: an index past the axis must not silently land in the next row.
func TestGatherScatterRejectOutOfRangeIndex(t *testing.T) {
	src := mustT(t, []float32{1, 2, 3, 4}, 2, 2)
	for _, bad := range [][]int64{{2, 0}, {0, -1}, {1, 7}} {
		idx := mustT(t, bad, 2, 1)
		if out, err := ops.Gather(src, idx, 1); err == nil {
			t.Errorf("Gather idx %v: want error, got %v", bad, tensor.MustSlice[float32](out))
		}
		upd := mustT(t, []float32{9, 8}, 2, 1)
		if out, err := ops.ScatterAdd(src, idx, upd, 1); err == nil {
			t.Errorf("ScatterAdd idx %v: want error, got %v", bad, tensor.MustSlice[float32](out))
		}
		if _, err := ops.Scatter(src, idx, upd, 1); err == nil {
			t.Errorf("Scatter idx %v: want error", bad)
		}
	}
	// 16-bit ScatterAdd and IndexAdd use separate kernels.
	bf, _ := ops.Cast(src, dtype.BF16)
	bfUpd, _ := ops.Cast(mustT(t, []float32{9, 8}, 2, 1), dtype.BF16)
	if _, err := ops.ScatterAdd(bf, mustT(t, []int64{2, 0}, 2, 1), bfUpd, 1); err == nil {
		t.Error("BF16 ScatterAdd: want error for index 2 on an axis of 2")
	}
	row := mustT(t, []float32{5, 6}, 1, 2)
	if _, err := ops.IndexAdd(src, mustT(t, []int64{2}, 1), row, 0); err == nil {
		t.Error("IndexAdd: want error for index 2 on an axis of 2")
	}
	if _, err := ops.IndexSelect(src, mustT(t, []int64{-1}, 1)); err == nil {
		t.Error("IndexSelect: want error for index -1")
	}
	// Shape contract: indices may not exceed src outside the gather dim.
	if _, err := ops.Gather(src, mustT(t, []int64{0, 0, 0}, 3, 1), 1); err == nil {
		t.Error("Gather: want shape error for indices [3,1] on src [2,2] dim 1")
	}
	// Shape contract: ScatterAdd src must match base outside dim.
	if _, err := ops.ScatterAdd(src, mustT(t, []int64{0, 0, 0}, 3, 1), mustT(t, []float32{1, 1, 1}, 3, 1), 1); err == nil {
		t.Error("ScatterAdd: want shape error for src [3,1] on base [2,2] dim 1")
	}

	// Valid inputs still work.
	out, err := ops.Gather(src, mustT(t, []int64{1, 0}, 2, 1), 1)
	if err != nil || !slices.Equal(tensor.MustSlice[float32](out), []float32{2, 3}) {
		t.Errorf("Gather valid: %v %v", out, err)
	}
	out, err = ops.ScatterAdd(src, mustT(t, []int64{1, 0}, 2, 1), mustT(t, []float32{9, 8}, 2, 1), 1)
	if err != nil || !slices.Equal(tensor.MustSlice[float32](out), []float32{1, 11, 11, 4}) {
		t.Errorf("ScatterAdd valid: %v %v", out, err)
	}
}

// Review R24: TopK with an empty output returns shaped empties.
func TestTopKEmpty(t *testing.T) {
	cases := []struct {
		dims []int
		k    int
		want []int
	}{
		{[]int{2, 4}, 0, []int{2, 0}},
		{[]int{0, 4}, 2, []int{0, 2}},
		{[]int{2, 0}, 0, []int{2, 0}},
	}
	for _, c := range cases {
		a, err := tensor.Zeros(dtype.F32, c.dims, device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		v, i, err := ops.TopK(a, c.k, true)
		if err != nil {
			t.Fatalf("TopK(%v, k=%d): %v", c.dims, c.k, err)
		}
		if !slices.Equal(v.Shape().Dims(), c.want) || !slices.Equal(i.Shape().Dims(), c.want) || i.DType() != dtype.I64 {
			t.Errorf("TopK(%v, k=%d): shapes %v/%v (%v), want %v", c.dims, c.k, v.Shape(), i.Shape(), i.DType(), c.want)
		}
	}
	// Normal k still works.
	a := mustT(t, []float32{1, 5, 3, 4, 0, 2}, 2, 3)
	v, i, err := ops.TopK(a, 2, true)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(tensor.MustSlice[float32](v), []float32{5, 3, 4, 2}) || !slices.Equal(tensor.MustSlice[int64](i), []int64{1, 2, 0, 2}) {
		t.Errorf("TopK k=2: %v %v", tensor.MustSlice[float32](v), tensor.MustSlice[int64](i))
	}
}

// Review R12 (CPU Softmax part): an empty softmax axis yields an empty result.
func TestSoftmaxEmpty(t *testing.T) {
	for _, dt := range []dtype.DType{dtype.F32, dtype.F64} {
		for _, dims := range [][]int{{2, 0}, {0, 3}, {0}} {
			a, err := tensor.Zeros(dt, dims, device.CPU())
			if err != nil {
				t.Fatal(err)
			}
			out, err := ops.Softmax(a, -1)
			if err != nil {
				t.Fatalf("Softmax %v %v: %v", dt, dims, err)
			}
			if !slices.Equal(out.Shape().Dims(), dims) {
				t.Errorf("Softmax %v %v: shape %v", dt, dims, out.Shape())
			}
		}
	}
}

func mustT[T int64 | float32](t *testing.T, data []T, dims ...int) *tensor.Tensor {
	t.Helper()
	x, err := tensor.New(data, dims, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	return x
}
