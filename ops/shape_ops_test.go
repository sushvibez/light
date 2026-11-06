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

func TestPad(t *testing.T) {
	cpu := device.CPU()
	a, _ := tensor.New([]float32{1, 2, 3}, []int{3}, cpu)
	b, err := ops.Pad(a, 0, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if b.Dim(0) != 6 {
		t.Fatalf("Pad dim: got %d", b.Dim(0))
	}
	if got := tensor.MustSlice[float32](b); !slices.Equal(got, []float32{0, 1, 2, 3, 0, 0}) {
		t.Errorf("Pad: got %v", got)
	}
	// 2-D, pad the first dim with the second untouched.
	m, _ := tensor.New([]float32{1, 2, 3, 4}, []int{2, 2}, cpu)
	p, err := ops.Pad(m, 0, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := tensor.MustSlice[float32](p); !slices.Equal(got, []float32{0, 0, 1, 2, 3, 4}) {
		t.Errorf("Pad 2-D: got %v", got)
	}
	// Negative dim, inner axis, I64 dtype.
	mi, _ := tensor.New([]int64{1, 2, 3, 4}, []int{2, 2}, cpu)
	pi, err := ops.Pad(mi, -1, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got := tensor.MustSlice[int64](pi); !slices.Equal(got, []int64{1, 2, 0, 3, 4, 0}) {
		t.Errorf("Pad inner I64: got %v", got)
	}
	if _, err := ops.Pad(a, 0, -1, 0); err == nil {
		t.Error("expected error for negative pad")
	}
	if _, err := ops.Pad(a, 3, 0, 0); err == nil {
		t.Error("expected error for bad dim")
	}
}

func TestFlip(t *testing.T) {
	cpu := device.CPU()
	a, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, cpu)
	f1, err := ops.Flip(a, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got := tensor.MustSlice[float32](f1); !slices.Equal(got, []float32{3, 2, 1, 6, 5, 4}) {
		t.Errorf("Flip dim1: got %v", got)
	}
	f0, _ := ops.Flip(a, 0)
	if got := tensor.MustSlice[float32](f0); !slices.Equal(got, []float32{4, 5, 6, 1, 2, 3}) {
		t.Errorf("Flip dim0: got %v", got)
	}
	fb, _ := ops.Flip(a, 0, -1)
	if got := tensor.MustSlice[float32](fb); !slices.Equal(got, []float32{6, 5, 4, 3, 2, 1}) {
		t.Errorf("Flip both: got %v", got)
	}
	// Flipping twice is the identity; no dims is a copy.
	ff, _ := ops.Flip(f1, 1)
	if got := tensor.MustSlice[float32](ff); !slices.Equal(got, tensor.MustSlice[float32](a)) {
		t.Errorf("Flip twice: got %v", got)
	}
	cp, _ := ops.Flip(a)
	if got := tensor.MustSlice[float32](cp); !slices.Equal(got, tensor.MustSlice[float32](a)) {
		t.Errorf("Flip none: got %v", got)
	}
	if _, err := ops.Flip(a, 0, 0); err == nil {
		t.Error("expected duplicate-dim error")
	}
	// Non-contiguous input (transposed view).
	tr, _ := a.Transpose(0, 1) // (3, 2): [[1,4],[2,5],[3,6]]
	ft, err := ops.Flip(tr, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := tensor.MustSlice[float32](ft); !slices.Equal(got, []float32{3, 6, 2, 5, 1, 4}) {
		t.Errorf("Flip transposed: got %v", got)
	}
}

func TestRoll(t *testing.T) {
	cpu := device.CPU()
	a, _ := tensor.New([]float32{1, 2, 3, 4}, []int{4}, cpu)
	r, err := ops.Roll(a, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := tensor.MustSlice[float32](r); !slices.Equal(got, []float32{4, 1, 2, 3}) {
		t.Errorf("Roll +1: got %v", got)
	}
	r, _ = ops.Roll(a, -1, 0)
	if got := tensor.MustSlice[float32](r); !slices.Equal(got, []float32{2, 3, 4, 1}) {
		t.Errorf("Roll -1: got %v", got)
	}
	r, _ = ops.Roll(a, 5, 0) // 5 mod 4 == 1
	if got := tensor.MustSlice[float32](r); !slices.Equal(got, []float32{4, 1, 2, 3}) {
		t.Errorf("Roll +5: got %v", got)
	}
	m, _ := tensor.New([]int32{1, 2, 3, 4, 5, 6}, []int{2, 3}, cpu)
	r, _ = ops.Roll(m, 1, 1)
	if got := tensor.MustSlice[int32](r); !slices.Equal(got, []int32{3, 1, 2, 6, 4, 5}) {
		t.Errorf("Roll dim1: got %v", got)
	}
	r, _ = ops.Roll(m, 1, 0)
	if got := tensor.MustSlice[int32](r); !slices.Equal(got, []int32{4, 5, 6, 1, 2, 3}) {
		t.Errorf("Roll dim0: got %v", got)
	}
}

func TestSort(t *testing.T) {
	cpu := device.CPU()
	a, _ := tensor.New([]float32{3, 1, 2}, []int{3}, cpu)
	v, idx, err := ops.Sort(a, false)
	if err != nil {
		t.Fatal(err)
	}
	if got := tensor.MustSlice[float32](v); !slices.Equal(got, []float32{1, 2, 3}) {
		t.Errorf("Sort asc vals: %v", got)
	}
	if idx.DType() != dtype.I64 {
		t.Errorf("Sort idx dtype = %v", idx.DType())
	}
	if got := tensor.MustSlice[int64](idx); !slices.Equal(got, []int64{1, 2, 0}) {
		t.Errorf("Sort asc idx: %v", got)
	}
	v, idx, _ = ops.Sort(a, true)
	if got := tensor.MustSlice[float32](v); !slices.Equal(got, []float32{3, 2, 1}) {
		t.Errorf("Sort desc vals: %v", got)
	}
	if got := tensor.MustSlice[int64](idx); !slices.Equal(got, []int64{0, 2, 1}) {
		t.Errorf("Sort desc idx: %v", got)
	}
	// Rows sorted independently; ties keep original order (stable).
	m, _ := tensor.New([]float64{2, 2, 1, 9, 8, 8}, []int{2, 3}, cpu)
	v, idx, _ = ops.Sort(m, false)
	if got := tensor.MustSlice[float64](v); !slices.Equal(got, []float64{1, 2, 2, 8, 8, 9}) {
		t.Errorf("Sort rows vals: %v", got)
	}
	if got := tensor.MustSlice[int64](idx); !slices.Equal(got, []int64{2, 0, 1, 1, 2, 0}) {
		t.Errorf("Sort rows idx: %v", got)
	}
	mi, _ := tensor.New([]int64{5, -1, 3}, []int{3}, cpu)
	v, _, _ = ops.Sort(mi, false)
	if got := tensor.MustSlice[int64](v); !slices.Equal(got, []int64{-1, 3, 5}) {
		t.Errorf("Sort I64: %v", got)
	}
}

func TestNorm(t *testing.T) {
	cpu := device.CPU()
	a, _ := tensor.New([]float32{3, 4}, []int{2}, cpu)
	l2, err := ops.Norm(a, 2, ops.ReduceOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if l2.Rank() != 0 {
		t.Errorf("Norm over all elements should be rank-0, got %v", l2.Shape())
	}
	if got := tensor.MustSlice[float32](l2)[0]; math.Abs(float64(got-5)) > 1e-5 {
		t.Errorf("L2 norm = %v, want 5", got)
	}
	l1, _ := ops.Norm(a, 1, ops.ReduceOptions{})
	if got := tensor.MustSlice[float32](l1)[0]; math.Abs(float64(got-7)) > 1e-5 {
		t.Errorf("L1 norm = %v, want 7", got)
	}
	// Along an axis, with negatives (abs) and F64.
	m, _ := tensor.New([]float64{-3, 4, 1, -1}, []int{2, 2}, cpu)
	rows, _ := ops.Norm(m, 2, ops.ReduceOptions{Axes: []int{1}})
	got := tensor.MustSlice[float64](rows)
	want := []float64{5, math.Sqrt2}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-9 {
			t.Errorf("row norm[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	cols, _ := ops.Norm(m, 3, ops.ReduceOptions{Axes: []int{0}, KeepDims: true})
	if cols.Rank() != 2 || cols.Dim(0) != 1 || cols.Dim(1) != 2 {
		t.Errorf("keepdims shape = %v", cols.Shape())
	}
	got = tensor.MustSlice[float64](cols)
	want = []float64{math.Cbrt(27 + 1), math.Cbrt(64 + 1)}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-9 {
			t.Errorf("col L3 norm[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	if _, err := ops.Norm(a, 0, ops.ReduceOptions{}); err == nil {
		t.Error("expected error for p=0")
	}
	ai, _ := tensor.New([]int32{1}, []int{1}, cpu)
	if _, err := ops.Norm(ai, 2, ops.ReduceOptions{}); err == nil {
		t.Error("expected dtype error")
	}
}

func TestUnfold(t *testing.T) {
	x, _ := tensor.New([]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, []int{2, 5}, device.CPU())
	u, err := ops.Unfold(x, 1, 3, 2) // windows [0:3], [2:5] per row
	if err != nil {
		t.Fatal(err)
	}
	if d := u.Shape().Dims(); d[0] != 2 || d[1] != 2 || d[2] != 3 {
		t.Fatalf("shape %v", u.Shape())
	}
	expectRow(t, "unfold", tensor.MustSlice[float32](u), []float32{0, 1, 2, 2, 3, 4, 5, 6, 7, 7, 8, 9}, 0)
	gpu := cudaOrSkip(t)
	xg, _ := x.To(gpu)
	ug, err := ops.Unfold(xg, 1, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	expectRow(t, "unfold cuda", f32s(t, ug), tensor.MustSlice[float32](u), 0)
}

func TestPixelShuffle(t *testing.T) {
	// (1, 4, 1, 2) with r=2 → (1, 1, 2, 4)
	x, _ := tensor.New([]float32{0, 1, 2, 3, 4, 5, 6, 7}, []int{1, 4, 1, 2}, device.CPU())
	y, err := ops.PixelShuffle(x, 2)
	if err != nil {
		t.Fatal(err)
	}
	if d := y.Shape().Dims(); d[1] != 1 || d[2] != 2 || d[3] != 4 {
		t.Fatalf("shape %v", y.Shape())
	}
	// torch.pixel_shuffle on this input: [[0,2,1,3],[4,6,5,7]]
	expectRow(t, "pixel_shuffle", tensor.MustSlice[float32](y), []float32{0, 2, 1, 3, 4, 6, 5, 7}, 0)
	back, err := ops.PixelUnshuffle(y, 2)
	if err != nil {
		t.Fatal(err)
	}
	expectRow(t, "pixel_unshuffle", tensor.MustSlice[float32](back), tensor.MustSlice[float32](x), 0)
	gpu := cudaOrSkip(t)
	xg, _ := x.To(gpu)
	yg, err := ops.PixelShuffle(xg, 2)
	if err != nil {
		t.Fatal(err)
	}
	expectRow(t, "pixel_shuffle cuda", f32s(t, yg), tensor.MustSlice[float32](y), 0)
}

func TestReplicationPad2d(t *testing.T) {
	x, _ := tensor.New([]float32{1, 2, 3, 4}, []int{1, 1, 2, 2}, device.CPU())
	y, err := ops.ReplicationPad2d(x, 1, 2, 1, 0) // left 1, right 2, top 1, bottom 0
	if err != nil {
		t.Fatal(err)
	}
	if d := y.Shape().Dims(); d[2] != 3 || d[3] != 5 {
		t.Fatalf("shape %v", y.Shape())
	}
	want := []float32{
		1, 1, 2, 2, 2,
		1, 1, 2, 2, 2,
		3, 3, 4, 4, 4,
	}
	expectRow(t, "replication_pad2d", tensor.MustSlice[float32](y), want, 0)
	gpu := cudaOrSkip(t)
	xg, _ := x.To(gpu)
	yg, err := ops.ReplicationPad2d(xg, 1, 2, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	expectRow(t, "replication_pad2d cuda", f32s(t, yg), want, 0)
}
