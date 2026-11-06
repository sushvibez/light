package ops_test

import (
	"slices"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

func TestGatherDim0(t *testing.T) {
	// src: (4, 3) = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
	// indices (dim=0, shape (2,3)): [[0,2,0],[3,1,2]]
	// out: [[src[0,0], src[2,1], src[0,2]], [src[3,0], src[1,1], src[2,2]]]
	//    = [[1, 8, 3], [10, 5, 9]]
	src, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, []int{4, 3}, device.CPU())
	idx, _ := tensor.New([]int64{0, 2, 0, 3, 1, 2}, []int{2, 3}, device.CPU())
	out, err := ops.Gather(src, idx, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 8, 3, 10, 5, 9}
	if !slices.Equal(tensor.MustSlice[float32](out), want) {
		t.Errorf("got %v, want %v", tensor.MustSlice[float32](out), want)
	}
}

func TestGatherDim1(t *testing.T) {
	// src (3, 4); gather along dim=1 with idx (3, 2)
	src, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, []int{3, 4}, device.CPU())
	idx, _ := tensor.New([]int64{1, 3, 0, 2, 2, 1}, []int{3, 2}, device.CPU())
	out, err := ops.Gather(src, idx, 1)
	if err != nil {
		t.Fatal(err)
	}
	// row 0: src[0, 1], src[0, 3] = 2, 4
	// row 1: src[1, 0], src[1, 2] = 5, 7
	// row 2: src[2, 2], src[2, 1] = 11, 10
	want := []float32{2, 4, 5, 7, 11, 10}
	if !slices.Equal(tensor.MustSlice[float32](out), want) {
		t.Errorf("got %v, want %v", tensor.MustSlice[float32](out), want)
	}
}

func TestGatherErrors(t *testing.T) {
	src, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	idxBadDt, _ := tensor.New([]int32{0}, []int{1}, device.CPU())
	if _, err := ops.Gather(src, idxBadDt, 0); err == nil {
		t.Error("expected indices-dtype error")
	}
	idxBadRank, _ := tensor.New([]int64{0, 1}, []int{1, 2}, device.CPU())
	if _, err := ops.Gather(src, idxBadRank, 0); err == nil {
		t.Error("expected rank mismatch error")
	}
}

func TestScatterAdd(t *testing.T) {
	// base: zeros(3, 4)
	base, _ := tensor.Zeros(dtype.F32, []int{3, 4}, device.CPU())
	// src: (3, 2), indices: (3, 2) — scatter along dim=1.
	src, _ := tensor.New([]float32{10, 20, 30, 40, 50, 60}, []int{3, 2}, device.CPU())
	idx, _ := tensor.New([]int64{0, 2, 1, 3, 0, 0}, []int{3, 2}, device.CPU())
	out, err := ops.ScatterAdd(base, idx, src, 1)
	if err != nil {
		t.Fatal(err)
	}
	// row 0: col 0 += 10, col 2 += 20 → [10, 0, 20, 0]
	// row 1: col 1 += 30, col 3 += 40 → [0, 30, 0, 40]
	// row 2: col 0 += 50, col 0 += 60 → [110, 0, 0, 0]
	want := []float32{10, 0, 20, 0, 0, 30, 0, 40, 110, 0, 0, 0}
	got := tensor.MustSlice[float32](out)
	if !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestScatterAddErrors(t *testing.T) {
	base, _ := tensor.Zeros(dtype.F32, []int{3, 4}, device.CPU())
	src, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	idx, _ := tensor.New([]int64{0, 1, 2}, []int{3}, device.CPU())
	// src/idx rank != base rank
	if _, err := ops.ScatterAdd(base, idx, src, 0); err == nil {
		t.Error("expected rank mismatch")
	}
}

func TestMaskedFill(t *testing.T) {
	base, _ := tensor.New([]float32{1, 2, 3, 4}, []int{4}, device.CPU())
	mask, _ := tensor.New([]uint8{0, 1, 0, 1}, []int{4}, device.CPU())
	out, err := ops.MaskedFill(base, mask, -99)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, -99, 3, -99}
	got := tensor.MustSlice[float32](out)
	if !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestMaskedFillErrors(t *testing.T) {
	base, _ := tensor.New([]float32{1, 2}, []int{2}, device.CPU())
	badMask, _ := tensor.New([]float32{1, 0}, []int{2}, device.CPU())
	if _, err := ops.MaskedFill(base, badMask, 0); err == nil {
		t.Error("expected mask-dtype error")
	}
}

func TestMaskedSelect(t *testing.T) {
	input, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device.CPU())
	mask, _ := tensor.New([]uint8{1, 0, 1, 0, 1, 0}, []int{2, 3}, device.CPU())
	out, err := ops.MaskedSelect(input, mask)
	if err != nil {
		t.Fatal(err)
	}
	if out.Rank() != 1 || out.Dim(0) != 3 {
		t.Errorf("shape = %v, want (3,)", out.Shape())
	}
	if !slices.Equal(tensor.MustSlice[float32](out), []float32{1, 3, 5}) {
		t.Errorf("got %v", tensor.MustSlice[float32](out))
	}
}

func TestMaskedSelectErrors(t *testing.T) {
	input, _ := tensor.New([]float32{1, 2}, []int{2}, device.CPU())
	bad, _ := tensor.New([]float32{1, 0}, []int{2}, device.CPU())
	if _, err := ops.MaskedSelect(input, bad); err == nil {
		t.Error("expected mask-dtype error")
	}
	mask, _ := tensor.New([]uint8{1, 0, 1}, []int{3}, device.CPU())
	if _, err := ops.MaskedSelect(input, mask); err == nil {
		t.Error("expected shape mismatch")
	}
}

// ---- scatter / index_add / slice_scatter / slice_assign (2026-10-28) ----

func TestScatterIndexAddValues(t *testing.T) {
	cpu := device.CPU()
	base, _ := tensor.New([]float32{0, 0, 0, 0, 0, 0, 0, 0}, []int{2, 4}, cpu)
	src, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, cpu)
	// torch.zeros(2,4).scatter(1, [[0,2,2],[3,1,0]], src) → [[1,0,3,0],[6,5,0,4]]
	idx, _ := tensor.New([]int64{0, 2, 2, 3, 1, 0}, []int{2, 3}, cpu)
	out, err := ops.Scatter(base, idx, src, 1)
	if err != nil {
		t.Fatal(err)
	}
	expectRow(t, "scatter dim1", f32s(t, out), []float32{1, 0, 3, 0, 6, 5, 0, 4}, 0)
	// scatter along dim 0: base (3,2), src (2,2), idx [[2,0],[0,1]] → [[3,2],[0,4],[1,0]]
	base30, _ := tensor.New([]float32{0, 0, 0, 0, 0, 0}, []int{3, 2}, cpu)
	src22, _ := tensor.New([]float32{1, 2, 3, 4}, []int{2, 2}, cpu)
	idx0, _ := tensor.New([]int64{2, 0, 0, 1}, []int{2, 2}, cpu)
	out0, err := ops.Scatter(base30, idx0, src22, 0)
	if err != nil {
		t.Fatal(err)
	}
	expectRow(t, "scatter dim0", f32s(t, out0), []float32{3, 2, 0, 4, 1, 0}, 0)
	// index_add: torch.ones(2,4).index_add(1, [3,0,3], src) → [[1+2+... ]]
	ones, _ := tensor.Ones(dtype.F32, []int{2, 4}, cpu)
	ia, _ := tensor.New([]int64{3, 0, 3}, []int{3}, cpu)
	outIA, err := ops.IndexAdd(ones, ia, src, 1)
	if err != nil {
		t.Fatal(err)
	}
	expectRow(t, "index_add dim1", f32s(t, outIA), []float32{3, 1, 1, 5, 6, 1, 1, 11}, 0)
	// index_add along dim 0 with slices: base (3,2) zeros, idx [1,1], src (2,2) → row1 = [4,6]
	ia0, _ := tensor.New([]int64{1, 1}, []int{2}, cpu)
	outIA0, err := ops.IndexAdd(base30, ia0, src22, 0)
	if err != nil {
		t.Fatal(err)
	}
	expectRow(t, "index_add dim0", f32s(t, outIA0), []float32{0, 0, 4, 6, 0, 0}, 0)
	// slice_scatter: base arange(12).reshape(3,4); src (3,2) → cols 1..3
	b12 := fillT(t, []int{3, 4}, 0, cpu)
	s32, _ := tensor.New([]float32{-1, -2, -3, -4, -5, -6}, []int{3, 2}, cpu)
	ss, err := ops.SliceScatter(b12, s32, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	bs := f32s(t, b12)
	expectRow(t, "slice_scatter", f32s(t, ss), []float32{bs[0], -1, -2, bs[3], bs[4], -3, -4, bs[7], bs[8], -5, -6, bs[11]}, 0)
	// slice_assign: (2,2) block at (1,2) of a (3,4)
	blk, _ := tensor.New([]float32{9, 8, 7, 6}, []int{2, 2}, cpu)
	sa, err := ops.SliceAssign(b12, blk, []int{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	expectRow(t, "slice_assign", f32s(t, sa), []float32{bs[0], bs[1], bs[2], bs[3], bs[4], bs[5], 9, 8, bs[8], bs[9], 7, 6}, 0)
	if _, err := ops.IndexAdd(ones, idx, src, 1); err == nil {
		t.Error("index_add with 2-D indices should fail")
	}
	if _, err := ops.SliceScatter(b12, s32, 1, 3); err == nil {
		t.Error("slice_scatter out of range should fail")
	}
}

func TestIndexOpsCUDAParity(t *testing.T) {
	gpu := cudaOrSkip(t)
	cpu := device.CPU()
	// Non-colliding indices so scatter / atomics are deterministic.
	idx2, _ := tensor.New([]int64{3, 0, 2, 1, 4, 0, 3, 1, 2, 4, 0, 1}, []int{2, 2, 3}, cpu)
	idx1, _ := tensor.New([]int64{4, 1}, []int{2}, cpu)
	gidx, _ := tensor.New([]int64{0, 4, 1, 3, 2, 2, 1, 0, 4, 3, 0, 1}, []int{2, 2, 3}, cpu)
	for _, dt := range []dtype.DType{dtype.F32, dtype.F64, dtype.I32, dtype.I64, dtype.BF16, dtype.F16} {
		mk := func(dims []int, seed float64) *tensor.Tensor {
			x := fillT(t, dims, seed, cpu)
			x, _ = ops.Affine(x, 8, 0) // integer-friendly magnitudes
			x, _ = ops.Round(x)
			c, err := ops.Cast(x, dt)
			if err != nil {
				t.Fatal(err)
			}
			return c
		}
		base := mk([]int{2, 5, 3}, 0.3)
		src := mk([]int{2, 2, 3}, 0.6)
		run := func(dev device.Device) [][]float32 {
			b, _ := base.To(dev)
			s, _ := src.To(dev)
			i2, _ := idx2.To(dev)
			i1, _ := idx1.To(dev)
			gi, _ := gidx.To(dev)
			g, err := ops.Gather(b, gi, 1)
			if err != nil {
				t.Fatalf("%v gather on %v: %v", dt, dev, err)
			}
			sc, err := ops.Scatter(b, i2, s, 1)
			if err != nil {
				t.Fatalf("%v scatter on %v: %v", dt, dev, err)
			}
			sa, err := ops.ScatterAdd(b, i2, s, 1)
			if err != nil {
				t.Fatalf("%v scatter_add on %v: %v", dt, dev, err)
			}
			ia, err := ops.IndexAdd(b, i1, s, 1)
			if err != nil {
				t.Fatalf("%v index_add on %v: %v", dt, dev, err)
			}
			return [][]float32{f32s(t, g), f32s(t, sc), f32s(t, sa), f32s(t, ia)}
		}
		c, g := run(cpu), run(gpu)
		for i, name := range []string{"gather", "scatter", "scatter_add", "index_add"} {
			expectRow(t, dt.String()+" "+name, g[i], c[i], 0)
		}
	}
}
