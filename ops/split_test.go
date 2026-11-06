package ops_test

import (
	"slices"
	"testing"

	"light/device"
	"light/ops"
	"light/tensor"
)

func TestSplit(t *testing.T) {
	// (5, 3) along dim 0 with sizes [2, 3] → [(2,3), (3,3)]
	a, _ := tensor.New([]float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		13, 14, 15,
	}, []int{5, 3}, device.CPU())
	parts, err := ops.Split(a, []int{2, 3}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(parts))
	}
	if parts[0].Dim(0) != 2 || parts[0].Dim(1) != 3 {
		t.Errorf("part 0 shape = %v", parts[0].Shape())
	}
	c0, _ := parts[0].Contiguous()
	if !slices.Equal(tensor.MustSlice[float32](c0), []float32{1, 2, 3, 4, 5, 6}) {
		t.Errorf("part 0: %v", tensor.MustSlice[float32](c0))
	}
	c1, _ := parts[1].Contiguous()
	if !slices.Equal(tensor.MustSlice[float32](c1), []float32{7, 8, 9, 10, 11, 12, 13, 14, 15}) {
		t.Errorf("part 1: %v", tensor.MustSlice[float32](c1))
	}
}

func TestSplitErrors(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3, 4}, []int{4}, device.CPU())
	if _, err := ops.Split(a, nil, 0); err == nil {
		t.Error("expected empty-sizes error")
	}
	if _, err := ops.Split(a, []int{2, 3}, 0); err == nil {
		t.Error("expected sum-mismatch error")
	}
	if _, err := ops.Split(a, []int{0, 4}, 0); err == nil {
		t.Error("expected non-positive size error")
	}
}

func TestChunk(t *testing.T) {
	a, _ := tensor.New([]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, []int{10}, device.CPU())
	parts, err := ops.Chunk(a, 3, 0)
	if err != nil {
		t.Fatal(err)
	}
	// chunk_size = ceil(10/3) = 4. parts of 4, 4, 2.
	wantSizes := []int{4, 4, 2}
	if len(parts) != 3 {
		t.Fatalf("got %d parts", len(parts))
	}
	for i, p := range parts {
		if p.Dim(0) != wantSizes[i] {
			t.Errorf("part %d size = %d, want %d", i, p.Dim(0), wantSizes[i])
		}
	}
	c2, _ := parts[2].Contiguous()
	if !slices.Equal(tensor.MustSlice[float32](c2), []float32{8, 9}) {
		t.Errorf("last chunk: %v", tensor.MustSlice[float32](c2))
	}
}

func TestChunkEven(t *testing.T) {
	a, _ := tensor.New([]float32{0, 1, 2, 3, 4, 5}, []int{6}, device.CPU())
	parts, _ := ops.Chunk(a, 3, 0)
	if len(parts) != 3 {
		t.Fatalf("got %d parts", len(parts))
	}
	for _, p := range parts {
		if p.Dim(0) != 2 {
			t.Errorf("size = %d", p.Dim(0))
		}
	}
}

func TestChunkErrors(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2}, []int{2}, device.CPU())
	if _, err := ops.Chunk(a, 0, 0); err == nil {
		t.Error("expected n=0 error")
	}
	if _, err := ops.Chunk(a, 2, 5); err == nil {
		t.Error("expected axis OOB error")
	}
}
