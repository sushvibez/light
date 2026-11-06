package shape_test

import (
	"testing"

	"light/shape"
)

func TestDefaultStrides(t *testing.T) {
	s := shape.DefaultStrides(shape.New(2, 3, 4))
	if !s.Equal(shape.NewStrides(12, 4, 1)) {
		t.Errorf("got %v, want [12 4 1]", s.Slice())
	}
}

func TestDefaultStridesScalar(t *testing.T) {
	s := shape.DefaultStrides(shape.New())
	if s.Rank() != 0 {
		t.Errorf("scalar strides rank = %d, want 0", s.Rank())
	}
}

func TestStridesGet(t *testing.T) {
	s := shape.NewStrides(12, 4, 1)
	if s.Get(0) != 12 {
		t.Errorf("Get(0) = %d", s.Get(0))
	}
	if s.Get(-1) != 1 {
		t.Errorf("Get(-1) = %d", s.Get(-1))
	}
}

func TestIsContiguous(t *testing.T) {
	sh := shape.New(2, 3, 4)
	if !shape.IsContiguous(sh, shape.DefaultStrides(sh)) {
		t.Error("default strides should be contiguous")
	}
	// Permuted (transposed): not contiguous.
	st := shape.NewStrides(1, 4, 12)
	if shape.IsContiguous(sh, st) {
		t.Error("permuted strides should not be contiguous")
	}
}

func TestPermute(t *testing.T) {
	orig := shape.NewStrides(12, 4, 1)
	// Transpose axes (0,2,1) → new strides [12, 1, 4]
	p := orig.Permute([]int{0, 2, 1})
	if !p.Equal(shape.NewStrides(12, 1, 4)) {
		t.Errorf("got %v, want [12 1 4]", p.Slice())
	}
}

func TestSliceIsCopy(t *testing.T) {
	s := shape.NewStrides(12, 4, 1)
	sl := s.Slice()
	sl[0] = 99
	if s.Get(0) != 12 {
		t.Error("Slice() should return a copy")
	}
}

func TestContigWithDim0Broadcast(t *testing.T) {
	sh := shape.New(2, 3, 4)
	// Default — yes.
	if !shape.ContigWithDim0Broadcast(sh, shape.DefaultStrides(sh)) {
		t.Error("default should satisfy")
	}
	// Dim0 broadcast (stride 0) — yes.
	st := shape.NewStrides(0, 4, 1)
	if !shape.ContigWithDim0Broadcast(sh, st) {
		t.Error("dim0 broadcast should satisfy")
	}
	// Inner axis broken — no.
	st = shape.NewStrides(0, 5, 1)
	if shape.ContigWithDim0Broadcast(sh, st) {
		t.Error("inner non-default should not satisfy")
	}
	// Scalar — trivially yes.
	if !shape.ContigWithDim0Broadcast(shape.New(), shape.NewStrides()) {
		t.Error("scalar should satisfy")
	}
}
