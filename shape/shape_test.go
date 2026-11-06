package shape_test

import (
	"testing"

	"light/shape"
)

func TestNewAndNumElements(t *testing.T) {
	s := shape.New(2, 3, 4)
	if s.Rank() != 3 {
		t.Errorf("Rank = %d, want 3", s.Rank())
	}
	if s.NumElements() != 24 {
		t.Errorf("NumElements = %d, want 24", s.NumElements())
	}
}

func TestScalar(t *testing.T) {
	s := shape.New()
	if !s.IsScalar() {
		t.Errorf("rank-0 should be scalar")
	}
	if s.NumElements() != 1 {
		t.Errorf("scalar NumElements = %d, want 1", s.NumElements())
	}
}

func TestDim(t *testing.T) {
	s := shape.New(2, 3, 4)
	if s.Dim(0) != 2 {
		t.Errorf("Dim(0) = %d, want 2", s.Dim(0))
	}
	if s.Dim(-1) != 4 {
		t.Errorf("Dim(-1) = %d, want 4", s.Dim(-1))
	}
	if s.Dim(-2) != 3 {
		t.Errorf("Dim(-2) = %d, want 3", s.Dim(-2))
	}
}

func TestDimPanicsOutOfRange(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic")
		}
	}()
	shape.New(2, 3).Dim(5)
}

func TestNewPanicsNegative(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic")
		}
	}()
	shape.New(2, -3)
}

func TestEqual(t *testing.T) {
	s1 := shape.New(2, 3)
	s2 := shape.New(2, 3)
	s3 := shape.New(3, 2)
	s4 := shape.New(2, 3, 4)
	if !s1.Equal(s2) {
		t.Errorf("%v should equal %v", s1, s2)
	}
	if s1.Equal(s3) {
		t.Errorf("%v should not equal %v", s1, s3)
	}
	if s1.Equal(s4) {
		t.Errorf("%v should not equal %v (different rank)", s1, s4)
	}
}

func TestClone(t *testing.T) {
	s := shape.New(2, 3)
	c := s.Clone()
	if !s.Equal(c) {
		t.Error("clone not equal")
	}
	d := c.Dims()
	d[0] = 99
	if s.Dim(0) != 2 {
		t.Error("clone not independent")
	}
}

func TestDimsIsCopy(t *testing.T) {
	s := shape.New(2, 3)
	d := s.Dims()
	d[0] = 99
	if s.Dim(0) != 2 {
		t.Error("Dims() should return a copy")
	}
}

func TestString(t *testing.T) {
	cases := map[string]shape.Shape{
		"()":        shape.New(),
		"(3)":       shape.New(3),
		"(2, 3, 4)": shape.New(2, 3, 4),
	}
	for want, s := range cases {
		if got := s.String(); got != want {
			t.Errorf("%v → %q, want %q", s, got, want)
		}
	}
}

func TestNormalizeAxis(t *testing.T) {
	s := shape.New(2, 3, 4)
	ax, err := s.NormalizeAxis(1)
	if err != nil || ax != 1 {
		t.Errorf("NormalizeAxis(1) = (%d, %v)", ax, err)
	}
	ax, err = s.NormalizeAxis(-1)
	if err != nil || ax != 2 {
		t.Errorf("NormalizeAxis(-1) = (%d, %v)", ax, err)
	}
	if _, err := s.NormalizeAxis(5); err == nil {
		t.Error("expected error for axis 5")
	}
	if _, err := s.NormalizeAxis(-5); err == nil {
		t.Error("expected error for axis -5")
	}
}
