package shape_test

import (
	"testing"

	"light/shape"
)

func TestBroadcast(t *testing.T) {
	cases := []struct {
		a, b, want shape.Shape
	}{
		{shape.New(2, 3, 4), shape.New(4), shape.New(2, 3, 4)},
		{shape.New(5, 1, 4), shape.New(3, 4), shape.New(5, 3, 4)},
		{shape.New(2, 1, 5), shape.New(1, 3, 5), shape.New(2, 3, 5)},
		{shape.New(), shape.New(3), shape.New(3)}, // scalar with vector
	}
	for i, c := range cases {
		got, err := shape.Broadcast(c.a, c.b)
		if err != nil {
			t.Errorf("case %d: unexpected error: %v", i, err)
			continue
		}
		if !got.Equal(c.want) {
			t.Errorf("case %d: %v broadcast %v → %v, want %v", i, c.a, c.b, got, c.want)
		}
	}
}

func TestBroadcastIncompatible(t *testing.T) {
	cases := [][2]shape.Shape{
		{shape.New(2, 3), shape.New(3, 2)},
		{shape.New(2, 3), shape.New(4)},
	}
	for i, c := range cases {
		if _, err := shape.Broadcast(c[0], c[1]); err == nil {
			t.Errorf("case %d: expected error broadcasting %v and %v", i, c[0], c[1])
		}
	}
}

func TestBroadcastN(t *testing.T) {
	got, err := shape.BroadcastN(shape.New(2, 1, 4), shape.New(3, 1), shape.New(4))
	if err != nil {
		t.Fatal(err)
	}
	if !got.Equal(shape.New(2, 3, 4)) {
		t.Errorf("got %v, want (2,3,4)", got)
	}

	// Zero args
	if _, err := shape.BroadcastN(); err == nil {
		t.Error("expected error on zero args")
	}

	// Incompatible
	if _, err := shape.BroadcastN(shape.New(2, 3), shape.New(3, 2)); err == nil {
		t.Error("expected error")
	}
}

func TestBroadcastStrides(t *testing.T) {
	// (3,) -> (2, 3): prepend axis with stride 0, keep last stride
	from := shape.New(3)
	fromSt := shape.DefaultStrides(from) // [1]
	to := shape.New(2, 3)
	got, err := shape.BroadcastStrides(from, fromSt, to)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Equal(shape.NewStrides(0, 1)) {
		t.Errorf("got %v, want [0 1]", got.Slice())
	}

	// (2, 1, 4) -> (2, 3, 4): middle axis broadcast
	from = shape.New(2, 1, 4)
	fromSt = shape.DefaultStrides(from) // [4, 4, 1]
	to = shape.New(2, 3, 4)
	got, err = shape.BroadcastStrides(from, fromSt, to)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Equal(shape.NewStrides(4, 0, 1)) {
		t.Errorf("got %v, want [4 0 1]", got.Slice())
	}

	// Higher rank can't broadcast to lower.
	if _, err := shape.BroadcastStrides(shape.New(2, 3), shape.DefaultStrides(shape.New(2, 3)), shape.New(3)); err == nil {
		t.Error("expected rank error")
	}

	// Incompatible axis
	if _, err := shape.BroadcastStrides(shape.New(2, 3), shape.DefaultStrides(shape.New(2, 3)), shape.New(4, 3)); err == nil {
		t.Error("expected axis mismatch error")
	}
}
