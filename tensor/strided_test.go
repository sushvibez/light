package tensor

import (
	"testing"

	"light/device"
	"light/dtype"
)

// Assign into a narrowed block of a (B, KH, cap, HD) buffer must land each
// (b, h) row at its pitch and leave the rest untouched.
func TestAssignIntoNarrowedBuffer(t *testing.T) {
	B, KH, capacity, HD := 2, 3, 5, 4
	buf, err := Zeros(dtype.F32, []int{B, KH, capacity, HD}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	T := 2
	src := make([]float32, B*KH*T*HD)
	for i := range src {
		src[i] = float32(i + 1)
	}
	s, _ := New(src, []int{B, KH, T, HD}, device.CPU())
	dst, err := buf.Narrow(2, 1, T)
	if err != nil {
		t.Fatal(err)
	}
	rows, width, pitch, ok := dst.rowLayout()
	if !ok || rows != B*KH || width != T*HD || pitch != capacity*HD {
		t.Fatalf("rowLayout = %d, %d, %d, %v; want %d, %d, %d, true", rows, width, pitch, ok, B*KH, T*HD, capacity*HD)
	}
	if err := dst.Assign(s); err != nil {
		t.Fatal(err)
	}
	got := MustSlice[float32](buf)
	for b := range B {
		for h := range KH {
			for p := range capacity {
				for d := range HD {
					v := got[((b*KH+h)*capacity+p)*HD+d]
					want := float32(0)
					if p >= 1 && p < 1+T {
						want = float32(((b*KH+h)*T+(p-1))*HD + d + 1)
					}
					if v != want {
						t.Fatalf("buf[%d,%d,%d,%d] = %v, want %v", b, h, p, d, v, want)
					}
				}
			}
		}
	}
	// Reading the block back through the view must round-trip.
	back, err := dst.Contiguous()
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range MustSlice[float32](back) {
		if v != src[i] {
			t.Fatalf("view[%d] = %v, want %v", i, v, src[i])
		}
	}
}

func TestAssignRejectsNonRowLayout(t *testing.T) {
	x, _ := Zeros(dtype.F32, []int{4, 4}, device.CPU())
	xt, _ := x.Transpose(0, 1)
	src, _ := Ones(dtype.F32, []int{4, 4}, device.CPU())
	if err := xt.Assign(src); err == nil {
		t.Fatal("Assign into a transposed view should fail")
	}
}
