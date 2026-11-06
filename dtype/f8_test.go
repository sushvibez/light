package dtype_test

import (
	"math"
	"testing"

	"light/dtype"
)

func TestF8E4M3RoundTrip(t *testing.T) {
	cases := map[float32]float32{
		0: 0, 1: 1, -1: -1, 0.5: 0.5, 1.125: 1.125, 448: 448, -448: -448,
		1.0 / 512: 1.0 / 512, 7.0 / 512: 7.0 / 512, 0.015625: 0.015625,
		1000: 448, -1000: -448, // saturate
		1.0625: 1.0, 1.1875: 1.25, // ties to even (mantissa steps of 0.125)
		3.3: 3.25, 0.1: 0.1015625, 100: 96,
		1.0 / 1024: 0, 1.6 / 512: 2.0 / 512,
	}
	for in, want := range cases {
		got := dtype.F8FromF32(in).ToF32()
		if got != want {
			t.Errorf("F8E4M3(%v) = %v, want %v (bits %#x)", in, got, want, uint8(dtype.F8FromF32(in)))
		}
	}
	if !math.IsNaN(float64(dtype.F8FromF32(float32(math.NaN())).ToF32())) {
		t.Error("NaN not preserved")
	}
	if dtype.F8E4M3.Size() != 1 || !dtype.F8E4M3.IsFloat() || dtype.F8E4M3.String() != "F8E4M3" {
		t.Error("dtype metadata wrong")
	}
	// Every encoding except NaN round-trips exactly.
	for b := 0; b < 256; b++ {
		if b&0x7F == 0x7F {
			continue
		}
		x := dtype.Float8(b)
		if back := dtype.F8FromF32(x.ToF32()); back != x && !(x.ToF32() == 0 && back&0x7F == 0) {
			t.Errorf("bits %#x → %v → %#x", b, x.ToF32(), uint8(back))
		}
	}
}
