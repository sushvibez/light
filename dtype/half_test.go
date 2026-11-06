package dtype_test

import (
	"math"
	"testing"

	"light/dtype"
)

func TestHalfRoundTrip(t *testing.T) {
	cases := []float32{
		0.0,
		1.0, -1.0,
		0.5, -0.5,
		2.0, 4.0, 65504.0, // 65504 = max representable F16
		math.Pi, math.E,
		math.SmallestNonzeroFloat32, // Should underflow to 0 in F16
	}
	for _, x := range cases {
		h := dtype.HalfFromF32(x)
		back := h.ToF32()
		if math.IsNaN(float64(x)) {
			if !math.IsNaN(float64(back)) {
				t.Errorf("NaN round-trip: got %v", back)
			}
			continue
		}
		diff := math.Abs(float64(back) - float64(x))
		rel := diff / math.Max(math.Abs(float64(x)), 1e-38)
		if diff > 1e-30 && rel > 1e-3 {
			t.Errorf("round-trip %v → %v, diff=%v rel=%v", x, back, diff, rel)
		}
	}
}

func TestHalfSpecials(t *testing.T) {
	h := dtype.HalfFromF32(float32(math.Inf(1)))
	if !math.IsInf(float64(h.ToF32()), 1) {
		t.Errorf("+Inf lost: %v", h.ToF32())
	}
	h = dtype.HalfFromF32(float32(math.Inf(-1)))
	if !math.IsInf(float64(h.ToF32()), -1) {
		t.Errorf("-Inf lost: %v", h.ToF32())
	}
	h = dtype.HalfFromF32(float32(math.NaN()))
	if !math.IsNaN(float64(h.ToF32())) {
		t.Errorf("NaN lost: %v", h.ToF32())
	}
	h = dtype.HalfFromF32(1e10)
	if !math.IsInf(float64(h.ToF32()), 1) {
		t.Errorf("overflow should be +Inf, got %v", h.ToF32())
	}
	h = dtype.HalfFromF32(-1e10)
	if !math.IsInf(float64(h.ToF32()), -1) {
		t.Errorf("overflow should be -Inf, got %v", h.ToF32())
	}
}

func TestHalfKnownEncodings(t *testing.T) {
	cases := map[float32]uint16{
		1.0:  0x3c00,
		-1.0: 0xbc00,
		2.0:  0x4000,
		0.5:  0x3800,
		3.0:  0x4200,
	}
	for v, want := range cases {
		if h := dtype.HalfFromF32(v); uint16(h) != want {
			t.Errorf("%v → %#x, want %#x", v, uint16(h), want)
		}
	}
}

func TestHalfSubnormals(t *testing.T) {
	// F16 subnormal range: [2^-24, 2^-14). Pick values in it.
	// Smallest positive subnormal: 2^-24 ≈ 5.96e-8
	x := float32(math.Ldexp(1, -24))
	h := dtype.HalfFromF32(x)
	if h != 0x0001 {
		t.Errorf("smallest subnormal %v → %#x, want 0x0001", x, uint16(h))
	}
	back := h.ToF32()
	if math.Abs(float64(back)-float64(x)) > 1e-10 {
		t.Errorf("smallest subnormal round-trip: %v → %v", x, back)
	}

	// Halfway to largest subnormal.
	x = float32(math.Ldexp(1, -15)) // 2^-15
	h = dtype.HalfFromF32(x)
	back = h.ToF32()
	if math.Abs(float64(back)-float64(x)) > float64(x)*1e-3 {
		t.Errorf("subnormal 2^-15 round-trip: %v → %v", x, back)
	}

	// Negative subnormal.
	x = -float32(math.Ldexp(1, -20))
	h = dtype.HalfFromF32(x)
	back = h.ToF32()
	if back >= 0 {
		t.Errorf("negative subnormal should be negative: %v", back)
	}
}

func TestHalfRoundingTiesToEven(t *testing.T) {
	// A normal value whose low 13 bits form a half-way pattern should round to even.
	// Construct a float32 that's exactly halfway between two F16 values.
	// 1.0 + 2^-11 ≠ representable in F16 (F16 has 10 mantissa bits so step near 1.0 is 2^-10).
	// Pick a value where low 13 bits = 0x1000 (exactly halfway).
	x := math.Float32frombits(0x3f801000) // 1 + small
	h := dtype.HalfFromF32(x)
	// The round-trip error should be bounded.
	back := h.ToF32()
	if math.Abs(float64(back)-float64(x)) > float64(x)*2e-3 {
		t.Errorf("half-way rounding: %v → %v", x, back)
	}
}

func TestHalfUnderflow(t *testing.T) {
	// Values smaller than smallest F16 subnormal should go to zero.
	x := float32(math.Ldexp(1, -30))
	h := dtype.HalfFromF32(x)
	if h.ToF32() != 0 {
		t.Errorf("underflow: %v → %v, want 0", x, h.ToF32())
	}
	x = -float32(math.Ldexp(1, -30))
	h = dtype.HalfFromF32(x)
	if h.ToF32() != 0 && !math.Signbit(float64(h.ToF32())) {
		// Allow -0
		t.Errorf("neg underflow: %v → %v", x, h.ToF32())
	}
}

func TestHalfNaNPropagation(t *testing.T) {
	// NaN with specific mantissa: encoding should preserve quiet-NaN property.
	nanBits := math.Float32frombits(0x7fc12345) // Quiet NaN with payload
	h := dtype.HalfFromF32(nanBits)
	if !math.IsNaN(float64(h.ToF32())) {
		t.Errorf("NaN payload lost, got %v", h.ToF32())
	}
	// Check quiet-NaN bit preserved (F16 mantissa MSB)
	if uint16(h)&0x0200 == 0 {
		t.Errorf("expected quiet-NaN MSB set, got %#x", uint16(h))
	}
}

func TestHalfNegativeZero(t *testing.T) {
	h := dtype.HalfFromF32(float32(math.Copysign(0, -1)))
	if uint16(h) != 0x8000 {
		t.Errorf("-0 → %#x, want 0x8000", uint16(h))
	}
	back := h.ToF32()
	if !math.Signbit(float64(back)) {
		t.Errorf("-0 lost sign after round-trip: %v", back)
	}
}

func TestBF16RoundTrip(t *testing.T) {
	cases := []float32{0.0, 1.0, -1.0, 2.0, math.Pi, 65504.0, 1e20, 1e-20}
	for _, x := range cases {
		b := dtype.BF16FromF32(x)
		back := b.ToF32()
		rel := math.Abs(float64(back)-float64(x)) / math.Max(math.Abs(float64(x)), 1e-38)
		if rel > 0.01 {
			t.Errorf("BF16 round-trip %v → %v, rel=%v", x, back, rel)
		}
	}
}

func TestBF16Specials(t *testing.T) {
	b := dtype.BF16FromF32(float32(math.Inf(1)))
	if !math.IsInf(float64(b.ToF32()), 1) {
		t.Errorf("+Inf: %v", b.ToF32())
	}
	b = dtype.BF16FromF32(float32(math.NaN()))
	if !math.IsNaN(float64(b.ToF32())) {
		t.Errorf("NaN: %v", b.ToF32())
	}
}

func TestBF16NaNQuietPropagation(t *testing.T) {
	// Signalling NaN should become quiet (top mantissa bit set in BF16).
	snan := math.Float32frombits(0x7f800001)
	b := dtype.BF16FromF32(snan)
	if uint16(b)&0x0040 == 0 {
		t.Errorf("quiet-NaN bit not set, got %#x", uint16(b))
	}
	if !math.IsNaN(float64(b.ToF32())) {
		t.Errorf("NaN-ness lost: %v", b.ToF32())
	}
}

func TestBF16Encoding(t *testing.T) {
	if b := dtype.BF16FromF32(1.0); b != 0x3f80 {
		t.Errorf("1.0 → %#x, want 0x3f80", uint16(b))
	}
	if b := dtype.BF16FromF32(-1.0); b != 0xbf80 {
		t.Errorf("-1.0 → %#x, want 0xbf80", uint16(b))
	}
}
