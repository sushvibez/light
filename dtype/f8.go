package dtype

import "math"

// Float8 is an 8-bit float in the "fn" (finite, NaN-only) E4M3 encoding
// used by PyTorch's float8_e4m3fn, cuBLASLt FP8 GEMMs and FP8 checkpoints:
// 1 sign, 4 exponent (bias 7), 3 mantissa bits; no infinities, NaN is
// 0x7F / 0xFF, largest finite magnitude 448, smallest subnormal 2⁻⁹.
// Arithmetic happens in float32 via ToF32; F8FromF32 saturates out-of-
// range values to ±448 (the `__NV_SATFINITE` convention cuBLASLt uses),
// which keeps a quantised matmul finite instead of producing NaN.
type Float8 uint8

const (
	f8MaxFinite = 448.0
	f8MinNormal = 1.0 / 64  // 2⁻⁶
	f8Subnormal = 1.0 / 512 // 2⁻⁹, the smallest representable step
)

// F8FromF32 converts with round-to-nearest-even and saturation.
func F8FromF32(f float32) Float8 {
	if f != f { // NaN
		return 0x7F
	}
	var sign uint8
	v := float64(f)
	if math.Signbit(v) {
		sign = 0x80
		v = -v
	}
	if v >= f8MaxFinite {
		return Float8(sign | 0x7E) // ±448 = 0b0_1111_110
	}
	if v < f8MinNormal {
		// Subnormal: value = m · 2⁻⁹, m ∈ [0, 7].
		m := math.RoundToEven(v / f8Subnormal)
		if m >= 8 {
			return Float8(sign | 0x08) // rounds up into the first normal
		}
		return Float8(sign | uint8(m))
	}
	e := math.Floor(math.Log2(v))
	mant := v/math.Ldexp(1, int(e)) - 1 // ∈ [0, 1)
	m := math.RoundToEven(mant * 8)
	if m >= 8 {
		m = 0
		e++
	}
	exp := int(e) + 7
	if exp >= 16 || (exp == 15 && m >= 7) {
		return Float8(sign | 0x7E) // beyond 448 after rounding
	}
	return Float8(sign | uint8(exp)<<3 | uint8(m))
}

// ToF32 decodes the 8-bit encoding.
func (x Float8) ToF32() float32 {
	sign := float32(1)
	if x&0x80 != 0 {
		sign = -1
	}
	if x&0x7F == 0x7F {
		return float32(math.NaN())
	}
	exp := int(x>>3) & 0xF
	m := float32(x & 7)
	if exp == 0 {
		return sign * m * f8Subnormal
	}
	return sign * (1 + m/8) * float32(math.Ldexp(1, exp-7))
}
