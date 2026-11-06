package dtype

import "math"

// Half is an IEEE 754 binary16 (half-precision) value stored as its 16-bit
// encoding. Arithmetic is performed by converting to float32 via ToF32,
// matching Candle's CPU-side half emulation policy.
type Half uint16

// BFloat16 is a bfloat16 value (1 sign, 8 exp, 7 mantissa) stored as its
// 16-bit encoding. BF16 is the top 16 bits of a float32's IEEE 754 encoding.
type BFloat16 uint16

// HalfFromF32 converts a float32 to F16 with round-to-nearest-even.
func HalfFromF32(f float32) Half {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 31) & 1)
	exp32 := int32((bits >> 23) & 0xff)
	mant32 := bits & 0x7fffff

	// NaN / +Inf / -Inf.
	if exp32 == 0xff {
		if mant32 == 0 {
			return Half(sign<<15 | 0x7c00)
		}
		// Quiet NaN: keep top mantissa bits, set MSB for quietness.
		m := uint16(mant32>>13) | 0x200
		return Half(sign<<15 | 0x7c00 | m)
	}

	// Zero or F32-subnormal → F16 zero (F32 subnormals are below F16 subnormal range).
	if exp32 == 0 {
		return Half(sign << 15)
	}

	realExp := exp32 - 127

	// Overflow to infinity.
	if realExp > 15 {
		return Half(sign<<15 | 0x7c00)
	}

	// Underflow to zero (smaller than smallest F16 subnormal).
	if realExp < -24 {
		return Half(sign << 15)
	}

	if realExp < -14 {
		// Subnormal in F16.
		// value = 2^realExp * (1 + mant32/2^23)
		// F16 subnormal value = 2^-14 * m/1024  → m = 2^(14+realExp) * (1 + mant32/2^23) * 1024
		// Implemented via shift + round-to-nearest-even.
		mant := mant32 | 0x800000 // implicit leading 1
		shift := uint32(-realExp - 14 + 13)
		h := mant >> shift
		// Round-to-nearest-even using the discarded bits.
		rem := mant & ((1 << shift) - 1)
		half := uint32(1) << (shift - 1)
		if rem > half || (rem == half && (h&1) != 0) {
			h++
		}
		return Half(sign<<15 | uint16(h))
	}

	// Normal in F16.
	hExp := uint32(realExp + 15)
	hMant := mant32 >> 13
	// Round-to-nearest-even on discarded 13 low bits.
	rem := mant32 & 0x1fff
	if rem > 0x1000 || (rem == 0x1000 && (hMant&1) != 0) {
		hMant++
		if hMant > 0x3ff {
			hMant = 0
			hExp++
			if hExp >= 31 {
				return Half(sign<<15 | 0x7c00) // overflow to inf
			}
		}
	}
	return Half(sign<<15 | uint16(hExp<<10) | uint16(hMant))
}

// ToF32 converts F16 to float32.
func (h Half) ToF32() float32 {
	bits := uint32(h)
	sign := (bits >> 15) & 1
	exp := (bits >> 10) & 0x1f
	mant := bits & 0x3ff

	var out uint32
	switch exp {
	case 0:
		if mant == 0 {
			out = sign << 31
		} else {
			// Subnormal: value = 2^-14 * (mant / 1024).
			v := float32(mant) * (1.0 / float32(1<<24)) // = mant * 2^-24
			if sign == 1 {
				v = -v
			}
			return v
		}
	case 0x1f:
		if mant == 0 {
			out = sign<<31 | 0x7f800000
		} else {
			out = sign<<31 | 0x7fc00000 // canonical NaN
		}
	default:
		out = sign<<31 | (exp+127-15)<<23 | mant<<13
	}
	return math.Float32frombits(out)
}

// BF16FromF32 converts a float32 to BF16. This is simple truncation (keep top
// 16 bits); we do not apply rounding since BF16 is primarily used for compute
// where the rounding is done in hardware tensor cores.
func BF16FromF32(f float32) BFloat16 {
	u := math.Float32bits(f)
	// Propagate NaN cleanly: if the low 16 bits encode "nan-ness" that the top
	// 16 bits would lose, set a quiet-NaN bit.
	if u&0x7f800000 == 0x7f800000 && u&0x7fffff != 0 {
		return BFloat16((u >> 16) | 0x0040) // ensure quiet NaN mantissa MSB
	}
	// Round to nearest even (what torch, Candle and the CUDA intrinsic do);
	// plain truncation biased every CPU BF16 result low by up to one ulp.
	u += 0x7fff + ((u >> 16) & 1)
	return BFloat16(u >> 16)
}

// ToF32 converts BF16 back to float32.
func (b BFloat16) ToF32() float32 {
	return math.Float32frombits(uint32(b) << 16)
}

// F32bits returns the IEEE-754 bit pattern of f as uint32. Thin wrapper over
// math.Float32bits so packages like quant don't all need to import math
// directly.
func F32bits(f float32) uint32 { return math.Float32bits(f) }
