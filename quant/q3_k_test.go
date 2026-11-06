package quant_test

import (
	"encoding/binary"
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/quant"
	"light/tensor"
)

// packQ3KScales is the inverse of unpackQ3KScales: takes 16 raw 6-bit values
// (0..63, i.e. signed scale + 32) and produces the 12-byte packed layout.
func packQ3KScales(raw [16]uint8) [12]byte {
	// Mirror of the unpack shuffle — operates on 4 uint32s.
	const kmask1 = uint32(0x03030303)
	const kmask2 = uint32(0x0f0f0f0f)
	var aux [4]uint32
	for k := range 4 {
		aux[k] = uint32(raw[4*k+0]) |
			uint32(raw[4*k+1])<<8 |
			uint32(raw[4*k+2])<<16 |
			uint32(raw[4*k+3])<<24
	}
	// Forward: split each byte into low-4 (stays in aux[0/1]) and high-2 (moves to aux[2]).
	// aux[0] final = low4 of scales[0..3]  = aux[0] & kmask2
	// aux[1] final = low4 of scales[4..7]  = aux[1] & kmask2
	// aux[2] final = low4 of scales[8..11] = (old aux[2]) & kmask2
	// aux[3] final = low4 of scales[12..15]= (old aux[3]) & kmask2
	// tmp bits 0-1 from each byte of aux[0]: scales[0..3] high2
	// tmp bits 2-3: scales[4..7] high2
	// tmp bits 4-5: scales[8..11] high2
	// tmp bits 6-7: scales[12..15] high2
	lo0 := aux[0] & kmask2
	lo1 := aux[1] & kmask2
	lo8 := aux[2] & kmask2
	lo12 := aux[3] & kmask2
	tmp := ((aux[0] >> 4) & kmask1) |
		(((aux[1] >> 4) & kmask1) << 2) |
		(((aux[2] >> 4) & kmask1) << 4) |
		(((aux[3] >> 4) & kmask1) << 6)
	packed0 := lo0 | (lo8 << 4)
	packed1 := lo1 | (lo12 << 4)
	packed2 := tmp

	var out [12]byte
	binary.LittleEndian.PutUint32(out[0:4], packed0)
	binary.LittleEndian.PutUint32(out[4:8], packed1)
	binary.LittleEndian.PutUint32(out[8:12], packed2)
	return out
}

// buildQ3_KBlock packs one 110-byte Q3_K super-block.
//   - `d`        : F16 super-scale
//   - `rawScales`: 16 values in [0,63]; dequant treats them as (x - 32) signed.
//   - `q3`       : 256 three-bit values stored as low2 + high-bit representation.
//     Callers pass `rawQ` (2-bit value in [0,3]) and `highBit`
//     (1 if the "upper" bit is set) separately — easier than
//     dealing with signed wrap-around in the test.
func buildQ3_KBlock(d float32, rawScales [16]uint8, rawQ2 [256]uint8, highBit [256]uint8) []byte {
	buf := make([]byte, quant.Q3_KBlockBytes)

	// hmask: 32 bytes; bit `half` of hmask[k] encodes the high bit for
	// the element whose hmask_byte_idx is k in that half.
	hmask := buf[0:32]
	// qs: 64 bytes, 4 elements per byte via shift 0/2/4/6.
	qs := buf[32:96]

	for idx := range 256 {
		half := idx / 128
		wh := idx % 128
		j := wh / 32
		withinJ := wh % 32
		subIdx := withinJ / 16
		l := withinJ % 16
		shift := uint(2 * j)

		qsByteIdx := half*32 + subIdx*16 + l
		qs[qsByteIdx] |= (rawQ2[idx] & 3) << shift

		hmByteIdx := subIdx*16 + l
		if highBit[idx] != 0 {
			hmask[hmByteIdx] |= 1 << half
		}
	}

	// scales: pack 12 bytes
	scalesPacked := packQ3KScales(rawScales)
	copy(buf[96:108], scalesPacked[:])

	putHalf(buf[108:110], d)
	return buf
}

// TestPackUnpackQ3KScales is a consistency check that packing then
// unpacking returns the original raw values for random patterns.
func TestPackUnpackQ3KScales(t *testing.T) {
	cases := [][16]uint8{
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		{63, 32, 17, 48, 7, 55, 23, 40, 1, 62, 31, 33, 8, 50, 25, 45},
		{33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33},
	}
	for _, raw := range cases {
		packed := packQ3KScales(raw)
		// Construct a minimal Q3_K block to run unpack.
		raw256 := [256]uint8{}
		high := [256]uint8{}
		b := buildQ3_KBlock(1.0, raw, raw256, high)
		// Pull scales through the dequant path: with q=0 all elements are 0, so
		// we can't introspect scales via output. Instead decode packed[] directly
		// by constructing a block with known q=1, highBit=1 → q_actual = 1.
		// Then y = d * (raw-32) * 1. For d=1: y = raw - 32 for sub-block 0.
		_ = packed
		_ = b
	}
	// Indirect check: run the full round-trip through DequantQ3_K.
	for _, raw := range cases {
		var q2 [256]uint8
		var hb [256]uint8
		for i := range q2 {
			q2[i] = 1
			hb[i] = 1 // so dequant q = 1 (no -4 offset)
		}
		block := buildQ3_KBlock(1.0, raw, q2, hb)
		in, _ := tensor.FromBytes(block, dtype.Q3_K, []int{256}, device.CPU())
		out, err := quant.DequantQ3_K(in)
		if err != nil {
			t.Fatal(err)
		}
		got := tensor.MustSlice[float32](out)
		for sub := range 16 {
			want := float32(int(raw[sub]) - 32)
			if math.Abs(float64(got[sub*16]-want)) > 1e-3 {
				t.Errorf("raw=%v sub=%d got %v want %v", raw, sub, got[sub*16], want)
				break
			}
		}
	}
}

func TestDequantQ3_K(t *testing.T) {
	// Each sub-block has a distinct signed scale; q=2 low2 + high-bit set
	// gives q_actual=2. Expected per element: d * (sc-32) * 2.
	var rawScales [16]uint8
	for i := range rawScales {
		rawScales[i] = uint8(32 + i) // signed scale i
	}
	var rawQ2 [256]uint8
	var highBit [256]uint8
	for i := range rawQ2 {
		rawQ2[i] = 2
		highBit[i] = 1
	}
	block := buildQ3_KBlock(0.5, rawScales, rawQ2, highBit)

	in, _ := tensor.FromBytes(block, dtype.Q3_K, []int{256}, device.CPU())
	out, err := quant.DequantQ3_K(in)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](out)
	for sub := range 16 {
		want := 0.5 * float32(sub) * 2
		for l := range 16 {
			g := got[sub*16+l]
			if math.Abs(float64(g-want)) > 1e-3 {
				t.Errorf("sub %d l %d got %v want %v", sub, l, g, want)
				break
			}
		}
	}
}

func TestDequantQ3_KHighBitOff(t *testing.T) {
	// When hmask bit is 0, dequant subtracts 4 from the 2-bit low value.
	// rawQ2 = 3, highBit = 0 → q = 3 - 4 = -1. sc=32 (signed=0) would give 0;
	// use sc=35 (signed=3), d=1. Expected: 1 * 3 * -1 = -3.
	var rawScales [16]uint8
	for i := range rawScales {
		rawScales[i] = 35
	}
	var rawQ2 [256]uint8
	var highBit [256]uint8
	for i := range rawQ2 {
		rawQ2[i] = 3
		highBit[i] = 0
	}
	block := buildQ3_KBlock(1.0, rawScales, rawQ2, highBit)

	in, _ := tensor.FromBytes(block, dtype.Q3_K, []int{256}, device.CPU())
	out, _ := quant.DequantQ3_K(in)
	got := tensor.MustSlice[float32](out)
	for i := range got {
		if math.Abs(float64(got[i]-(-3))) > 1e-3 {
			t.Errorf("[%d] got %v want -3", i, got[i])
			break
		}
	}
}

func TestDequantQ3_KWrongDType(t *testing.T) {
	tt, _ := tensor.Zeros(dtype.F32, []int{256}, device.CPU())
	if _, err := quant.DequantQ3_K(tt); err == nil {
		t.Error("expected error for non-Q3_K input")
	}
}
