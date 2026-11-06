package quant_test

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/quant"
	"light/tensor"
)

// buildQ5_KBlock builds one 176-byte Q5_K super-block.
//   - `d`,`dmin`   : F16 super-scales.
//   - subScales/subMins : 8 six-bit sub-block scales/mins.
//   - q5 : 256 five-bit quants in [0, 31].
//
// Uses Q4_K scale packing + a 32-byte qh bitmap for the 5th bit.
func buildQ5_KBlock(d, dmin float32, subScales, subMins [8]uint8, q5 [256]uint8) []byte {
	buf := make([]byte, quant.Q5_KBlockBytes)
	putHalf(buf[0:2], d)
	putHalf(buf[2:4], dmin)

	// Pack 12-byte scales (same layout as Q4_K).
	var sc12 [12]byte
	for j := range 4 {
		sc12[j] = (subScales[j] & 0x3F) | ((subScales[j+4] & 0x30) << 2)
		sc12[j+4] = (subMins[j] & 0x3F) | ((subMins[j+4] & 0x30) << 2)
		sc12[j+8] = (subScales[j+4] & 0x0F) | ((subMins[j+4] & 0x0F) << 4)
	}
	copy(buf[4:16], sc12[:])

	// qh bitmap: 32 bytes, bit pattern per byte qh[l]:
	//   bit 2p   → element (p*64 + l) high bit
	//   bit 2p+1 → element (p*64 + 32 + l) high bit
	qh := buf[16:48]
	for p := range 4 {
		bit1 := uint8(1 << (2 * p))
		bit2 := uint8(1 << (2*p + 1))
		for l := range 32 {
			if q5[p*64+l]&0x10 != 0 {
				qh[l] |= bit1
			}
			if q5[p*64+32+l]&0x10 != 0 {
				qh[l] |= bit2
			}
		}
	}

	// qs: 128 bytes of packed low-4 nibbles, same as Q4_K.
	qs := buf[48:176]
	for p := range 4 {
		qsOff := p * 32
		for l := range 32 {
			lo := q5[p*64+l] & 0x0F
			hi := q5[p*64+32+l] & 0x0F
			qs[qsOff+l] = lo | (hi << 4)
		}
	}
	return buf
}

func TestDequantQ5_K(t *testing.T) {
	// d = 1, dmin = 1; each sub-block has scale = j+1, min = j.
	// All 32 elements in sub-block j have q5 = j+1 (so we exercise values up
	// through 8, all representable in 5 bits).
	var subScales, subMins [8]uint8
	for j := range 8 {
		subScales[j] = uint8(j + 1)
		subMins[j] = uint8(j)
	}
	var q5 [256]uint8
	for j := range 8 {
		for i := range 32 {
			q5[j*32+i] = uint8(j + 1)
		}
	}
	raw := buildQ5_KBlock(1.0, 1.0, subScales, subMins, q5)

	in, err := tensor.FromBytes(raw, dtype.Q5_K, []int{256}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	out, err := quant.DequantQ5_K(in)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](out)

	for j := range 8 {
		sc := float32(subScales[j])
		m := float32(subMins[j])
		want := sc*float32(j+1) - m
		for i := range 32 {
			g := got[j*32+i]
			if math.Abs(float64(g-want)) > 1e-4 {
				t.Errorf("sub %d elem %d: got %v want %v", j, i, g, want)
				break
			}
		}
	}
}

func TestDequantQ5_KFifthBit(t *testing.T) {
	// Every quant = 17 → low 4 bits = 1, high bit = 1. Exercises the qh
	// bitmap. scale = 2, min = 0, d = 1, dmin = 1 → y = 2 * 17 = 34.
	var subScales, subMins [8]uint8
	for j := range 8 {
		subScales[j] = 2
		subMins[j] = 0
	}
	var q5 [256]uint8
	for i := range q5 {
		q5[i] = 17
	}
	raw := buildQ5_KBlock(1.0, 1.0, subScales, subMins, q5)

	in, _ := tensor.FromBytes(raw, dtype.Q5_K, []int{256}, device.CPU())
	out, err := quant.DequantQ5_K(in)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](out)
	for i := range got {
		if math.Abs(float64(got[i]-34)) > 1e-3 {
			t.Errorf("[%d] got %v want 34", i, got[i])
			break
		}
	}
}

func TestDequantQ5_KWrongDType(t *testing.T) {
	tt, _ := tensor.Zeros(dtype.F32, []int{256}, device.CPU())
	if _, err := quant.DequantQ5_K(tt); err == nil {
		t.Error("expected error for non-Q5_K input")
	}
}
