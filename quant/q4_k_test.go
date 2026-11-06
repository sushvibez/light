package quant_test

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/quant"
	"light/tensor"
)

// buildQ4_KBlock builds one 144-byte Q4_K super-block from F32 d / dmin,
// 6-bit sub-block scales and mins (length 8 each), and 256 4-bit quants.
// Uses the ggml Q4_K packing:
//
//	scales[j]    j in [0,4): low6 = sc_j, top2 = high bits of sc_{j+4}
//	scales[j+4]  j in [0,4): low6 = m_j,  top2 = high bits of m_{j+4}
//	scales[j+8]  j in [0,4): low4 = low4 of sc_{j+4}, high4 = low4 of m_{j+4}
func buildQ4_KBlock(d, dmin float32, subScales, subMins [8]uint8, quants [256]uint8) []byte {
	buf := make([]byte, quant.Q4_KBlockBytes)
	putHalf(buf[0:2], d)
	putHalf(buf[2:4], dmin)

	var sc12 [12]byte
	for j := range 4 {
		sc12[j] = (subScales[j] & 0x3F) | ((subScales[j+4] & 0x30) << 2)
		sc12[j+4] = (subMins[j] & 0x3F) | ((subMins[j+4] & 0x30) << 2)
		sc12[j+8] = (subScales[j+4] & 0x0F) | ((subMins[j+4] & 0x0F) << 4)
	}
	copy(buf[4:16], sc12[:])

	// Pack quants. For pair p in [0,4), sub-blocks 2p and 2p+1 share
	// qs[p*32:(p+1)*32]; low nibble is element (p*64+l), high nibble is
	// element (p*64+32+l) for l in [0,32).
	qs := buf[16:144]
	for p := range 4 {
		qsOff := p * 32
		for l := range 32 {
			lo := quants[p*64+l] & 0x0F
			hi := quants[p*64+32+l] & 0x0F
			qs[qsOff+l] = lo | (hi << 4)
		}
	}
	return buf
}

func TestDequantQ4_K(t *testing.T) {
	// Each sub-block j has scale = j+1, min = j, and all 32 elements
	// quantized to (j+1) & 0x0F. With d=1, dmin=1: y = sc * q4 - m.
	var subScales, subMins [8]uint8
	for j := range 8 {
		subScales[j] = uint8(j + 1)
		subMins[j] = uint8(j)
	}
	var quants [256]uint8
	for j := range 8 {
		for i := range 32 {
			quants[j*32+i] = uint8((j + 1) & 0x0F)
		}
	}
	raw := buildQ4_KBlock(1.0, 1.0, subScales, subMins, quants)

	in, err := tensor.FromBytes(raw, dtype.Q4_K, []int{256}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	out, err := quant.DequantQ4_K(in)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](out)

	for j := range 8 {
		sc := float32(subScales[j])
		m := float32(subMins[j])
		q4 := float32(subScales[j] & 0x0F)
		want := sc*q4 - m
		for i := range 32 {
			g := got[j*32+i]
			if math.Abs(float64(g-want)) > 1e-4 {
				t.Errorf("sub-block %d elem %d: got %v, want %v", j, i, g, want)
				break
			}
		}
	}
}

func TestDequantQ4_KMultiBlock(t *testing.T) {
	var subScales, subMins [8]uint8
	for j := range 8 {
		subScales[j] = uint8(j * 2)
		subMins[j] = uint8(j)
	}
	var quants [256]uint8
	for i := range quants {
		quants[i] = uint8(i % 16)
	}
	b1 := buildQ4_KBlock(0.5, 0.25, subScales, subMins, quants)
	b2 := buildQ4_KBlock(2.0, 1.0, subScales, subMins, quants)
	raw := make([]byte, 0, len(b1)+len(b2))
	raw = append(raw, b1...)
	raw = append(raw, b2...)

	in, err := tensor.FromBytes(raw, dtype.Q4_K, []int{512}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	out, err := quant.DequantQ4_K(in)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](out)

	// Sub-block 7 first element in block 0: sc=14, m=7, q=(7*32)%16=0
	//  → 0.5 * 14 * 0 - 0.25 * 7 = -1.75
	wantB0Sub7 := -float32(0.25 * 7)
	if math.Abs(float64(got[7*32]-wantB0Sub7)) > 1e-3 {
		t.Errorf("block0 sub7 elem0 = %v, want %v", got[7*32], wantB0Sub7)
	}
	// Same sub-block in block 1 (d=2.0, dmin=1.0, same structure):
	// → 2.0 * 14 * 0 - 1.0 * 7 = -7
	wantB1Sub7 := float32(-7)
	if math.Abs(float64(got[256+7*32]-wantB1Sub7)) > 1e-3 {
		t.Errorf("block1 sub7 elem0 = %v, want %v", got[256+7*32], wantB1Sub7)
	}
}

func TestDequantQ4_KWrongDType(t *testing.T) {
	tt, _ := tensor.Zeros(dtype.F32, []int{256}, device.CPU())
	if _, err := quant.DequantQ4_K(tt); err == nil {
		t.Error("expected error for non-Q4_K input")
	}
}
