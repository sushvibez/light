package quant_test

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/quant"
	"light/tensor"
)

// buildQ2_KBlock packs one 84-byte Q2_K super-block.
//   - `d`,`dmin`      : F16 super-scales.
//   - `subScales`/`subMins` : 16 four-bit sub-block scales/mins.
//   - `q2`             : 256 two-bit quants (each in [0,3]).
func buildQ2_KBlock(d, dmin float32, subScales, subMins [16]uint8, q2 [256]uint8) []byte {
	buf := make([]byte, quant.Q2_KBlockBytes)
	// scales[i] = (subScales[i] & 0xF) | (subMins[i] & 0xF) << 4
	for i := range 16 {
		buf[i] = (subScales[i] & 0x0F) | ((subMins[i] & 0x0F) << 4)
	}
	// qs: byte at position (half*32 + sub_index*16 + l) holds four 2-bit
	// quants, one per shift ∈ {0,2,4,6}. Element idx layout:
	//   half = idx/128, wh = idx%128
	//   j = wh/32 (→ shift=2j), within-j = wh%32
	//   sub_index = within-j/16, l = within-j%16
	qs := buf[16:80]
	for idx := range 256 {
		half := idx / 128
		wh := idx % 128
		j := wh / 32
		withinJ := wh % 32
		subIdx := withinJ / 16
		l := withinJ % 16
		shift := uint(2 * j)
		byteIdx := half*32 + subIdx*16 + l
		qs[byteIdx] |= (q2[idx] & 3) << shift
	}
	putHalf(buf[80:82], d)
	putHalf(buf[82:84], dmin)
	return buf
}

func TestDequantQ2_K(t *testing.T) {
	// d=1, dmin=1; 4-bit scales and mins (0..15). Each sub-block i has
	// scale=i, min=i, and all 16 elements quantized to i%4 ∈ [0,3].
	// Expected per element: i * (i%4) - i.
	var subScales, subMins [16]uint8
	for i := range subScales {
		subScales[i] = uint8(i)
		subMins[i] = uint8(i)
	}
	var q2 [256]uint8
	for i := range q2 {
		sub := i / 16
		q2[i] = uint8(sub & 3)
	}
	raw := buildQ2_KBlock(1.0, 1.0, subScales, subMins, q2)

	in, _ := tensor.FromBytes(raw, dtype.Q2_K, []int{256}, device.CPU())
	out, err := quant.DequantQ2_K(in)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](out)
	for sub := range 16 {
		want := float32(sub)*float32(sub&3) - float32(sub)
		for l := range 16 {
			g := got[sub*16+l]
			if math.Abs(float64(g-want)) > 1e-4 {
				t.Errorf("sub %d l %d: got %v want %v", sub, l, g, want)
				break
			}
		}
	}
}

func TestDequantQ2_KAllZero(t *testing.T) {
	var subScales, subMins [16]uint8
	var q2 [256]uint8
	// All zeros — output should be 0 regardless of scales.
	raw := buildQ2_KBlock(0.25, 0.5, subScales, subMins, q2)
	in, _ := tensor.FromBytes(raw, dtype.Q2_K, []int{256}, device.CPU())
	out, _ := quant.DequantQ2_K(in)
	got := tensor.MustSlice[float32](out)
	for i := range got {
		if got[i] != 0 {
			t.Errorf("[%d] got %v want 0", i, got[i])
			break
		}
	}
}

func TestDequantQ2_KWrongDType(t *testing.T) {
	tt, _ := tensor.Zeros(dtype.F32, []int{256}, device.CPU())
	if _, err := quant.DequantQ2_K(tt); err == nil {
		t.Error("expected error for non-Q2_K input")
	}
}
