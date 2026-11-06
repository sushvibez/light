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

// buildQ6_KBlock encodes one 210-byte Q6_K super-block.
//   - `d`     : F16 super-scale.
//   - `scales`: 16 int8 sub-block scales (one per 16 elements).
//   - `raw`   : 256 six-bit raw values in [0, 63]; the dequant formula uses
//     (raw - 32) times the scale, so to represent target integer
//     quants q ∈ [-32, 31] pass raw = q + 32.
func buildQ6_KBlock(d float32, scales [16]int8, raw [256]uint8) []byte {
	buf := make([]byte, quant.Q6_KBlockBytes)

	// Layout helpers: the 256 elements split into 2 halves of 128. Within a
	// half, the four groups (l+0, l+32, l+64, l+96) share ql[l] and ql[l+32]
	// plus qh[l] bits (0-1, 2-3, 4-5, 6-7 respectively).
	ql := make([]byte, 128)
	qh := make([]byte, 64)
	for half := range 2 {
		qlBase := half * 64
		qhBase := half * 32
		yBase := half * 128
		for l := range 32 {
			r1 := raw[yBase+l] & 0x3F    // elem l+0
			r2 := raw[yBase+l+32] & 0x3F // elem l+32
			r3 := raw[yBase+l+64] & 0x3F // elem l+64
			r4 := raw[yBase+l+96] & 0x3F // elem l+96
			// ql: low nibble from r1/r3 low4; high nibble from r2/r4 low4? No.
			// From dequant: raw1 = (ql[l]&0xF) | ((qh[l]>>0)&3)<<4
			//              raw3 = (ql[l]>>4)  | ((qh[l]>>4)&3)<<4
			// So ql[l] low nibble  = low 4 bits of r1
			//    ql[l] high nibble = low 4 bits of r3
			// And ql[l+32] low  = low 4 of r2; high = low 4 of r4.
			ql[qlBase+l] = (r1 & 0x0F) | ((r3 & 0x0F) << 4)
			ql[qlBase+l+32] = (r2 & 0x0F) | ((r4 & 0x0F) << 4)
			// qh[l]: bits 0-1 = r1>>4; bits 2-3 = r2>>4; bits 4-5 = r3>>4; bits 6-7 = r4>>4.
			qh[qhBase+l] = ((r1 >> 4) & 3) |
				(((r2 >> 4) & 3) << 2) |
				(((r3 >> 4) & 3) << 4) |
				(((r4 >> 4) & 3) << 6)
		}
	}
	copy(buf[0:128], ql)
	copy(buf[128:192], qh)
	for i, s := range scales {
		buf[192+i] = byte(s)
	}
	putHalf(buf[208:210], d)
	return buf
}

func TestDequantQ6_K(t *testing.T) {
	// All raw values = 32 → q = 0 → output = 0 regardless of scales.
	var scales [16]int8
	for i := range scales {
		scales[i] = int8(i + 1)
	}
	var raw [256]uint8
	for i := range raw {
		raw[i] = 32
	}
	block := buildQ6_KBlock(0.5, scales, raw)

	in, _ := tensor.FromBytes(block, dtype.Q6_K, []int{256}, device.CPU())
	out, err := quant.DequantQ6_K(in)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](out)
	for i := range got {
		if got[i] != 0 {
			t.Errorf("[%d] got %v, want 0", i, got[i])
			break
		}
	}
}

func TestDequantQ6_KKnownValues(t *testing.T) {
	// scale = 1 for every sub-block, d = 1; each sub-block's 16 elements
	// encode q = sub (raw = sub + 32). Signed q range is [-32, 31], so
	// keeping q in [0, 15] stays well within the representable window.
	// Expected output per sub-block = sub.
	var scales [16]int8
	for i := range scales {
		scales[i] = 1
	}
	var raw [256]uint8
	for sub := range 16 {
		for j := range 16 {
			raw[sub*16+j] = uint8(sub + 32)
		}
	}
	block := buildQ6_KBlock(1.0, scales, raw)

	in, _ := tensor.FromBytes(block, dtype.Q6_K, []int{256}, device.CPU())
	out, err := quant.DequantQ6_K(in)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](out)
	for sub := range 16 {
		want := float32(sub)
		for j := range 16 {
			g := got[sub*16+j]
			if math.Abs(float64(g-want)) > 1e-4 {
				t.Errorf("sub %d elem %d: got %v, want %v", sub, j, g, want)
				break
			}
		}
	}
}

func TestDequantQ6_KNegativeScale(t *testing.T) {
	// Exercise the signed-scale path: every scale = -3, every raw = 40 so q = 8.
	// Expected y = 0.25 * -3 * 8 = -6.
	var scales [16]int8
	for i := range scales {
		scales[i] = -3
	}
	var raw [256]uint8
	for i := range raw {
		raw[i] = 40
	}
	block := buildQ6_KBlock(0.25, scales, raw)

	in, _ := tensor.FromBytes(block, dtype.Q6_K, []int{256}, device.CPU())
	out, err := quant.DequantQ6_K(in)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](out)
	for i := range got {
		if math.Abs(float64(got[i]-(-6))) > 1e-3 {
			t.Errorf("[%d] got %v, want -6", i, got[i])
			break
		}
	}
}

func TestDequantQ6_KWrongDType(t *testing.T) {
	tt, _ := tensor.Zeros(dtype.F32, []int{256}, device.CPU())
	if _, err := quant.DequantQ6_K(tt); err == nil {
		t.Error("expected error for non-Q6_K input")
	}
	_ = binary.LittleEndian // keep import stable if future code removes it
}
