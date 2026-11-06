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

// buildQ8_KBlock packs one 292-byte Q8_K super-block. bsums is metadata not
// used by pure dequant; we leave it zeroed.
func buildQ8_KBlock(d float32, qs [256]int8) []byte {
	buf := make([]byte, quant.Q8_KBlockBytes)
	binary.LittleEndian.PutUint32(buf[0:4], math.Float32bits(d))
	for i, v := range qs {
		buf[4+i] = byte(v)
	}
	// bsums at buf[260:292] left zero.
	return buf
}

func TestDequantQ8_K(t *testing.T) {
	var qs [256]int8
	for i := range qs {
		qs[i] = int8(i - 128) // full range -128..127
	}
	raw := buildQ8_KBlock(0.25, qs)

	in, _ := tensor.FromBytes(raw, dtype.Q8_K, []int{256}, device.CPU())
	out, err := quant.DequantQ8_K(in)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](out)
	for i := range got {
		want := 0.25 * float32(qs[i])
		if math.Abs(float64(got[i]-want)) > 1e-6 {
			t.Errorf("[%d] got %v want %v", i, got[i], want)
			break
		}
	}
}

func TestDequantQ8_KWrongDType(t *testing.T) {
	tt, _ := tensor.Zeros(dtype.F32, []int{256}, device.CPU())
	if _, err := quant.DequantQ8_K(tt); err == nil {
		t.Error("expected error for non-Q8_K input")
	}
}
