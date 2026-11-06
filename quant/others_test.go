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

func putHalf(buf []byte, f float32) {
	h := dtype.HalfFromF32(f)
	binary.LittleEndian.PutUint16(buf, uint16(h))
}

// ---- Q8_0 ----

func buildQ8_0Block(scale float32, values [32]int8) []byte {
	buf := make([]byte, quant.Q8_0BlockBytes)
	putHalf(buf[0:2], scale)
	for i, v := range values {
		buf[2+i] = byte(v)
	}
	return buf
}

func TestDequantQ8_0(t *testing.T) {
	var vals [32]int8
	for i := range vals {
		vals[i] = int8(i - 16) // -16..15
	}
	blk := buildQ8_0Block(0.25, vals)
	in, err := tensor.FromBytes(blk, dtype.Q8_0, []int{32}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	out, err := quant.DequantQ8_0(in)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](out)
	for i, v := range vals {
		want := 0.25 * float32(v)
		if math.Abs(float64(got[i]-want)) > 1e-5 {
			t.Errorf("[%d] got %v, want %v", i, got[i], want)
		}
	}
}

// ---- Q4_1 ----

func buildQ4_1Block(delta, minV float32, values [32]uint8) []byte {
	buf := make([]byte, quant.Q4_1BlockBytes)
	putHalf(buf[0:2], delta)
	putHalf(buf[2:4], minV)
	for j := range 16 {
		lo := values[j] & 0x0F
		hi := values[j+16] & 0x0F
		buf[4+j] = lo | (hi << 4)
	}
	return buf
}

func TestDequantQ4_1(t *testing.T) {
	var vals [32]uint8
	for i := range vals {
		vals[i] = uint8(i % 16) // 0..15
	}
	blk := buildQ4_1Block(0.5, 10, vals)
	in, err := tensor.FromBytes(blk, dtype.Q4_1, []int{32}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	out, _ := quant.DequantQ4_1(in)
	got := tensor.MustSlice[float32](out)
	for i, v := range vals {
		want := 0.5*float32(v) + 10
		if math.Abs(float64(got[i]-want)) > 1e-3 {
			t.Errorf("[%d] got %v, want %v", i, got[i], want)
		}
	}
}

// ---- Q5_0 ----

func buildQ5_0Block(delta float32, values [32]int8) []byte {
	buf := make([]byte, quant.Q5_0BlockBytes)
	putHalf(buf[0:2], delta)
	// Store values as signed 5-bit: raw = value + 16 (range 0..31).
	var qh uint32
	for j := range 16 {
		raw0 := uint8(values[j] + 16)
		raw1 := uint8(values[j+16] + 16)
		qh |= uint32((raw0>>4)&1) << j
		qh |= uint32((raw1>>4)&1) << (j + 16)
		buf[6+j] = (raw0 & 0x0F) | ((raw1 & 0x0F) << 4)
	}
	binary.LittleEndian.PutUint32(buf[2:6], qh)
	return buf
}

func TestDequantQ5_0(t *testing.T) {
	// Use a mix of negative and positive values in [-16, 15].
	var vals [32]int8
	for i := range vals {
		vals[i] = int8((i % 32) - 16)
	}
	blk := buildQ5_0Block(0.1, vals)
	in, _ := tensor.FromBytes(blk, dtype.Q5_0, []int{32}, device.CPU())
	out, err := quant.DequantQ5_0(in)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](out)
	for i, v := range vals {
		want := 0.1 * float32(v)
		if math.Abs(float64(got[i]-want)) > 1e-3 {
			t.Errorf("[%d] got %v, want %v", i, got[i], want)
		}
	}
}

// ---- Q5_1 ----

func buildQ5_1Block(delta, minV float32, values [32]uint8) []byte {
	buf := make([]byte, quant.Q5_1BlockBytes)
	putHalf(buf[0:2], delta)
	putHalf(buf[2:4], minV)
	var qh uint32
	for j := range 16 {
		raw0 := values[j]
		raw1 := values[j+16]
		qh |= uint32((raw0>>4)&1) << j
		qh |= uint32((raw1>>4)&1) << (j + 16)
		buf[8+j] = (raw0 & 0x0F) | ((raw1 & 0x0F) << 4)
	}
	binary.LittleEndian.PutUint32(buf[4:8], qh)
	return buf
}

func TestDequantQ5_1(t *testing.T) {
	var vals [32]uint8
	for i := range vals {
		vals[i] = uint8(i) // 0..31
	}
	blk := buildQ5_1Block(0.25, 5.0, vals)
	in, _ := tensor.FromBytes(blk, dtype.Q5_1, []int{32}, device.CPU())
	out, err := quant.DequantQ5_1(in)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](out)
	for i, v := range vals {
		want := 0.25*float32(v) + 5.0
		if math.Abs(float64(got[i]-want)) > 1e-3 {
			t.Errorf("[%d] got %v, want %v", i, got[i], want)
		}
	}
}

// ---- Q8_1 ----

func buildQ8_1Block(delta, sumMeta float32, values [32]int8) []byte {
	buf := make([]byte, quant.Q8_1BlockBytes)
	putHalf(buf[0:2], delta)
	putHalf(buf[2:4], sumMeta)
	for i, v := range values {
		buf[4+i] = byte(v)
	}
	return buf
}

func TestDequantQ8_1(t *testing.T) {
	var vals [32]int8
	for i := range vals {
		vals[i] = int8(i - 16)
	}
	blk := buildQ8_1Block(0.125, 0 /*sum ignored*/, vals)
	in, _ := tensor.FromBytes(blk, dtype.Q8_1, []int{32}, device.CPU())
	out, err := quant.DequantQ8_1(in)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](out)
	for i, v := range vals {
		want := 0.125 * float32(v)
		if math.Abs(float64(got[i]-want)) > 1e-5 {
			t.Errorf("[%d] got %v, want %v", i, got[i], want)
		}
	}
}

func TestDequantWrongDType(t *testing.T) {
	tt, _ := tensor.Zeros(dtype.F32, []int{32}, device.CPU())
	// All five should reject non-matching dtypes.
	if _, err := quant.DequantQ8_0(tt); err == nil {
		t.Error("Q8_0 should reject F32")
	}
	if _, err := quant.DequantQ4_1(tt); err == nil {
		t.Error("Q4_1 should reject F32")
	}
	if _, err := quant.DequantQ5_0(tt); err == nil {
		t.Error("Q5_0 should reject F32")
	}
	if _, err := quant.DequantQ5_1(tt); err == nil {
		t.Error("Q5_1 should reject F32")
	}
	if _, err := quant.DequantQ8_1(tt); err == nil {
		t.Error("Q8_1 should reject F32")
	}
}
