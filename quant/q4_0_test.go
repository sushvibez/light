package quant_test

import (
	"encoding/binary"
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/quant"
	"light/storage"
	"light/tensor"
)

// buildQ4_0Block constructs a single Q4_0 block from a scale and 32 int4 values
// in [-8, 7]. Returns the 18-byte representation.
func buildQ4_0Block(scale float32, values [32]int8) []byte {
	buf := make([]byte, 18)
	// Write scale as F16.
	h := dtype.HalfFromF32(scale)
	binary.LittleEndian.PutUint16(buf[0:2], uint16(h))
	// Pack: qs[j] low nibble = values[j] + 8, high nibble = values[j+16] + 8.
	for j := 0; j < 16; j++ {
		lo := uint8(values[j] + 8)
		hi := uint8(values[j+16] + 8)
		buf[2+j] = (lo & 0x0F) | ((hi & 0x0F) << 4)
	}
	return buf
}

// q4_0Tensor builds a Q4_0 tensor from raw block bytes.
func q4_0Tensor(t *testing.T, bytes []byte, nelems int, dev device.Device) *tensor.Tensor {
	t.Helper()
	s := storage.WrapCPU(dtype.Q4_0, nelems, bytes)
	// Can't use tensor.New (generic); instead build via FromBytes.
	tt, err := tensor.FromBytes(bytes, dtype.Q4_0, []int{nelems}, dev)
	if err != nil {
		t.Fatal(err)
	}
	_ = s
	return tt
}

func TestDequantQ4_0Basic(t *testing.T) {
	// One block: scale=1.0, values = [0, 1, 2, 3, …, -8, -7, -6, …] etc.
	var vals [32]int8
	for i := 0; i < 16; i++ {
		vals[i] = int8(i - 8) // -8 through 7
	}
	for i := 16; i < 32; i++ {
		vals[i] = int8(i - 16 - 8) // same sequence
	}
	scale := float32(0.5)
	block := buildQ4_0Block(scale, vals)

	in := q4_0Tensor(t, block, 32, device.CPU())
	out, err := quant.DequantQ4_0(in)
	if err != nil {
		t.Fatal(err)
	}
	if out.DType() != dtype.F32 {
		t.Errorf("dtype = %v", out.DType())
	}
	if out.NumElements() != 32 {
		t.Errorf("nelems = %d", out.NumElements())
	}
	got := tensor.MustSlice[float32](out)
	// Check positions 0..15 = scale * (value + 0) ... wait, the dequant is
	// scale * (nibble_unsigned - 8). We stored nibble = value + 8, so dequant
	// recovers value. With scale=0.5, dequant[i] = 0.5 * i_raw.
	for i := 0; i < 16; i++ {
		want := scale * float32(vals[i])
		if math.Abs(float64(got[i]-want)) > 1e-5 {
			t.Errorf("got[%d] = %v, want %v", i, got[i], want)
		}
	}
	// Positions 16..31 come from high nibbles.
	for i := 16; i < 32; i++ {
		want := scale * float32(vals[i])
		if math.Abs(float64(got[i]-want)) > 1e-5 {
			t.Errorf("got[%d] = %v, want %v", i, got[i], want)
		}
	}
}

func TestDequantQ4_0MultipleBlocks(t *testing.T) {
	var vals1, vals2 [32]int8
	for i := range vals1 {
		vals1[i] = 1
		vals2[i] = -2
	}
	b1 := buildQ4_0Block(2.0, vals1)
	b2 := buildQ4_0Block(3.0, vals2)
	allBytes := append(append([]byte(nil), b1...), b2...)

	in := q4_0Tensor(t, allBytes, 64, device.CPU())
	out, err := quant.DequantQ4_0(in)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](out)
	// First 32 elements: scale 2.0 * value 1 = 2.0
	for i := 0; i < 32; i++ {
		if math.Abs(float64(got[i]-2.0)) > 1e-5 {
			t.Errorf("block 0 [%d] = %v, want 2.0", i, got[i])
		}
	}
	// Next 32 elements: scale 3.0 * value -2 = -6.0
	for i := 32; i < 64; i++ {
		if math.Abs(float64(got[i]-(-6.0))) > 1e-5 {
			t.Errorf("block 1 [%d] = %v, want -6.0", i, got[i])
		}
	}
}

func TestDequantQ4_0WrongDType(t *testing.T) {
	tt, _ := tensor.Zeros(dtype.F32, []int{4}, device.CPU())
	if _, err := quant.DequantQ4_0(tt); err == nil {
		t.Error("expected dtype error")
	}
}

func TestDequantQ4_0BadSize(t *testing.T) {
	// 17 elements aren't a multiple of 32.
	buf := make([]byte, 18) // just one block's worth of bytes
	in := q4_0Tensor(t, buf, 17, device.CPU())
	if _, err := quant.DequantQ4_0(in); err == nil {
		t.Error("expected size error")
	}
}
