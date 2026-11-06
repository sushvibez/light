package gguf_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/gguf"
	"light/quant"
	"light/tensor"
)

// byteReaderAt wraps a []byte as an io.ReaderAt for the Open call.
type byteReaderAt struct{ b []byte }

func (r *byteReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if off >= int64(len(r.b)) {
		return 0, nil
	}
	return copy(p, r.b[off:]), nil
}

func TestGGUFRoundTripScalar(t *testing.T) {
	w := gguf.NewWriter()
	w.AddString("general.name", "toy-model")
	w.AddU32("general.architecture_version", 1)
	w.AddI64("block_count", 4)
	w.AddF32("rope_theta", 10000.0)
	w.AddBool("is_quantized", true)
	w.AddStringArray("tokenizer.tokens", []string{"hello", "world"})

	// Add an F32 tensor with known values.
	dims := []int{2, 3}
	vals := []float32{1, 2, 3, 4, 5, 6}
	buf := &bytes.Buffer{}
	for _, v := range vals {
		binary.Write(buf, binary.LittleEndian, math.Float32bits(v))
	}
	if err := w.AddTensor("weight", dims, 0 /*F32*/, buf.Bytes()); err != nil {
		t.Fatal(err)
	}

	data, err := w.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	f, err := gguf.Open(&byteReaderAt{b: data})
	if err != nil {
		t.Fatal(err)
	}

	if got, _ := f.String("general.name"); got != "toy-model" {
		t.Errorf("name = %q", got)
	}
	if got, _ := f.Int64("block_count"); got != 4 {
		t.Errorf("block_count = %d", got)
	}
	if got, _ := f.Float64("rope_theta"); got != 10000.0 {
		t.Errorf("rope_theta = %v", got)
	}
	if v, ok := f.Meta["is_quantized"]; !ok || v.(bool) != true {
		t.Errorf("is_quantized = %v", v)
	}
	if v, ok := f.Meta["tokenizer.tokens"]; !ok {
		t.Fatalf("missing tokens array")
	} else {
		arr := v.([]any)
		if len(arr) != 2 || arr[0].(string) != "hello" || arr[1].(string) != "world" {
			t.Errorf("tokens = %v", arr)
		}
	}

	tt, err := f.Tensor("weight", device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if tt.DType() != dtype.F32 {
		t.Errorf("dtype = %v", tt.DType())
	}
	if !equalInts(tt.Shape().Dims(), dims) {
		t.Errorf("shape = %v", tt.Shape().Dims())
	}
	got := tensor.MustSlice[float32](tt)
	for i, v := range vals {
		if got[i] != v {
			t.Errorf("[%d] %v != %v", i, got[i], v)
		}
	}
}

func TestGGUFQuantTensorLoadsAndDequantizes(t *testing.T) {
	// Build a Q4_0 block by hand, wrap it in a GGUF file, then verify that
	// reading + DequantQ4_0 yields the expected F32 values. Q4_0's signed
	// 4-bit range is [-8, 7].
	scale := float32(0.5)
	var vals [32]int8
	for i := range vals {
		vals[i] = int8((i % 16) - 8)
	}
	block := buildQ4_0BlockRaw(scale, vals)

	w := gguf.NewWriter()
	w.AddString("general.name", "q4-test")
	if err := w.AddTensor("w", []int{32}, 2 /*Q4_0*/, block); err != nil {
		t.Fatal(err)
	}
	data, err := w.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	f, err := gguf.Open(&byteReaderAt{b: data})
	if err != nil {
		t.Fatal(err)
	}

	qt, err := f.Tensor("w", device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if qt.DType() != dtype.Q4_0 {
		t.Fatalf("dtype = %v, want Q4_0", qt.DType())
	}
	dq, err := quant.DequantQ4_0(qt)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](dq)
	for i, v := range vals {
		want := scale * float32(v)
		if math.Abs(float64(got[i]-want)) > 1e-5 {
			t.Errorf("[%d] got %v want %v", i, got[i], want)
		}
	}
}

func TestGGUFBadMagic(t *testing.T) {
	data := make([]byte, 32)
	copy(data, []byte{0, 0, 0, 0})
	if _, err := gguf.Open(&byteReaderAt{b: data}); err == nil {
		t.Error("expected error on bad magic")
	}
}

func TestGGUFTensorMissing(t *testing.T) {
	w := gguf.NewWriter()
	data, _ := w.Bytes()
	f, err := gguf.Open(&byteReaderAt{b: data})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Tensor("nope", device.CPU()); err == nil {
		t.Error("expected error for missing tensor")
	}
}

func TestGGUFEmptyFile(t *testing.T) {
	// Minimum valid GGUF: header only, zero metadata, zero tensors.
	w := gguf.NewWriter()
	data, err := w.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	f, err := gguf.Open(&byteReaderAt{b: data})
	if err != nil {
		t.Fatal(err)
	}
	if len(f.Tensors) != 0 {
		t.Errorf("expected no tensors, got %d", len(f.Tensors))
	}
	// Only the auto-added alignment key should be present.
	if len(f.Meta) != 1 {
		t.Errorf("expected only alignment key, got %v", f.KeyOrder)
	}
}

// buildQ4_0BlockRaw packs a Q4_0 block identically to the quant package's
// builder — duplicated here so the gguf_test package has no test-helper cross-
// dependency on quant_test.
func buildQ4_0BlockRaw(scale float32, values [32]int8) []byte {
	buf := make([]byte, 18)
	h := halfFromF32(scale)
	binary.LittleEndian.PutUint16(buf[0:2], h)
	for j := range 16 {
		lo := uint8(values[j]+8) & 0x0F
		hi := uint8(values[j+16]+8) & 0x0F
		buf[2+j] = lo | (hi << 4)
	}
	return buf
}

// halfFromF32 converts F32 to F16 bits via the dtype package's helper.
func halfFromF32(f float32) uint16 { return uint16(dtype.HalfFromF32(f)) }

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
