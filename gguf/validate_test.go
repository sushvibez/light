package gguf_test

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"testing"

	"light/device"
	"light/gguf"
	"light/tensor"
)

// unsized hides Size so Open cannot learn the input length; unlike the
// byteReaderAt helper it reports short reads as io.EOF, per io.ReaderAt.
type unsized struct{ r *bytes.Reader }

func (u unsized) ReadAt(p []byte, off int64) (int, error) { return u.r.ReadAt(p, off) }

// ggufStream builds a v3 header with the given counts followed by body.
func ggufStream(nTensors, nKV uint64, body []byte) []byte {
	b := binary.LittleEndian.AppendUint32(nil, 0x46554747)
	b = binary.LittleEndian.AppendUint32(b, 3)
	b = binary.LittleEndian.AppendUint64(b, nTensors)
	b = binary.LittleEndian.AppendUint64(b, nKV)
	return append(b, body...)
}

func str(b []byte, s string) []byte {
	b = binary.LittleEndian.AppendUint64(b, uint64(len(s)))
	return append(b, s...)
}

// tensorInfo appends one tensor-info entry (F32, ggml type 0).
func tensorInfo(b []byte, name string, dims []uint64, off uint64) []byte {
	b = str(b, name)
	b = binary.LittleEndian.AppendUint32(b, uint32(len(dims)))
	for _, d := range dims {
		b = binary.LittleEndian.AppendUint64(b, d)
	}
	b = binary.LittleEndian.AppendUint32(b, 0)
	return binary.LittleEndian.AppendUint64(b, off)
}

// Malformed counts, ranks, products and offsets must be errors before any
// allocation sized by them, never panics (R27).
func TestMalformedGGUFErrors(t *testing.T) {
	arrayKV := func(elemT uint32, n uint64) []byte {
		b := str(nil, "k")
		b = binary.LittleEndian.AppendUint32(b, gguf.TypeArray)
		b = binary.LittleEndian.AppendUint32(b, elemT)
		return binary.LittleEndian.AppendUint64(b, n)
	}
	// Arrays nested 100 deep, each claiming one element.
	var nested []byte
	nested = str(nested, "deep")
	nested = binary.LittleEndian.AppendUint32(nested, gguf.TypeArray)
	for range 100 {
		nested = binary.LittleEndian.AppendUint32(nested, gguf.TypeArray)
		nested = binary.LittleEndian.AppendUint64(nested, 1)
	}
	cases := map[string][]byte{
		"array count MaxUint64":   ggufStream(0, 1, arrayKV(gguf.TypeUInt8, math.MaxUint64)),
		"array of u64 too long":   ggufStream(0, 1, append(arrayKV(gguf.TypeUInt64, 4), make([]byte, 16)...)),
		"array of arrays too big": ggufStream(0, 1, arrayKV(gguf.TypeArray, 1<<40)),
		"kv count MaxUint64":      ggufStream(0, math.MaxUint64, nil),
		"tensor count MaxUint64":  ggufStream(math.MaxUint64, 0, nil),
		"deep nesting":            ggufStream(0, 1, nested),
		"huge rank":               ggufStream(1, 0, binary.LittleEndian.AppendUint32(str(nil, "t"), math.MaxUint32)),
		"dims product overflow":   ggufStream(1, 0, tensorInfo(nil, "t", []uint64{1 << 32, 1 << 32}, 0)),
		"byte size overflow":      ggufStream(1, 0, tensorInfo(nil, "t", []uint64{1 << 62}, 0)),
		"offset overflow":         ggufStream(1, 0, tensorInfo(nil, "t", []uint64{1}, math.MaxUint64-2)),
		"offset near MaxInt64":    ggufStream(1, 0, tensorInfo(nil, "t", []uint64{4}, math.MaxInt64-8)),
		"string length too big":   ggufStream(0, 1, binary.LittleEndian.AppendUint64(nil, 1<<29)),
	}
	for name, file := range cases {
		if _, err := gguf.Open(bytes.NewReader(file)); err == nil {
			t.Errorf("%s: Open (sized) accepted it", name)
		}
		if _, err := gguf.Open(unsized{bytes.NewReader(file)}); err == nil {
			t.Errorf("%s: Open (unsized) accepted it", name)
		}
	}
	// Tensor data past the end of a sized input is rejected at Open.
	file := ggufStream(1, 0, tensorInfo(nil, "t", []uint64{1000}, 0))
	if _, err := gguf.Open(bytes.NewReader(file)); err == nil {
		t.Error("tensor data past end of input accepted")
	}
}

// Writer output still opens with the size known (the end-of-data check
// must accept every valid file), and loads.
func TestSizedRoundTrip(t *testing.T) {
	w := gguf.NewWriter()
	w.AddString("general.name", "toy")
	f32 := func(vs ...float32) []byte {
		var out []byte
		for _, v := range vs {
			out = binary.LittleEndian.AppendUint32(out, math.Float32bits(v))
		}
		return out
	}
	if err := w.AddTensor("a", []int{3}, 0 /*F32*/, f32(1, 2, 3)); err != nil {
		t.Fatal(err)
	}
	if err := w.AddTensor("b", []int{2}, 0 /*F32*/, f32(4, 5)); err != nil {
		t.Fatal(err)
	}
	data, err := w.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	f, err := gguf.Open(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	got, err := f.Tensor("b", device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if v := tensor.MustSlice[float32](got); len(v) != 2 || v[0] != 4 || v[1] != 5 {
		t.Errorf("b = %v", v)
	}
}

var _ io.ReaderAt = unsized{}
