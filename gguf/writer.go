package gguf

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
)

// Writer builds a GGUF file in-memory for tests and round-trip verification.
// Production model export is out of scope for v1; use safetensors for that.
type Writer struct {
	version  uint32
	align    int64
	meta     []kvEntry
	tensors  []twEntry
	alignSet bool
}

type kvEntry struct {
	key   string
	vtype uint32
	val   any
}

type twEntry struct {
	name  string
	dt    uint32
	dims  []uint64
	bytes []byte
	elems int
}

// NewWriter returns a fresh Writer. Default GGUF version 3, alignment 32.
func NewWriter() *Writer {
	return &Writer{version: 3, align: 32}
}

// SetAlignment sets `general.alignment` (must be > 0 and a power of 2 in
// practice). Automatically stored as a u32 metadata entry.
func (w *Writer) SetAlignment(n int64) {
	w.align = n
	w.alignSet = true
}

// AddU32 adds a uint32 metadata entry.
func (w *Writer) AddU32(key string, v uint32) {
	w.meta = append(w.meta, kvEntry{key, TypeUInt32, v})
}

// AddString adds a string metadata entry.
func (w *Writer) AddString(key, v string) {
	w.meta = append(w.meta, kvEntry{key, TypeString, v})
}

// AddF32 adds a float32 metadata entry.
func (w *Writer) AddF32(key string, v float32) {
	w.meta = append(w.meta, kvEntry{key, TypeFloat32, v})
}

// AddI64 adds an int64 metadata entry.
func (w *Writer) AddI64(key string, v int64) {
	w.meta = append(w.meta, kvEntry{key, TypeInt64, v})
}

// AddBool adds a boolean metadata entry.
func (w *Writer) AddBool(key string, v bool) {
	w.meta = append(w.meta, kvEntry{key, TypeBool, v})
}

// AddStringArray adds a GGUF_TYPE_ARRAY of strings.
func (w *Writer) AddStringArray(key string, arr []string) {
	vals := make([]any, len(arr))
	for i, s := range arr {
		vals[i] = s
	}
	w.meta = append(w.meta, kvEntry{key, TypeArray, arrayPayload{elem: TypeString, vals: vals}})
}

// arrayPayload is a typed array value used with TypeArray entries.
type arrayPayload struct {
	elem uint32
	vals []any
}

// AddTensor registers one tensor with its raw byte payload. The shape-×-dtype
// byte count is computed by the dtype helpers; the caller is responsible for
// passing consistent bytes.
func (w *Writer) AddTensor(name string, dims []int, ggmlType uint32, data []byte) error {
	elems := 1
	dims64 := make([]uint64, len(dims))
	for i, d := range dims {
		if d < 0 {
			return fmt.Errorf("gguf.Writer: negative dim %d", d)
		}
		dims64[i] = uint64(d)
		elems *= d
	}
	w.tensors = append(w.tensors, twEntry{name: name, dt: ggmlType, dims: dims64, bytes: data, elems: elems})
	return nil
}

// Bytes finalizes the file and returns the full GGUF byte stream.
func (w *Writer) Bytes() ([]byte, error) {
	// First pass: serialize header + kv + tensor-info-header bytes into a
	// staging buffer so we know the base at which tensor data begins.
	if !w.alignSet {
		w.AddU32("general.alignment", uint32(w.align))
	}

	head := &bytes.Buffer{}
	binary.Write(head, binary.LittleEndian, magic)
	binary.Write(head, binary.LittleEndian, w.version)
	binary.Write(head, binary.LittleEndian, uint64(len(w.tensors)))
	binary.Write(head, binary.LittleEndian, uint64(len(w.meta)))

	for _, kv := range w.meta {
		writeString(head, kv.key)
		binary.Write(head, binary.LittleEndian, kv.vtype)
		if err := writeValue(head, kv.vtype, kv.val); err != nil {
			return nil, err
		}
	}

	// Compute cumulative data offsets for tensors, each aligned to w.align.
	offsets := make([]uint64, len(w.tensors))
	var cursor uint64
	for i, t := range w.tensors {
		if gap := cursor % uint64(w.align); gap != 0 {
			cursor += uint64(w.align) - gap
		}
		offsets[i] = cursor
		cursor += uint64(len(t.bytes))
	}

	// Write tensor info entries.
	for i, t := range w.tensors {
		writeString(head, t.name)
		binary.Write(head, binary.LittleEndian, uint32(len(t.dims)))
		for _, d := range t.dims {
			binary.Write(head, binary.LittleEndian, d)
		}
		binary.Write(head, binary.LittleEndian, t.dt)
		binary.Write(head, binary.LittleEndian, offsets[i])
	}

	// Pad to alignment before the data section.
	if gap := int64(head.Len()) % w.align; gap != 0 {
		pad := make([]byte, w.align-gap)
		head.Write(pad)
	}

	// Append tensor payloads at their chosen offsets.
	dataStart := uint64(head.Len())
	_ = dataStart
	// dataEnd tracks bytes already written in the data section.
	dataBuf := &bytes.Buffer{}
	for i, t := range w.tensors {
		want := int(offsets[i])
		for dataBuf.Len() < want {
			dataBuf.WriteByte(0)
		}
		dataBuf.Write(t.bytes)
	}
	head.Write(dataBuf.Bytes())
	return head.Bytes(), nil
}

func writeString(w *bytes.Buffer, s string) {
	binary.Write(w, binary.LittleEndian, uint64(len(s)))
	w.WriteString(s)
}

func writeValue(w *bytes.Buffer, t uint32, v any) error {
	switch t {
	case TypeUInt8:
		binary.Write(w, binary.LittleEndian, v.(uint8))
	case TypeInt8:
		binary.Write(w, binary.LittleEndian, v.(int8))
	case TypeUInt16:
		binary.Write(w, binary.LittleEndian, v.(uint16))
	case TypeInt16:
		binary.Write(w, binary.LittleEndian, v.(int16))
	case TypeUInt32:
		binary.Write(w, binary.LittleEndian, v.(uint32))
	case TypeInt32:
		binary.Write(w, binary.LittleEndian, v.(int32))
	case TypeUInt64:
		binary.Write(w, binary.LittleEndian, v.(uint64))
	case TypeInt64:
		binary.Write(w, binary.LittleEndian, v.(int64))
	case TypeFloat32:
		binary.Write(w, binary.LittleEndian, math.Float32bits(v.(float32)))
	case TypeFloat64:
		binary.Write(w, binary.LittleEndian, math.Float64bits(v.(float64)))
	case TypeBool:
		b := byte(0)
		if v.(bool) {
			b = 1
		}
		w.WriteByte(b)
	case TypeString:
		writeString(w, v.(string))
	case TypeArray:
		arr := v.(arrayPayload)
		binary.Write(w, binary.LittleEndian, arr.elem)
		binary.Write(w, binary.LittleEndian, uint64(len(arr.vals)))
		for _, ev := range arr.vals {
			if err := writeValue(w, arr.elem, ev); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("gguf.Writer: unsupported value type %d", t)
	}
	return nil
}
