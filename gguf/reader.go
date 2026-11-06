// Package gguf reads the llama.cpp GGUF model format.
//
// Header, metadata key-values, and tensor-info table are parsed up-front at
// Open time; tensor data stays on disk (or in memory) and is materialized per
// request by [File.Tensor], which interprets the raw bytes through
// [tensor.FromBytes]. Quantized tensors are returned in their native block
// dtype and must be dequantized via the [light/quant] package.
package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// TensorInfo describes one tensor's shape, dtype, and position in the file's
// raw data section. The Offset is relative to dataBase.
type TensorInfo struct {
	Name   string
	DType  dtype.DType
	Dims   []int
	Offset uint64
	bytes  int // cached byte length of the tensor payload
}

// File is a parsed GGUF file. Tensor data is read lazily from the underlying
// io.ReaderAt.
type File struct {
	Version  uint32
	Meta     map[string]any
	Tensors  map[string]*TensorInfo
	Order    []string // tensor names in declaration order
	KeyOrder []string // metadata keys in declaration order

	ra       io.ReaderAt
	dataBase int64
}

// Open parses a GGUF header + metadata + tensor info table from ra.
func Open(ra io.ReaderAt) (*File, error) {
	r := &cursor{ra: ra}

	m, err := r.u32()
	if err != nil {
		return nil, fmt.Errorf("gguf: read magic: %w", err)
	}
	if m != magic {
		return nil, fmt.Errorf("gguf: bad magic %#x (want %#x)", m, magic)
	}
	ver, err := r.u32()
	if err != nil {
		return nil, fmt.Errorf("gguf: read version: %w", err)
	}
	if ver < MinVersion || ver > MaxVersion {
		return nil, fmt.Errorf("gguf: version %d not in [%d, %d]", ver, MinVersion, MaxVersion)
	}
	nTensors, err := r.u64()
	if err != nil {
		return nil, err
	}
	nKV, err := r.u64()
	if err != nil {
		return nil, err
	}

	f := &File{
		Version: ver,
		Meta:    make(map[string]any, nKV),
		Tensors: make(map[string]*TensorInfo, nTensors),
		ra:      ra,
	}

	for i := range nKV {
		key, err := r.string()
		if err != nil {
			return nil, fmt.Errorf("gguf: kv[%d] key: %w", i, err)
		}
		vtype, err := r.u32()
		if err != nil {
			return nil, fmt.Errorf("gguf: kv[%q] type: %w", key, err)
		}
		val, err := r.value(vtype)
		if err != nil {
			return nil, fmt.Errorf("gguf: kv[%q] value: %w", key, err)
		}
		f.Meta[key] = val
		f.KeyOrder = append(f.KeyOrder, key)
	}

	for i := range nTensors {
		name, err := r.string()
		if err != nil {
			return nil, fmt.Errorf("gguf: tensor[%d] name: %w", i, err)
		}
		nd, err := r.u32()
		if err != nil {
			return nil, err
		}
		dims := make([]int, nd)
		var nElems int64 = 1
		for j := range dims {
			d, err := r.u64()
			if err != nil {
				return nil, err
			}
			if d > math.MaxInt64 {
				return nil, fmt.Errorf("gguf: tensor[%q] dim %d too large", name, d)
			}
			dims[j] = int(d)
			nElems *= int64(d)
		}
		ggml, err := r.u32()
		if err != nil {
			return nil, err
		}
		dt, err := ggmlToDType(ggml)
		if err != nil {
			return nil, fmt.Errorf("gguf: tensor[%q]: %w", name, err)
		}
		off, err := r.u64()
		if err != nil {
			return nil, err
		}
		info := &TensorInfo{
			Name:   name,
			DType:  dt,
			Dims:   dims,
			Offset: off,
			bytes:  dt.ByteSize(int(nElems)),
		}
		f.Tensors[name] = info
		f.Order = append(f.Order, name)
	}

	// Tensor data begins after the info table, padded up to `general.alignment`.
	align := int64(32)
	if v, ok := f.Meta["general.alignment"]; ok {
		if n, ok := toInt64(v); ok && n > 0 {
			align = n
		}
	}
	f.dataBase = (r.pos + align - 1) / align * align
	return f, nil
}

// Tensor loads one tensor by name onto dev. Returns an error for unknown names
// or if the raw byte length does not match the declared shape × dtype.
func (f *File) Tensor(name string, dev device.Device) (*tensor.Tensor, error) {
	info, ok := f.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("gguf: tensor %q not found", name)
	}
	buf := make([]byte, info.bytes)
	absOff := f.dataBase + int64(info.Offset)
	if _, err := f.ra.ReadAt(buf, absOff); err != nil {
		return nil, fmt.Errorf("gguf: read tensor %q: %w", name, err)
	}
	return tensor.FromBytes(buf, info.DType, info.Dims, dev)
}

// TensorRowMajor is Tensor with the dimensions reversed into row-major
// order. GGUF lists dims innermost-first (ne[0] = contiguous axis), so a
// Linear weight stored as [in, out] in the file is the (out, in) matrix our
// layers expect; TensorInfo.Dims keeps the file order.
func (f *File) TensorRowMajor(name string, dev device.Device) (*tensor.Tensor, error) {
	info, ok := f.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("gguf: tensor %q not found", name)
	}
	buf, err := f.TensorBytes(name)
	if err != nil {
		return nil, err
	}
	dims := make([]int, len(info.Dims))
	for i, d := range info.Dims {
		dims[len(dims)-1-i] = d
	}
	return tensor.FromBytes(buf, info.DType, dims, dev)
}

// TensorBytes returns the raw (possibly quantized) bytes of one tensor.
func (f *File) TensorBytes(name string) ([]byte, error) {
	info, ok := f.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("gguf: tensor %q not found", name)
	}
	buf := make([]byte, info.bytes)
	if _, err := f.ra.ReadAt(buf, f.dataBase+int64(info.Offset)); err != nil {
		return nil, fmt.Errorf("gguf: read tensor %q: %w", name, err)
	}
	return buf, nil
}

// String returns a metadata string value if present.
func (f *File) String(key string) (string, bool) {
	v, ok := f.Meta[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}

// Int64 coerces any integer metadata value to int64.
func (f *File) Int64(key string) (int64, bool) {
	v, ok := f.Meta[key]
	if !ok {
		return 0, false
	}
	return toInt64(v)
}

// Float64 coerces F32/F64 metadata values to float64.
func (f *File) Float64(key string) (float64, bool) {
	v, ok := f.Meta[key]
	if !ok {
		return 0, false
	}
	switch x := v.(type) {
	case float32:
		return float64(x), true
	case float64:
		return x, true
	}
	return 0, false
}

// toInt64 coerces any signed/unsigned integer to int64, returning (0, false)
// for other types or unsigned values that overflow.
func toInt64(v any) (int64, bool) {
	switch x := v.(type) {
	case uint8:
		return int64(x), true
	case int8:
		return int64(x), true
	case uint16:
		return int64(x), true
	case int16:
		return int64(x), true
	case uint32:
		return int64(x), true
	case int32:
		return int64(x), true
	case int64:
		return x, true
	case uint64:
		if x > math.MaxInt64 {
			return 0, false
		}
		return int64(x), true
	}
	return 0, false
}

// cursor is a sequential little-endian reader over an io.ReaderAt.
type cursor struct {
	ra  io.ReaderAt
	pos int64
}

func (c *cursor) read(n int) ([]byte, error) {
	buf := make([]byte, n)
	if _, err := c.ra.ReadAt(buf, c.pos); err != nil {
		return nil, err
	}
	c.pos += int64(n)
	return buf, nil
}

func (c *cursor) u8() (uint8, error) {
	b, err := c.read(1)
	if err != nil {
		return 0, err
	}
	return b[0], nil
}
func (c *cursor) u16() (uint16, error) {
	b, err := c.read(2)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint16(b), nil
}
func (c *cursor) u32() (uint32, error) {
	b, err := c.read(4)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b), nil
}
func (c *cursor) u64() (uint64, error) {
	b, err := c.read(8)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b), nil
}
func (c *cursor) f32() (float32, error) {
	x, err := c.u32()
	return math.Float32frombits(x), err
}
func (c *cursor) f64() (float64, error) {
	x, err := c.u64()
	return math.Float64frombits(x), err
}

func (c *cursor) string() (string, error) {
	n, err := c.u64()
	if err != nil {
		return "", err
	}
	if n > 1<<30 {
		return "", fmt.Errorf("gguf: string length %d too large", n)
	}
	b, err := c.read(int(n))
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// value decodes one typed value for the given GGUF type tag.
func (c *cursor) value(t uint32) (any, error) {
	switch t {
	case TypeUInt8:
		return c.u8()
	case TypeInt8:
		v, err := c.u8()
		return int8(v), err
	case TypeUInt16:
		return c.u16()
	case TypeInt16:
		v, err := c.u16()
		return int16(v), err
	case TypeUInt32:
		return c.u32()
	case TypeInt32:
		v, err := c.u32()
		return int32(v), err
	case TypeUInt64:
		return c.u64()
	case TypeInt64:
		v, err := c.u64()
		return int64(v), err
	case TypeFloat32:
		return c.f32()
	case TypeFloat64:
		return c.f64()
	case TypeBool:
		v, err := c.u8()
		return v != 0, err
	case TypeString:
		return c.string()
	case TypeArray:
		elemT, err := c.u32()
		if err != nil {
			return nil, err
		}
		n, err := c.u64()
		if err != nil {
			return nil, err
		}
		arr := make([]any, n)
		for i := range arr {
			arr[i], err = c.value(elemT)
			if err != nil {
				return nil, err
			}
		}
		return arr, nil
	}
	return nil, fmt.Errorf("gguf: unknown value type %d", t)
}
