// Package safetensors implements a pure-Go reader for Hugging Face's
// safetensors file format.
//
// Layout:
//
//	[ 8 bytes LE uint64: header_len ]
//	[ header_len bytes: JSON header ]
//	[ remainder: concatenated tensor data ]
//
// The JSON header maps each tensor name to {dtype, shape, data_offsets}
// where data_offsets are [start, end) byte ranges inside the data section
// (i.e. relative to the first byte AFTER the header).
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// tensorInfo is the JSON entry for a single tensor in the header.
type tensorInfo struct {
	DType       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets [2]int `json:"data_offsets"`
}

// Header parses and returns just the JSON header of a safetensors file,
// without loading any tensor bytes. Useful for listing contents.
func Header(r io.ReaderAt) (map[string]tensorInfo, map[string]string, error) {
	var lenBuf [8]byte
	if _, err := r.ReadAt(lenBuf[:], 0); err != nil {
		return nil, nil, fmt.Errorf("safetensors: read header length: %w", err)
	}
	headerLen := int64(binary.LittleEndian.Uint64(lenBuf[:]))
	if headerLen <= 0 || headerLen > 1<<32 {
		return nil, nil, fmt.Errorf("safetensors: implausible header length %d", headerLen)
	}
	buf := make([]byte, headerLen)
	if _, err := r.ReadAt(buf, 8); err != nil {
		return nil, nil, fmt.Errorf("safetensors: read header: %w", err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(buf, &raw); err != nil {
		return nil, nil, fmt.Errorf("safetensors: parse header JSON: %w", err)
	}
	tensors := make(map[string]tensorInfo, len(raw))
	var metadata map[string]string
	for name, msg := range raw {
		if name == "__metadata__" {
			if err := json.Unmarshal(msg, &metadata); err != nil {
				return nil, nil, fmt.Errorf("safetensors: parse __metadata__: %w", err)
			}
			continue
		}
		var info tensorInfo
		if err := json.Unmarshal(msg, &info); err != nil {
			return nil, nil, fmt.Errorf("safetensors: parse %q: %w", name, err)
		}
		tensors[name] = info
	}
	return tensors, metadata, nil
}

// Load reads all tensors in the file and allocates them on the given device.
func Load(r io.ReaderAt, dev device.Device) (map[string]*tensor.Tensor, map[string]string, error) {
	infos, metadata, err := Header(r)
	if err != nil {
		return nil, nil, err
	}

	// Determine the byte offset at which tensor data begins: 8 + headerLen.
	var lenBuf [8]byte
	_, _ = r.ReadAt(lenBuf[:], 0)
	dataStart := int64(8 + binary.LittleEndian.Uint64(lenBuf[:]))

	out := make(map[string]*tensor.Tensor, len(infos))
	for name, info := range infos {
		dt, err := parseDType(info.DType)
		if err != nil {
			return nil, nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}
		nBytes := info.DataOffsets[1] - info.DataOffsets[0]
		if nBytes < 0 {
			return nil, nil, fmt.Errorf("safetensors: tensor %q: negative byte range", name)
		}
		expected := dt.ByteSize(product(info.Shape))
		if nBytes != expected {
			return nil, nil, fmt.Errorf("safetensors: tensor %q: header says %d bytes, shape/dtype imply %d", name, nBytes, expected)
		}
		buf := make([]byte, nBytes)
		if _, err := r.ReadAt(buf, dataStart+int64(info.DataOffsets[0])); err != nil {
			return nil, nil, fmt.Errorf("safetensors: read %q: %w", name, err)
		}
		t, err := tensor.FromBytes(buf, dt, info.Shape, dev)
		if err != nil {
			return nil, nil, fmt.Errorf("safetensors: build %q: %w", name, err)
		}
		out[name] = t
	}
	return out, metadata, nil
}

// LoadTensor loads only one named tensor, avoiding allocation for the rest.
// Useful for very large files where you only need a subset of the tensors.
func LoadTensor(r io.ReaderAt, name string, dev device.Device) (*tensor.Tensor, error) {
	infos, _, err := Header(r)
	if err != nil {
		return nil, err
	}
	info, ok := infos[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: no tensor named %q", name)
	}
	dt, err := parseDType(info.DType)
	if err != nil {
		return nil, err
	}
	var lenBuf [8]byte
	_, _ = r.ReadAt(lenBuf[:], 0)
	dataStart := int64(8 + binary.LittleEndian.Uint64(lenBuf[:]))
	nBytes := info.DataOffsets[1] - info.DataOffsets[0]
	buf := make([]byte, nBytes)
	if _, err := r.ReadAt(buf, dataStart+int64(info.DataOffsets[0])); err != nil {
		return nil, err
	}
	return tensor.FromBytes(buf, dt, info.Shape, dev)
}

// Write serialises a map of tensors to the safetensors format. Tensor data is
// concatenated in name-sorted order to produce a deterministic layout, which
// matches how HuggingFace's Python implementation writes files.
//
// Only CPU tensors are supported; callers should `t.To(device.CPU())` first.
func Write(w io.Writer, tensors map[string]*tensor.Tensor, metadata map[string]string) error {
	// Pre-validate: all tensors must be CPU and contiguous.
	names := make([]string, 0, len(tensors))
	for n, t := range tensors {
		if t.Device().Kind() != device.KindCPU {
			return fmt.Errorf("safetensors.Write: tensor %q not on CPU", n)
		}
		if !t.IsContiguous() || t.Offset() != 0 {
			return fmt.Errorf("safetensors.Write: tensor %q not contiguous", n)
		}
		names = append(names, n)
	}
	// Sort for determinism.
	sortStrings(names)

	// Build header.
	type entry struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets [2]int `json:"data_offsets"`
	}
	header := make(map[string]any, len(tensors)+1)
	if len(metadata) > 0 {
		header["__metadata__"] = metadata
	}
	offset := 0
	for _, n := range names {
		t := tensors[n]
		size := t.DType().ByteSize(t.NumElements())
		header[n] = entry{
			DType:       dtypeString(t.DType()),
			Shape:       t.Shape().Dims(),
			DataOffsets: [2]int{offset, offset + size},
		}
		offset += size
	}
	headerJSON, err := json.Marshal(header)
	if err != nil {
		return err
	}
	// Write: [8-byte length][header JSON][tensor data]
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(headerJSON)))
	if _, err := w.Write(lenBuf[:]); err != nil {
		return err
	}
	if _, err := w.Write(headerJSON); err != nil {
		return err
	}
	for _, n := range names {
		t := tensors[n]
		if _, err := w.Write(t.Storage().CPUBytes()); err != nil {
			return err
		}
	}
	return nil
}

// --- helpers ---

func parseDType(s string) (dtype.DType, error) {
	switch strings.ToUpper(s) {
	case "F64":
		return dtype.F64, nil
	case "F32":
		return dtype.F32, nil
	case "F16":
		return dtype.F16, nil
	case "BF16":
		return dtype.BF16, nil
	case "F8_E4M3":
		return dtype.F8E4M3, nil
	case "I64":
		return dtype.I64, nil
	case "I32":
		return dtype.I32, nil
	case "I16":
		return dtype.I16, nil
	case "I8":
		return dtype.I8, nil
	case "U8":
		return dtype.U8, nil
	case "U16":
		return dtype.U16, nil
	case "U32":
		return dtype.U32, nil
	case "U64":
		return dtype.U64, nil
	}
	return 0, fmt.Errorf("unsupported safetensors dtype %q", s)
}

func dtypeString(dt dtype.DType) string {
	switch dt {
	case dtype.F64:
		return "F64"
	case dtype.F32:
		return "F32"
	case dtype.F16:
		return "F16"
	case dtype.BF16:
		return "BF16"
	case dtype.F8E4M3:
		return "F8_E4M3"
	case dtype.I64:
		return "I64"
	case dtype.I32:
		return "I32"
	case dtype.I16:
		return "I16"
	case dtype.I8:
		return "I8"
	case dtype.U8:
		return "U8"
	case dtype.U16:
		return "U16"
	case dtype.U32:
		return "U32"
	case dtype.U64:
		return "U64"
	}
	return "UNKNOWN"
}

func product(dims []int) int {
	p := 1
	for _, d := range dims {
		p *= d
	}
	return p
}

// sortStrings is a local sort to avoid an extra import in the hot path.
func sortStrings(s []string) {
	// Insertion sort — fine for typical safetensors size (100s of names).
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j-1] > s[j]; j-- {
			s[j-1], s[j] = s[j], s[j-1]
		}
	}
}
