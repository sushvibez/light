// Package npy reads and writes NumPy's .npy (one array) and .npz (zip of
// .npy members) files, the counterpart of Candle's `candle_core::npy`.
//
// Format: https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
//
//	"\x93NUMPY" | major | minor | header_len (u16 for v1, u32 for v2/3) | header dict | data
//
// The header is a Python dict literal with `descr`, `fortran_order` and
// `shape`. Little-endian and byte-order-independent descriptors are accepted
// (f2/f4/f8, i1/i2/i4/i8, u1/u2/u4/u8, b1 as U8); Fortran-ordered arrays are
// transposed into row-major on load. Writing always emits version 1 headers in
// C order.
package npy

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"

	"light/device"
	"light/dtype"
	"light/tensor"
)

var magic = []byte("\x93NUMPY")

// Header is the parsed .npy header.
type Header struct {
	DType        dtype.DType
	FortranOrder bool
	Shape        []int
}

// ReadHeader parses the header of a .npy stream, leaving r positioned at the
// first data byte.
func ReadHeader(r io.Reader) (Header, error) {
	var h Header
	pre := make([]byte, 8)
	if _, err := io.ReadFull(r, pre); err != nil {
		return h, fmt.Errorf("npy: read preamble: %w", err)
	}
	if !bytes.Equal(pre[:6], magic) {
		return h, fmt.Errorf("npy: bad magic %q", pre[:6])
	}
	var hlen int
	switch pre[6] {
	case 1:
		var b [2]byte
		if _, err := io.ReadFull(r, b[:]); err != nil {
			return h, err
		}
		hlen = int(binary.LittleEndian.Uint16(b[:]))
	case 2, 3:
		var b [4]byte
		if _, err := io.ReadFull(r, b[:]); err != nil {
			return h, err
		}
		hlen = int(binary.LittleEndian.Uint32(b[:]))
	default:
		return h, fmt.Errorf("npy: unsupported version %d.%d", pre[6], pre[7])
	}
	raw := make([]byte, hlen)
	if _, err := io.ReadFull(r, raw); err != nil {
		return h, fmt.Errorf("npy: read header: %w", err)
	}
	return parseHeader(string(raw))
}

// parseHeader handles the dict literal NumPy writes, e.g.
// {'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }
func parseHeader(s string) (Header, error) {
	var h Header
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "{")
	s = strings.TrimSuffix(s, "}")
	// Split on commas that are outside parentheses.
	var parts []string
	depth, start := 0, 0
	for i, c := range s {
		switch c {
		case '(':
			depth++
		case ')':
			depth--
		case ',':
			if depth == 0 {
				parts = append(parts, s[start:i])
				start = i + 1
			}
		}
	}
	parts = append(parts, s[start:])
	seen := 0
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		kv := strings.SplitN(p, ":", 2)
		if len(kv) != 2 {
			return h, fmt.Errorf("npy: malformed header entry %q", p)
		}
		key := strings.Trim(strings.TrimSpace(kv[0]), "'\"")
		val := strings.TrimSpace(kv[1])
		switch key {
		case "descr":
			dt, err := parseDescr(strings.Trim(val, "'\""))
			if err != nil {
				return h, err
			}
			h.DType = dt
		case "fortran_order":
			h.FortranOrder = val == "True"
		case "shape":
			val = strings.TrimSuffix(strings.TrimPrefix(val, "("), ")")
			for _, d := range strings.Split(val, ",") {
				d = strings.TrimSpace(d)
				if d == "" {
					continue
				}
				n, err := strconv.Atoi(d)
				if err != nil {
					return h, fmt.Errorf("npy: bad shape %q", val)
				}
				h.Shape = append(h.Shape, n)
			}
		default:
			return h, fmt.Errorf("npy: unknown header key %q", key)
		}
		seen++
	}
	if seen < 3 {
		return h, fmt.Errorf("npy: incomplete header %q", s)
	}
	return h, nil
}

func parseDescr(d string) (dtype.DType, error) {
	if len(d) < 2 {
		return 0, fmt.Errorf("npy: bad descr %q", d)
	}
	switch d[0] {
	case '<', '|', '=':
		d = d[1:]
	case '>':
		return 0, fmt.Errorf("npy: big-endian descr %q not supported", d)
	}
	switch d {
	case "f2":
		return dtype.F16, nil
	case "f4":
		return dtype.F32, nil
	case "f8":
		return dtype.F64, nil
	case "i1":
		return dtype.I8, nil
	case "i2":
		return dtype.I16, nil
	case "i4":
		return dtype.I32, nil
	case "i8":
		return dtype.I64, nil
	case "u1", "b1":
		return dtype.U8, nil
	case "u2":
		return dtype.U16, nil
	case "u4":
		return dtype.U32, nil
	case "u8":
		return dtype.U64, nil
	}
	return 0, fmt.Errorf("npy: unsupported descr %q", d)
}

func descr(dt dtype.DType) (string, error) {
	switch dt {
	case dtype.F16:
		return "<f2", nil
	case dtype.F32:
		return "<f4", nil
	case dtype.F64:
		return "<f8", nil
	case dtype.I8:
		return "|i1", nil
	case dtype.I16:
		return "<i2", nil
	case dtype.I32:
		return "<i4", nil
	case dtype.I64:
		return "<i8", nil
	case dtype.U8:
		return "|u1", nil
	case dtype.U16:
		return "<u2", nil
	case dtype.U32:
		return "<u4", nil
	case dtype.U64:
		return "<u8", nil
	}
	return "", fmt.Errorf("npy: dtype %v has no NumPy descr (BF16/F8 are not NumPy types)", dt)
}

// Read decodes one .npy stream onto dev.
func Read(r io.Reader, dev device.Device) (*tensor.Tensor, error) {
	h, err := ReadHeader(r)
	if err != nil {
		return nil, err
	}
	n := 1
	for _, d := range h.Shape {
		n *= d
	}
	data := make([]byte, h.DType.ByteSize(n))
	if _, err := io.ReadFull(r, data); err != nil {
		return nil, fmt.Errorf("npy: read %d data bytes: %w", len(data), err)
	}
	if h.FortranOrder && len(h.Shape) > 1 {
		data = fortranToC(data, h.Shape, h.DType.Size())
	}
	return tensor.FromBytes(data, h.DType, h.Shape, dev)
}

// fortranToC reorders column-major element bytes into row-major.
func fortranToC(src []byte, shape []int, esz int) []byte {
	dst := make([]byte, len(src))
	rank := len(shape)
	// Fortran strides (in elements): stride[0]=1, stride[i]=stride[i-1]*shape[i-1].
	fs := make([]int, rank)
	fs[0] = 1
	for i := 1; i < rank; i++ {
		fs[i] = fs[i-1] * shape[i-1]
	}
	idx := make([]int, rank)
	n := len(src) / esz
	for e := 0; e < n; e++ {
		off := 0
		for i := 0; i < rank; i++ {
			off += idx[i] * fs[i]
		}
		copy(dst[e*esz:(e+1)*esz], src[off*esz:(off+1)*esz])
		for i := rank - 1; i >= 0; i-- { // C-order increment
			idx[i]++
			if idx[i] < shape[i] {
				break
			}
			idx[i] = 0
		}
	}
	return dst
}

// ReadFile loads a .npy file.
func ReadFile(path string, dev device.Device) (*tensor.Tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Read(f, dev)
}

// Write encodes a CPU-contiguous tensor as .npy (format version 1).
func Write(w io.Writer, t *tensor.Tensor) error {
	if t.Device().Kind() != device.KindCPU {
		return fmt.Errorf("npy.Write: tensor must be on CPU")
	}
	if !t.IsContiguous() || t.Offset() != 0 {
		return fmt.Errorf("npy.Write: tensor must be contiguous")
	}
	d, err := descr(t.DType())
	if err != nil {
		return err
	}
	dims := t.Shape().Dims()
	parts := make([]string, len(dims))
	for i, v := range dims {
		parts[i] = strconv.Itoa(v)
	}
	shape := strings.Join(parts, ", ")
	if len(dims) == 1 {
		shape += ","
	}
	hdr := fmt.Sprintf("{'descr': '%s', 'fortran_order': False, 'shape': (%s), }", d, shape)
	// Pad so that preamble(10) + header is a multiple of 64 and ends in '\n'.
	pad := 64 - (10+len(hdr)+1)%64
	if pad == 64 {
		pad = 0
	}
	hdr += strings.Repeat(" ", pad) + "\n"
	if len(hdr) > 0xFFFF {
		return fmt.Errorf("npy.Write: header too long")
	}
	buf := make([]byte, 0, 10+len(hdr))
	buf = append(buf, magic...)
	buf = append(buf, 1, 0)
	buf = binary.LittleEndian.AppendUint16(buf, uint16(len(hdr)))
	buf = append(buf, hdr...)
	if _, err := w.Write(buf); err != nil {
		return err
	}
	_, err = w.Write(t.Storage().CPUBytes()[:t.DType().ByteSize(t.NumElements())])
	return err
}

// WriteFile saves one tensor as a .npy file.
func WriteFile(path string, t *tensor.Tensor) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := Write(f, t); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// Npz is an opened .npz archive; members are decoded lazily by name.
type Npz struct {
	zr    *zip.Reader
	close io.Closer
	files map[string]*zip.File
}

// OpenNpz opens a .npz file (stored or deflate-compressed members).
func OpenNpz(path string) (*Npz, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	zr, err := zip.NewReader(f, st.Size())
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("npz: %s: %w", path, err)
	}
	n := &Npz{zr: zr, close: f, files: make(map[string]*zip.File, len(zr.File))}
	for _, zf := range zr.File {
		n.files[strings.TrimSuffix(zf.Name, ".npy")] = zf
	}
	return n, nil
}

// Close releases the underlying file.
func (n *Npz) Close() error { return n.close.Close() }

// Names lists the member arrays in sorted order.
func (n *Npz) Names() []string {
	names := make([]string, 0, len(n.files))
	for k := range n.files {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

// Header returns the shape/dtype of a member without decoding its data.
func (n *Npz) Header(name string) (Header, error) {
	zf, ok := n.files[name]
	if !ok {
		return Header{}, fmt.Errorf("npz: no member %q", name)
	}
	rc, err := zf.Open()
	if err != nil {
		return Header{}, err
	}
	defer rc.Close()
	return ReadHeader(rc)
}

// Get decodes one member onto dev.
func (n *Npz) Get(name string, dev device.Device) (*tensor.Tensor, error) {
	zf, ok := n.files[name]
	if !ok {
		return nil, fmt.Errorf("npz: no member %q", name)
	}
	rc, err := zf.Open()
	if err != nil {
		return nil, err
	}
	defer rc.Close()
	t, err := Read(rc, dev)
	if err != nil {
		return nil, fmt.Errorf("npz: member %q: %w", name, err)
	}
	return t, nil
}

// ReadNpz loads every member of a .npz file onto dev.
func ReadNpz(path string, dev device.Device) (map[string]*tensor.Tensor, error) {
	n, err := OpenNpz(path)
	if err != nil {
		return nil, err
	}
	defer n.Close()
	out := make(map[string]*tensor.Tensor, len(n.files))
	for _, name := range n.Names() {
		t, err := n.Get(name, dev)
		if err != nil {
			return nil, err
		}
		out[name] = t
	}
	return out, nil
}

// WriteNpz writes the tensors as a stored (uncompressed, like np.savez)
// .npz archive, members in sorted name order. Tensors must be CPU-contiguous.
func WriteNpz(path string, tensors map[string]*tensor.Tensor) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	zw := zip.NewWriter(f)
	names := make([]string, 0, len(tensors))
	for k := range tensors {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, name := range names {
		w, err := zw.CreateHeader(&zip.FileHeader{Name: name + ".npy", Method: zip.Store})
		if err != nil {
			f.Close()
			return err
		}
		if err := Write(w, tensors[name]); err != nil {
			f.Close()
			return fmt.Errorf("npz: member %q: %w", name, err)
		}
	}
	if err := zw.Close(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
