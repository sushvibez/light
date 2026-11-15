package safetensors_test

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
	"light/safetensors"
	"light/tensor"
)

// A contiguous prefix view keeps its parent's larger backing storage; Write
// must serialise only the logical bytes or every later tensor shifts (R1).
func TestWritePrefixViewKeepsLaterOffsets(t *testing.T) {
	parent, _ := tensor.New([]float32{1, 2, 3, 4}, []int{4}, device.CPU())
	a, err := parent.Narrow(0, 0, 2)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := tensor.New([]float32{9}, []int{1}, device.CPU())
	r := buildFile(t, map[string]*tensor.Tensor{"a": a, "b": b}, nil)
	loaded, _, err := safetensors.Load(r, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if got := tensor.MustSlice[float32](loaded["a"]); !slices.Equal(got, []float32{1, 2}) {
		t.Errorf("a = %v, want [1 2]", got)
	}
	if got := tensor.MustSlice[float32](loaded["b"]); !slices.Equal(got, []float32{9}) {
		t.Errorf("b = %v, want [9]", got)
	}
}

func TestWriteRejectsReservedName(t *testing.T) {
	x, _ := tensor.New([]float32{1}, []int{1}, device.CPU())
	var buf bytes.Buffer
	if err := safetensors.Write(&buf, map[string]*tensor.Tensor{"__metadata__": x}, nil); err == nil {
		t.Fatal("Write accepted the reserved __metadata__ name")
	}
}

// rawFile assembles a safetensors byte stream from a literal header and data.
func rawFile(header string, data []byte) []byte {
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	out = append(out, header...)
	return append(out, data...)
}

// readerAtOnly hides Size so the reader paths cannot learn the input length.
type readerAtOnly struct{ r io.ReaderAt }

func (o readerAtOnly) ReadAt(p []byte, off int64) (int, error) { return o.r.ReadAt(p, off) }

// Malformed headers must produce errors, never panics or huge allocations,
// through every reader variant: Load (sized and unsized input), LoadTensor,
// and the memory-mapped LoadFiles path (R13).
func TestMalformedHeadersError(t *testing.T) {
	cases := map[string]string{
		"reversed range":     `{"x":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}`,
		"negative offsets":   `{"x":{"dtype":"F32","shape":[1],"data_offsets":[-12,-8]}}`,
		"very negative":      `{"x":{"dtype":"F32","shape":[1],"data_offsets":[-9000000000000000000,-8999999999999999996]}}`,
		"negative dim":       `{"x":{"dtype":"F32","shape":[-1],"data_offsets":[0,0]}}`,
		"negative dim pair":  `{"x":{"dtype":"F32","shape":[-1,-4],"data_offsets":[0,16]}}`,
		"shape overflow":     `{"x":{"dtype":"F32","shape":[4611686018427387904,4],"data_offsets":[0,0]}}`,
		"byte size overflow": `{"x":{"dtype":"F64","shape":[2305843009213693952],"data_offsets":[0,0]}}`,
		"past data end":      `{"x":{"dtype":"F32","shape":[1000000000],"data_offsets":[0,4000000000]}}`,
		"bad dtype":          `{"x":{"dtype":"Q9","shape":[1],"data_offsets":[0,4]}}`,
		"size mismatch":      `{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}`,
	}
	dir := t.TempDir()
	i := 0
	for name, header := range cases {
		file := rawFile(header, []byte{1, 2, 3, 4})
		if _, _, err := safetensors.Load(bytes.NewReader(file), device.CPU()); err == nil {
			t.Errorf("%s: Load accepted it", name)
		}
		if _, err := safetensors.LoadTensor(bytes.NewReader(file), "x", device.CPU()); err == nil {
			t.Errorf("%s: LoadTensor accepted it", name)
		}
		if name != "past data end" { // unsized: only the short read rejects it
			if _, _, err := safetensors.Load(readerAtOnly{bytes.NewReader(file)}, device.CPU()); err == nil {
				t.Errorf("%s: unsized Load accepted it", name)
			}
		}
		path := filepath.Join(dir, fmt.Sprintf("m%d.safetensors", i))
		i++
		if err := os.WriteFile(path, file, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, _, err := safetensors.LoadFiles([]string{path}, device.CPU()); err == nil {
			t.Errorf("%s: LoadFiles (mmap) accepted it", name)
		}
	}
}

func TestHugeHeaderLengthRejected(t *testing.T) {
	for _, n := range []uint64{1 << 31, 1 << 40, 1<<63 + 5, ^uint64(0)} {
		file := binary.LittleEndian.AppendUint64(nil, n)
		file = append(file, "{}"...)
		if _, _, err := safetensors.Header(bytes.NewReader(file)); err == nil {
			t.Errorf("header length %d accepted for a 10-byte input", n)
		}
		// Unknown size: still bounded before allocating.
		if _, _, err := safetensors.Header(readerAtOnly{bytes.NewReader(file)}); err == nil {
			t.Errorf("header length %d accepted for an unsized input", n)
		}
	}
}

// Valid files still round-trip through the mapped path, including an
// empty tensor next to a non-empty one.
func TestMappedRoundTrip(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	e, _ := tensor.Zeros(dtype.F32, []int{0, 2}, device.CPU())
	var buf bytes.Buffer
	if err := safetensors.Write(&buf, map[string]*tensor.Tensor{"a": a, "e": e}, nil); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "ok.safetensors")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	got, _, err := safetensors.LoadFiles([]string{path}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if v := tensor.MustSlice[float32](got["a"]); !slices.Equal(v, []float32{1, 2, 3}) {
		t.Errorf("a = %v", v)
	}
	if got["e"].NumElements() != 0 {
		t.Errorf("e has %d elements", got["e"].NumElements())
	}
}
