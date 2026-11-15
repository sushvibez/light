package npy

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"light/device"
)

// npyV1 assembles a version-1 .npy stream from a header dict and data.
func npyV1(dict string, data []byte) []byte {
	out := append([]byte{}, magic...)
	out = append(out, 1, 0)
	out = binary.LittleEndian.AppendUint16(out, uint16(len(dict)))
	out = append(out, dict...)
	return append(out, data...)
}

// Malformed headers must be errors from ReadHeader, Read and NPZ members,
// never panics or allocations sized by untrusted fields (R26).
func TestMalformedNpyErrors(t *testing.T) {
	cases := map[string]string{
		"negative dim":         `{'descr': '<f4', 'fortran_order': False, 'shape': (-1,), }`,
		"negative dims pair":   `{'descr': '<f4', 'fortran_order': False, 'shape': (-2, -3), }`,
		"count overflow":       `{'descr': '<f4', 'fortran_order': False, 'shape': (4611686018427387904, 4), }`,
		"byte overflow":        `{'descr': '<f8', 'fortran_order': False, 'shape': (2305843009213693952,), }`,
		"duplicate hides key":  `{'descr': '<f4', 'descr': '<f4', 'shape': (1,), }`,
		"duplicate shape":      `{'descr': '<f4', 'fortran_order': False, 'shape': (1,), 'shape': (2,), }`,
		"bad fortran_order":    `{'descr': '<f4', 'fortran_order': maybe, 'shape': (1,), }`,
		"missing shape":        `{'descr': '<f4', 'fortran_order': False, }`,
		"huge short data":      `{'descr': '<f4', 'fortran_order': False, 'shape': (1000000000,), }`,
		"huge fortran short":   `{'descr': '<f8', 'fortran_order': True, 'shape': (100000, 100000), }`,
		"unsupported descr":    `{'descr': '<c8', 'fortran_order': False, 'shape': (1,), }`,
		"malformed shape item": `{'descr': '<f4', 'fortran_order': False, 'shape': (1x,), }`,
	}
	dir := t.TempDir()
	for name, dict := range cases {
		file := npyV1(dict, []byte{0, 0, 0, 0})
		if name != "huge short data" && name != "huge fortran short" {
			if _, err := ReadHeader(bytes.NewReader(file)); err == nil {
				t.Errorf("%s: ReadHeader accepted it", name)
			}
		}
		if _, err := Read(bytes.NewReader(file), device.CPU()); err == nil {
			t.Errorf("%s: Read accepted it", name)
		}
		path := filepath.Join(dir, "bad.npz")
		var zb bytes.Buffer
		zw := zip.NewWriter(&zb)
		w, _ := zw.Create("x.npy")
		w.Write(file)
		zw.Close()
		if err := os.WriteFile(path, zb.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := ReadNpz(path, device.CPU()); err == nil {
			t.Errorf("%s: ReadNpz accepted it", name)
		}
	}
}

// A v2 header length of ~4 GiB must be rejected before allocating it.
func TestHugeV2HeaderRejected(t *testing.T) {
	file := append([]byte{}, magic...)
	file = append(file, 2, 0)
	file = binary.LittleEndian.AppendUint32(file, 0xFFFFFFF0)
	file = append(file, "{}"...)
	if _, err := ReadHeader(bytes.NewReader(file)); err == nil {
		t.Fatal("4 GiB v2 header length accepted")
	}
}

// Valid inputs are unchanged: a strict-bool header still parses both ways.
func TestValidHeaderStillParses(t *testing.T) {
	for _, fo := range []string{"False", "True"} {
		file := npyV1(`{'descr': '<f4', 'fortran_order': `+fo+`, 'shape': (2,), }`, make([]byte, 8))
		x, err := Read(bytes.NewReader(file), device.CPU())
		if err != nil || x.NumElements() != 2 {
			t.Errorf("fortran_order %s: %v", fo, err)
		}
	}
	file := npyV1(`{'descr': '<f4', 'fortran_order': False, 'shape': (0, 3), }`, nil)
	if x, err := Read(bytes.NewReader(file), device.CPU()); err != nil || x.NumElements() != 0 {
		t.Errorf("empty array: %v", err)
	}
}
