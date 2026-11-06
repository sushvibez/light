package npy

import (
	"bytes"
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// Fixtures written by scripts/verify/gen_npy_pth_fixtures.py (numpy 2.5):
// a = arange(24, f32).reshape(2,3,4) * 0.5.
func wantA() []float32 {
	a := make([]float32, 24)
	for i := range a {
		a[i] = float32(i) * 0.5
	}
	return a
}

func f32s(t *testing.T, x *tensor.Tensor) []float32 {
	t.Helper()
	if x.DType() != dtype.F32 {
		var err error
		if x, err = ops.Cast(x, dtype.F32); err != nil {
			t.Fatal(err)
		}
	}
	if x.Device().Kind() != device.KindCPU {
		var err error
		if x, err = x.To(device.CPU()); err != nil {
			t.Fatal(err)
		}
	}
	return tensor.MustSlice[float32](x)
}

func expect(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d want %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s[%d] = %v want %v", name, i, got[i], want[i])
		}
	}
}

func TestReadNpyCAndFortran(t *testing.T) {
	for _, f := range []string{"a_f32.npy", "a_f32_fortran.npy"} {
		x, err := ReadFile(filepath.Join("testdata", f), device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		if d := x.Shape().Dims(); len(d) != 3 || d[0] != 2 || d[1] != 3 || d[2] != 4 || x.DType() != dtype.F32 {
			t.Fatalf("%s: shape %v dtype %v", f, x.Shape(), x.DType())
		}
		expect(t, f, f32s(t, x), wantA())
	}
	s, err := ReadFile("testdata/scalar_f64.npy", device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if s.Rank() != 0 || s.DType() != dtype.F64 || tensor.MustSlice[float64](s)[0] != 3.25 {
		t.Fatalf("scalar: %v %v", s.Shape(), s.DType())
	}
}

func TestReadNpzAllDTypes(t *testing.T) {
	m, err := ReadNpz("testdata/mixed.npz", device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]struct {
		dt   dtype.DType
		vals []float32
	}{
		"f16": {dtype.F16, wantA()}, "f32": {dtype.F32, wantA()}, "f64": {dtype.F64, wantA()},
		"i8": {dtype.I8, []float32{-3, 4}}, "u8": {dtype.U8, []float32{250, 7}},
		"i16": {dtype.I16, []float32{-300, 400}}, "u16": {dtype.U16, []float32{60000, 7}},
		"i32": {dtype.I32, []float32{-70000, 5}}, "u32": {dtype.U32, []float32{4000000000, 1}},
		"i64": {dtype.I64, []float32{-1099511627776, 9}}, "u64": {dtype.U64, []float32{9223372036854775808, 2}},
		"b": {dtype.U8, []float32{1, 0, 1}},
	}
	if len(m) != len(want) {
		t.Fatalf("got %d members, want %d", len(m), len(want))
	}
	for name, w := range want {
		x := m[name]
		if x == nil {
			t.Fatalf("missing %s", name)
		}
		if x.DType() != w.dt {
			t.Fatalf("%s: dtype %v want %v", name, x.DType(), w.dt)
		}
		expect(t, name, f32s(t, x), w.vals)
	}
}

func TestReadNpzCompressedLazy(t *testing.T) {
	z, err := OpenNpz("testdata/compressed.npz")
	if err != nil {
		t.Fatal(err)
	}
	defer z.Close()
	if n := z.Names(); len(n) != 2 || n[0] != "x" || n[1] != "y" {
		t.Fatalf("names %v", n)
	}
	h, err := z.Header("y")
	if err != nil {
		t.Fatal(err)
	}
	if len(h.Shape) != 3 || h.Shape[0] != 4 || h.Shape[2] != 2 || h.DType != dtype.F32 {
		t.Fatalf("header %+v", h)
	}
	y, err := z.Get("y", device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	// y = a.T (numpy view, saved as Fortran order); compare with a permuted copy.
	a, _ := tensor.New(wantA(), []int{2, 3, 4}, device.CPU())
	at, _ := a.Permute(2, 1, 0)
	at, _ = at.Contiguous()
	expect(t, "y", f32s(t, y), tensor.MustSlice[float32](at))
}

func TestWriteRoundTrip(t *testing.T) {
	a, _ := tensor.New(wantA(), []int{2, 3, 4}, device.CPU())
	i, _ := tensor.New([]int64{-7, 8, 9}, []int{3}, device.CPU())
	var buf bytes.Buffer
	if err := Write(&buf, a); err != nil {
		t.Fatal(err)
	}
	if (buf.Len()-24*4)%64 != 0 { // 128-byte header + data
		t.Fatalf("header not 64-aligned: total %d", buf.Len())
	}
	back, err := Read(&buf, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	expect(t, "npy", f32s(t, back), wantA())

	path := filepath.Join(t.TempDir(), "rt.npz")
	if err := WriteNpz(path, map[string]*tensor.Tensor{"a": a, "i": i}); err != nil {
		t.Fatal(err)
	}
	m, err := ReadNpz(path, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	expect(t, "a", f32s(t, m["a"]), wantA())
	if v := tensor.MustSlice[int64](m["i"]); v[0] != -7 || v[2] != 9 || m["i"].DType() != dtype.I64 {
		t.Fatalf("i = %v", v)
	}
	bf, _ := ops.Cast(a, dtype.BF16)
	if err := Write(&buf, bf); err == nil {
		t.Fatal("BF16 should be rejected (not a NumPy dtype)")
	}
}

func TestReadNpyCUDA(t *testing.T) {
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	m, err := ReadNpz("testdata/mixed.npz", dev)
	if err != nil {
		t.Fatal(err)
	}
	expect(t, "f16 cuda", f32s(t, m["f16"]), wantA())
	expect(t, "i32 cuda", f32s(t, m["i32"]), []float32{-70000, 5})
}
