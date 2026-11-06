package pth

import (
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// Fixtures from scripts/verify/gen_npy_pth_fixtures.py (torch 2.11):
// w = arange(12, f32).reshape(3,4) * 0.25.
func wantW() []float32 {
	w := make([]float32, 12)
	for i := range w {
		w[i] = float32(i) * 0.25
	}
	return w
}

func f32s(t *testing.T, x *tensor.Tensor) []float32 {
	t.Helper()
	var err error
	if x.DType() != dtype.F32 {
		if x, err = ops.Cast(x, dtype.F32); err != nil {
			t.Fatal(err)
		}
	}
	if x.Device().Kind() != device.KindCPU {
		if x, err = x.To(device.CPU()); err != nil {
			t.Fatal(err)
		}
	}
	return tensor.MustSlice[float32](x)
}

func expect(t *testing.T, name string, x *tensor.Tensor, dt dtype.DType, dims []int, want []float32) {
	t.Helper()
	if x == nil {
		t.Fatalf("%s: missing", name)
	}
	if x.DType() != dt {
		t.Fatalf("%s: dtype %v want %v", name, x.DType(), dt)
	}
	d := x.Shape().Dims()
	if len(d) != len(dims) {
		t.Fatalf("%s: shape %v want %v", name, d, dims)
	}
	for i := range d {
		if d[i] != dims[i] {
			t.Fatalf("%s: shape %v want %v", name, d, dims)
		}
	}
	got := f32s(t, x)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s[%d] = %v want %v (all %v)", name, i, got[i], want[i], got)
		}
	}
}

func checkStateDict(t *testing.T, m map[string]*tensor.Tensor) {
	t.Helper()
	if len(m) != 14 {
		t.Fatalf("got %d tensors: %v", len(m), keys(m))
	}
	w := wantW()
	expect(t, "lin.weight", m["lin.weight"], dtype.F32, []int{3, 4}, w)
	expect(t, "lin.bias", m["lin.bias"], dtype.F32, []int{3}, []float32{1, -2, 3})
	// w.t(): strides (1,4) over the same storage → gathered row-major.
	expect(t, "w_t", m["w_t"], dtype.F32, []int{4, 3}, []float32{0, 1, 2, 0.25, 1.25, 2.25, 0.5, 1.5, 2.5, 0.75, 1.75, 2.75})
	// w[1:, 1:3]: offset 5, strides (4,1).
	expect(t, "w_slice", m["w_slice"], dtype.F32, []int{2, 2}, []float32{1.25, 1.5, 2.25, 2.5})
	expect(t, "bf16", m["bf16"], dtype.BF16, []int{3, 4}, w)
	expect(t, "f16", m["f16"], dtype.F16, []int{3, 4}, w)
	expect(t, "f64", m["f64"], dtype.F64, []int{3, 4}, w)
	expect(t, "i64", m["i64"], dtype.I64, []int{2, 2}, []float32{-5, 6, 7, 8})
	expect(t, "i32", m["i32"], dtype.I32, []int{2}, []float32{9, -10})
	expect(t, "i16", m["i16"], dtype.I16, []int{2}, []float32{11, -12})
	expect(t, "i8", m["i8"], dtype.I8, []int{2}, []float32{13, -14})
	expect(t, "u8", m["u8"], dtype.U8, []int{2}, []float32{255, 16})
	expect(t, "bool", m["bool"], dtype.U8, []int{2}, []float32{1, 0})
	expect(t, "scalar", m["scalar"], dtype.F32, []int{}, []float32{2.5})
}

func keys(m map[string]*tensor.Tensor) []string {
	var ks []string
	for k := range m {
		ks = append(ks, k)
	}
	return ks
}

func TestStateDictProtocols(t *testing.T) {
	for _, f := range []string{"testdata/state_dict.pth", "testdata/state_dict_p4.pth"} {
		m, err := Load(f, "", device.CPU())
		if err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		checkStateDict(t, m)
	}
}

func TestCheckpointKeyAndLazy(t *testing.T) {
	// Without a key the non-tensor entries (epoch, lr) are skipped and the
	// nested state dict is flattened with its prefix.
	m, err := Load("testdata/checkpoint.pth", "", device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if len(m) != 14 || m["state_dict.lin.weight"] == nil {
		t.Fatalf("flattened: %v", keys(m))
	}
	f, err := Open("testdata/checkpoint.pth", "state_dict")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if len(f.Names()) != 14 {
		t.Fatalf("names %v", f.Names())
	}
	info := f.Infos()["w_t"]
	if info.Offset != 0 || info.Strides[0] != 1 || info.Strides[1] != 4 || info.storageSize != 12 {
		t.Fatalf("w_t info %+v", info)
	}
	x, err := f.Get("w_slice", device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	expect(t, "w_slice", x, dtype.F32, []int{2, 2}, []float32{1.25, 1.5, 2.25, 2.5})
	if _, err := Open("testdata/checkpoint.pth", "nope"); err == nil {
		t.Fatal("missing key should error")
	}
}

func TestPickledModule(t *testing.T) {
	m, err := Load("testdata/module.pth", "", device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if len(m) != 2 {
		t.Fatalf("got %v", keys(m))
	}
	expect(t, "weight", m["weight"], dtype.F32, []int{3, 4}, wantW())
	expect(t, "bias", m["bias"], dtype.F32, []int{3}, []float32{1, -2, 3})
}

func TestLoadCUDA(t *testing.T) {
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	m, err := Load("testdata/state_dict.pth", "", dev)
	if err != nil {
		t.Fatal(err)
	}
	checkStateDict(t, m)
}
