package safetensors_test

import (
	"bytes"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
	"light/safetensors"
	"light/tensor"
)

// buildFile writes a map of tensors to a safetensors buffer and returns it.
func buildFile(t *testing.T, tensors map[string]*tensor.Tensor, metadata map[string]string) *bytes.Reader {
	t.Helper()
	var buf bytes.Buffer
	if err := safetensors.Write(&buf, tensors, metadata); err != nil {
		t.Fatal(err)
	}
	return bytes.NewReader(buf.Bytes())
}

func TestRoundTripSingleTensor(t *testing.T) {
	w, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device.CPU())
	in := map[string]*tensor.Tensor{"weight": w}

	r := buildFile(t, in, nil)

	loaded, meta, err := safetensors.Load(r, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if meta != nil {
		t.Errorf("expected nil metadata, got %v", meta)
	}
	got, ok := loaded["weight"]
	if !ok {
		t.Fatal("missing 'weight' tensor")
	}
	if got.DType() != dtype.F32 {
		t.Errorf("dtype = %v", got.DType())
	}
	if got.Dim(0) != 2 || got.Dim(1) != 3 {
		t.Errorf("shape = %v", got.Shape())
	}
	if !slices.Equal(tensor.MustSlice[float32](got), []float32{1, 2, 3, 4, 5, 6}) {
		t.Errorf("data mismatch: %v", tensor.MustSlice[float32](got))
	}
}

func TestRoundTripMultiple(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3, 4}, []int{4}, device.CPU())
	b, _ := tensor.New([]int64{10, 20}, []int{2}, device.CPU())
	c, _ := tensor.Ones(dtype.F64, []int{2, 2}, device.CPU())
	in := map[string]*tensor.Tensor{"a": a, "b": b, "c": c}

	r := buildFile(t, in, map[string]string{"framework": "light", "format": "pt"})

	loaded, meta, err := safetensors.Load(r, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if meta["framework"] != "light" {
		t.Errorf("metadata framework = %q", meta["framework"])
	}

	// Verify tensor values
	if !slices.Equal(tensor.MustSlice[float32](loaded["a"]), []float32{1, 2, 3, 4}) {
		t.Error("a mismatch")
	}
	if !slices.Equal(tensor.MustSlice[int64](loaded["b"]), []int64{10, 20}) {
		t.Error("b mismatch")
	}
	cVals := tensor.MustSlice[float64](loaded["c"])
	for _, v := range cVals {
		if v != 1.0 {
			t.Errorf("c should be all ones: %v", cVals)
			break
		}
	}
}

func TestHeaderOnly(t *testing.T) {
	w, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	r := buildFile(t, map[string]*tensor.Tensor{"w": w}, map[string]string{"k": "v"})

	infos, meta, err := safetensors.Header(r)
	if err != nil {
		t.Fatal(err)
	}
	if meta["k"] != "v" {
		t.Errorf("metadata: %v", meta)
	}
	if _, ok := infos["w"]; !ok {
		t.Error("missing w in header")
	}
}

func TestLoadTensorSubset(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	b, _ := tensor.New([]float32{99}, []int{1}, device.CPU())
	r := buildFile(t, map[string]*tensor.Tensor{"a": a, "b": b}, nil)

	loaded, err := safetensors.LoadTensor(r, "b", device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if tensor.MustSlice[float32](loaded)[0] != 99 {
		t.Errorf("b = %v", tensor.MustSlice[float32](loaded))
	}

	// Missing tensor
	if _, err := safetensors.LoadTensor(r, "doesntexist", device.CPU()); err == nil {
		t.Error("expected error for missing tensor")
	}
}

func TestAllDTypes(t *testing.T) {
	cases := map[string]*tensor.Tensor{}
	f32, _ := tensor.New([]float32{1, 2}, []int{2}, device.CPU())
	f64, _ := tensor.New([]float64{1, 2}, []int{2}, device.CPU())
	i32, _ := tensor.New([]int32{1, 2}, []int{2}, device.CPU())
	i64, _ := tensor.New([]int64{1, 2}, []int{2}, device.CPU())
	u8, _ := tensor.New([]uint8{1, 2}, []int{2}, device.CPU())
	i8, _ := tensor.New([]int8{-1, 2}, []int{2}, device.CPU())
	i16, _ := tensor.New([]int16{-1, 2}, []int{2}, device.CPU())
	u16, _ := tensor.New([]uint16{1, 2}, []int{2}, device.CPU())
	u32, _ := tensor.New([]uint32{1, 2}, []int{2}, device.CPU())
	u64, _ := tensor.New([]uint64{1, 2}, []int{2}, device.CPU())
	cases["f32"] = f32
	cases["f64"] = f64
	cases["i32"] = i32
	cases["i64"] = i64
	cases["u8"] = u8
	cases["i8"] = i8
	cases["i16"] = i16
	cases["u16"] = u16
	cases["u32"] = u32
	cases["u64"] = u64

	var buf bytes.Buffer
	if err := safetensors.Write(&buf, cases, nil); err != nil {
		t.Fatal(err)
	}
	loaded, _, err := safetensors.Load(bytes.NewReader(buf.Bytes()), device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded) != len(cases) {
		t.Errorf("got %d tensors, want %d", len(loaded), len(cases))
	}
	// Spot-check dtypes survived.
	for name, orig := range cases {
		got := loaded[name]
		if got.DType() != orig.DType() {
			t.Errorf("%s dtype: got %v, want %v", name, got.DType(), orig.DType())
		}
	}
}

func TestInvalidHeader(t *testing.T) {
	// Too-short file
	r := bytes.NewReader([]byte{1, 2, 3})
	if _, _, err := safetensors.Header(r); err == nil {
		t.Error("expected error on short file")
	}
	// Bad JSON
	badJSON := []byte("garbage")
	var buf bytes.Buffer
	// Write bogus 8-byte length = len(badJSON)
	lenBuf := []byte{byte(len(badJSON)), 0, 0, 0, 0, 0, 0, 0}
	buf.Write(lenBuf)
	buf.Write(badJSON)
	if _, _, err := safetensors.Header(bytes.NewReader(buf.Bytes())); err == nil {
		t.Error("expected JSON parse error")
	}
}

func TestWriteRejectsGPU(t *testing.T) {
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skip("no CUDA")
	}
	w, _ := tensor.New([]float32{1, 2}, []int{2}, device.CPU())
	wG, err := w.To(gpu)
	if err != nil {
		t.Skip("CUDA transfer failed:", err)
	}
	var buf bytes.Buffer
	if err := safetensors.Write(&buf, map[string]*tensor.Tensor{"x": wG}, nil); err == nil {
		t.Error("Write should reject GPU tensors")
	}
}

func TestRoundTripWithCUDATarget(t *testing.T) {
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skip("no CUDA")
	}
	w, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device.CPU())
	r := buildFile(t, map[string]*tensor.Tensor{"w": w}, nil)

	loaded, _, err := safetensors.Load(r, gpu)
	if err != nil {
		t.Fatal(err)
	}
	if loaded["w"].Device().Kind() != device.KindCUDA {
		t.Error("expected tensor on CUDA")
	}
	back, _ := loaded["w"].To(device.CPU())
	if !slices.Equal(tensor.MustSlice[float32](back), []float32{1, 2, 3, 4, 5, 6}) {
		t.Errorf("GPU-loaded values mismatch: %v", tensor.MustSlice[float32](back))
	}
}
