package nn

import (
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
	"light/tensor"
)

func TestVarMapSaveLoad(t *testing.T) {
	w, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device.CPU())
	b, _ := tensor.New([]float32{-1, 0.5}, []int{2}, device.CPU())
	vm := NewVarMap()
	vm.Set("fc.weight", w)
	vm.Set("fc.bias", b)
	if n := vm.Names(); len(n) != 2 || n[0] != "fc.weight" {
		t.Fatalf("names %v", n)
	}
	if p := vm.Prefixed("fc"); p.Len() != 2 || p.Names()[1] != "bias" {
		t.Fatalf("prefixed %v", p.Names())
	}
	for _, ext := range []string{".safetensors", ".npz"} {
		path := filepath.Join(t.TempDir(), "ckpt"+ext)
		if err := vm.Save(path); err != nil {
			t.Fatal(err)
		}
		// Fresh zero tensors receive the file's values in place.
		w2, _ := tensor.Zeros(dtype.F32, []int{2, 3}, device.CPU())
		b2, _ := tensor.Zeros(dtype.F32, []int{2}, device.CPU())
		vm2 := NewVarMap()
		vm2.SetAll(map[string]*tensor.Tensor{"fc.weight": w2, "fc.bias": b2})
		if err := vm2.Load(path, ""); err != nil {
			t.Fatal(err)
		}
		if got := tensor.MustSlice[float32](w2); got[5] != 6 || got[0] != 1 {
			t.Fatalf("%s: weight %v", ext, got)
		}
		if got := tensor.MustSlice[float32](b2); got[1] != 0.5 {
			t.Fatalf("%s: bias %v", ext, got)
		}
		// Shape mismatch is an error, missing names too.
		bad, _ := tensor.Zeros(dtype.F32, []int{3}, device.CPU())
		vm3 := NewVarMap()
		vm3.Set("fc.bias", bad)
		if err := vm3.Load(path, ""); err == nil {
			t.Fatalf("%s: shape mismatch not reported", ext)
		}
		vm3 = NewVarMap()
		vm3.Set("nope", bad)
		if err := vm3.Load(path, ""); err == nil {
			t.Fatalf("%s: missing name not reported", ext)
		}
	}
	// PyTorch checkpoint from the pth fixtures: lin.weight (3,4), lin.bias (3).
	lw, _ := tensor.Zeros(dtype.F32, []int{3, 4}, device.CPU())
	lb, _ := tensor.Zeros(dtype.F32, []int{3}, device.CPU())
	vm4 := NewVarMap()
	vm4.Set("lin.weight", lw)
	vm4.Set("lin.bias", lb)
	if err := vm4.Load("../pth/testdata/checkpoint.pth", "state_dict"); err != nil {
		t.Fatal(err)
	}
	if got := tensor.MustSlice[float32](lw); got[11] != 2.75 || got[1] != 0.25 {
		t.Fatalf("pth weight %v", got)
	}
	if got := tensor.MustSlice[float32](lb); got[1] != -2 {
		t.Fatalf("pth bias %v", got)
	}
}

func TestVarMapLoadCUDA(t *testing.T) {
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	lw, _ := tensor.Zeros(dtype.F32, []int{3, 4}, gpu)
	vm := NewVarMap()
	vm.Set("lin.weight", lw)
	if err := vm.Load("../pth/testdata/state_dict.pth", ""); err != nil {
		t.Fatal(err)
	}
	c, _ := lw.To(device.CPU())
	if got := tensor.MustSlice[float32](c); got[11] != 2.75 {
		t.Fatalf("cuda weight %v", got)
	}
	path := filepath.Join(t.TempDir(), "gpu.safetensors")
	if err := vm.Save(path); err != nil { // GPU tensors are staged to CPU
		t.Fatal(err)
	}
}
