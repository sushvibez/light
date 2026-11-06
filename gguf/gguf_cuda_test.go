//go:build !nocuda

package gguf_test

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/gguf"
	_ "light/kernels/cuda" // register CUDA dequant hook
	"light/quant"
	"light/tensor"
)

func TestGGUFQuantTensorOnCUDA(t *testing.T) {
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}

	scale := float32(0.25)
	var vals [32]int8
	for i := range vals {
		vals[i] = int8((i % 16) - 8)
	}
	block := buildQ4_0BlockRaw(scale, vals)

	w := gguf.NewWriter()
	if err := w.AddTensor("w", []int{32}, 2 /*Q4_0*/, block); err != nil {
		t.Fatal(err)
	}
	data, err := w.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	f, err := gguf.Open(&byteReaderAt{b: data})
	if err != nil {
		t.Fatal(err)
	}

	// Load directly onto the GPU.
	qt, err := f.Tensor("w", gpu)
	if err != nil {
		t.Fatal(err)
	}
	if qt.Device().Kind() != device.KindCUDA {
		t.Fatalf("expected CUDA tensor, got %v", qt.Device())
	}
	if qt.DType() != dtype.Q4_0 {
		t.Fatalf("dtype = %v", qt.DType())
	}

	// Dequant on the GPU, round-trip back to CPU, compare.
	dq, err := quant.DequantQ4_0(qt)
	if err != nil {
		t.Fatal(err)
	}
	if dq.Device().Kind() != device.KindCUDA {
		t.Fatalf("expected GPU dequant output, got %v", dq.Device())
	}
	cpu, err := dq.To(device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](cpu)
	for i, v := range vals {
		want := scale * float32(v)
		if math.Abs(float64(got[i]-want)) > 1e-5 {
			t.Errorf("[%d] got %v want %v", i, got[i], want)
		}
	}
}
