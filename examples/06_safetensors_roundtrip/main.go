// Example 06 — safetensors round-trip with MLP inference on GPU.
//
//  1. Build an MLP on CPU and manually set one Linear's weights to something
//     identifiable (a small identity-ish matrix).
//  2. Save all parameters to an in-memory safetensors buffer.
//  3. Load the buffer directly onto the GPU.
//  4. Build a fresh MLP on GPU, overwrite its parameters from the loaded
//     tensors, and run a forward pass.
//
// This demonstrates the end-to-end loader path: safetensors bytes → Tensors
// on any device → usable inside nn.Module.
package main

import (
	"bytes"
	"fmt"
	"log"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/safetensors"
	"light/tensor"
)

func main() {
	gpu, err := device.CUDA(0)
	if err != nil {
		log.Fatal(err)
	}
	cpu := device.CPU()
	fmt.Printf("device=%s\n", gpu)

	// ---- Build the reference MLP on CPU and set known weights. ----
	l1CPU := mustLinear(4, 4, true, cpu)
	l2CPU := mustLinear(4, 2, true, cpu)

	// Overwrite l1: W = I4 (identity), B = zeros. So l1 is the identity op.
	l1CPU.W = mustTensor([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}, []int{4, 4}, cpu)
	l1CPU.B = mustTensor([]float32{0, 0, 0, 0}, []int{4}, cpu)

	// l2: row-sum-ish — output 0 = sum of input, output 1 = -sum.
	l2CPU.W = mustTensor([]float32{
		1, 1, 1, 1,
		-1, -1, -1, -1,
	}, []int{2, 4}, cpu)
	l2CPU.B = mustTensor([]float32{0, 0}, []int{2}, cpu)

	// ---- Save parameters to safetensors. ----
	paramMap := map[string]*tensor.Tensor{
		"l1.weight": l1CPU.W,
		"l1.bias":   l1CPU.B,
		"l2.weight": l2CPU.W,
		"l2.bias":   l2CPU.B,
	}
	var buf bytes.Buffer
	if err := safetensors.Write(&buf, paramMap, map[string]string{"framework": "light"}); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("wrote safetensors (%d bytes)\n", buf.Len())

	// ---- Load directly onto GPU. ----
	loaded, meta, err := safetensors.Load(bytes.NewReader(buf.Bytes()), gpu)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("loaded %d tensors; framework=%q\n", len(loaded), meta["framework"])
	for k, v := range loaded {
		fmt.Printf("  %s  device=%s  shape=%s\n", k, v.Device(), v.Shape())
		if v.Device().Kind() != device.KindCUDA {
			log.Fatalf("%s should be on CUDA, got %v", k, v.Device())
		}
	}

	// ---- Build a fresh MLP on GPU, swap in loaded weights. ----
	l1GPU := mustLinear(4, 4, true, gpu)
	l2GPU := mustLinear(4, 2, true, gpu)
	l1GPU.W = loaded["l1.weight"]
	l1GPU.B = loaded["l1.bias"]
	l2GPU.W = loaded["l2.weight"]
	l2GPU.B = loaded["l2.bias"]
	net := nn.NewSequential(l1GPU, l2GPU)

	// ---- Run forward on a simple input. ----
	// x = [[1, 2, 3, 4]] — expected: l1(identity) keeps it, l2 row-sum is [10, -10].
	x, _ := tensor.New([]float32{1, 2, 3, 4}, []int{1, 4}, cpu)
	xG, _ := x.To(gpu)

	y, err := net.Forward(xG)
	if err != nil {
		log.Fatal(err)
	}
	yCPU, _ := y.To(cpu)
	got := tensor.MustSlice[float32](yCPU)
	fmt.Printf("forward output: %v\n", got)

	if got[0] != 10 || got[1] != -10 {
		log.Fatalf("expected [10 -10], got %v", got)
	}

	// Also run softmax to make sure the full chain works.
	probs, _ := ops.Softmax(y, -1)
	probsCPU, _ := probs.To(cpu)
	p := tensor.MustSlice[float32](probsCPU)
	fmt.Printf("softmax(output): %v (rows sum to %v)\n", p, p[0]+p[1])

	if y.DType() != dtype.F32 {
		log.Fatalf("wrong dtype: %v", y.DType())
	}
	fmt.Println("safetensors → GPU → nn.Module forward pass works.")
}

func mustLinear(in, out int, bias bool, dev device.Device) *nn.Linear {
	l, err := nn.NewLinear(in, out, bias, dev)
	if err != nil {
		log.Fatal(err)
	}
	return l
}

func mustTensor(data []float32, dims []int, dev device.Device) *tensor.Tensor {
	t, err := tensor.New(data, dims, dev)
	if err != nil {
		log.Fatal(err)
	}
	return t
}
