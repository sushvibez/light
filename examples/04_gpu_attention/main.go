// Example 04 — attention scores on GPU (natural form).
//
// Computes softmax(Q · Kᵀ / √d_k) fully on the GPU using the same syntax as
// the CPU version (example 02): Transpose, scalar Div, Softmax. The MatMul
// auto-contiguifies the transposed K via the CUDA strided-copy kernel, and
// Div broadcasts the scalar across all elements via the strided binary kernel.
package main

import (
	"fmt"
	"log"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

func main() {
	gpu, err := device.CUDA(0)
	if err != nil {
		log.Fatal(err)
	}
	if gpu.Kind() != device.KindCUDA {
		log.Fatalf("expected CUDA, got %v", gpu)
	}
	fmt.Printf("device=%s\n", gpu)
	cpu := device.CPU()

	// Q (2 queries, 4 dims) and K (3 keys, 4 dims) — exactly as in CPU example 02.
	q, _ := tensor.New([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
	}, []int{2, 4}, cpu)
	k, _ := tensor.New([]float32{
		1, 0, 0, 0,
		1, 1, 0, 0,
		0, 0, 1, 0,
	}, []int{3, 4}, cpu)

	qG, _ := q.To(gpu)
	kG, _ := k.To(gpu)

	// Natural form: Q @ Kᵀ. MatMul auto-Contiguous-ifies the transposed view.
	kGT, err := kG.Transpose(0, 1)
	if err != nil {
		log.Fatal(err)
	}
	scores, err := ops.MatMul(qG, kGT)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("scores device=%s shape=%s\n", scores.Device(), scores.Shape())

	// Scale by √d_k = 2. Broadcast-div via a scalar tensor (CUDA strided-binary).
	scale, _ := tensor.New([]float32{2.0}, []int{1}, cpu)
	scaleG, _ := scale.To(gpu)
	scaled, err := ops.Div(scores, scaleG)
	if err != nil {
		log.Fatal(err)
	}

	// Softmax along the last axis — CUDA fused kernel.
	attn, err := ops.Softmax(scaled, -1)
	if err != nil {
		log.Fatal(err)
	}
	if attn.Device().Kind() != device.KindCUDA {
		log.Fatalf("softmax output on wrong device: %v", attn.Device())
	}

	attnCPU, err := attn.To(cpu)
	if err != nil {
		log.Fatal(err)
	}
	vals := tensor.MustSlice[float32](attnCPU)
	fmt.Printf("attention shape=%s data=%v\n", attn.Shape(), vals)

	for row := range 2 {
		sum := vals[row*3] + vals[row*3+1] + vals[row*3+2]
		fmt.Printf("row %d sum=%.6f\n", row, sum)
	}

	if attn.DType() != dtype.F32 {
		log.Fatalf("wrong dtype: %v", attn.DType())
	}
}
