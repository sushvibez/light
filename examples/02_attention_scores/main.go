// Example 02 — mini attention scores on CPU.
//
// Computes a one-head attention score tensor from a tiny Q (2×4) and K (2×4),
// then applies softmax across the key dimension. Exercises reductions,
// softmax, matmul, and broadcast arithmetic together.
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
	dev := device.CPU()
	if dev.Kind() != device.KindCPU {
		log.Fatalf("expected CPU, got %v", dev)
	}
	fmt.Printf("device=%s\n", dev)

	// Q: (2 queries, 4-dim embedding)
	q, _ := tensor.New([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
	}, []int{2, 4}, dev)
	// K: (3 keys, 4-dim embedding)
	k, _ := tensor.New([]float32{
		1, 0, 0, 0,
		1, 1, 0, 0,
		0, 0, 1, 0,
	}, []int{3, 4}, dev)

	// scores = Q @ K^T — shape (2, 3)
	kT, _ := k.Transpose(0, 1)
	scores, err := ops.MatMul(q, kT)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("scores shape=%s data=%v\n", scores.Shape(), tensor.MustSlice[float32](scores))

	// Scale by sqrt(d_k) = sqrt(4) = 2 — divide by the scalar 2.
	scale, _ := tensor.Full[float32](2.0, []int{1}, dev)
	scaled, err := ops.Div(scores, scale)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("scaled shape=%s data=%v\n", scaled.Shape(), tensor.MustSlice[float32](scaled))

	// Softmax along the last axis (per-query normalization).
	attn, err := ops.Softmax(scaled, -1)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("attention shape=%s data=%v\n", attn.Shape(), tensor.MustSlice[float32](attn))

	// Sanity: each row should sum to 1.
	rowSums, _ := ops.Sum(attn, ops.ReduceOptions{Axes: []int{-1}})
	fmt.Printf("row sums (each must ≈ 1): %v\n", tensor.MustSlice[float32](rowSums))

	// Max attention per query and which key it prefers is not trivially given by
	// Max() alone (we'd need Argmax — Phase 3). For now, report the max value.
	maxAttn, _ := ops.Max(attn, ops.ReduceOptions{Axes: []int{-1}})
	fmt.Printf("max attention per query: %v\n", tensor.MustSlice[float32](maxAttn))

	// Dtype check.
	if attn.DType() != dtype.F32 {
		log.Fatalf("expected F32, got %v", attn.DType())
	}
}
