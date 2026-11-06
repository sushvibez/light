// Example 01 — tensor basics (CPU).
//
// Creates a few tensors, reshapes and transposes them, adds with broadcasting,
// and does a small matrix multiplication. Asserts the device is CPU (Phase 1
// has no CUDA yet; this assertion exists to catch accidental regressions).
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

	// Arange → reshape.
	a, err := tensor.Arange(0, 6, 1, dtype.F32, dev)
	if err != nil {
		log.Fatal(err)
	}
	a, err = a.Reshape(2, 3)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("a shape=%s data=%v\n", a.Shape(), tensor.MustSlice[float32](mustContig(a)))

	// Ones + broadcast add.
	b, err := tensor.Ones(dtype.F32, []int{3}, dev)
	if err != nil {
		log.Fatal(err)
	}
	c, err := ops.Add(a, b)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("a+b shape=%s data=%v\n", c.Shape(), tensor.MustSlice[float32](c))

	// MatMul with a transpose.
	at, err := a.Transpose(0, 1) // (3,2)
	if err != nil {
		log.Fatal(err)
	}
	p, err := ops.MatMul(a, at)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("a @ a^T shape=%s data=%v\n", p.Shape(), tensor.MustSlice[float32](p))
}

func mustContig(t *tensor.Tensor) *tensor.Tensor {
	c, err := t.Contiguous()
	if err != nil {
		log.Fatal(err)
	}
	return c
}
