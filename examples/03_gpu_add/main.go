// Example 03 — first real GPU op.
//
// Allocates two F32 tensors on the GPU, runs Add and SiLU, copies result
// back to CPU, prints it. Asserts the device is CUDA at startup — failure
// to find a CUDA device is a hard error (no silent CPU fallback).
//
// Build & run:
//
//	go generate ./kernels/cuda/...     # produce PTX (once; needs nvcc)
//	go run            ./examples/03_gpu_add
package main

import (
	"fmt"
	"log"
	"time"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

func main() {
	gpu, err := device.CUDA(0)
	if err != nil {
		log.Fatalf("CUDA(0) failed: %v", err)
	}
	if gpu.Kind() != device.KindCUDA {
		log.Fatalf("expected CUDA, got %v", gpu)
	}
	fmt.Printf("device=%s\n", gpu)

	// Build two large tensors on CPU and upload them.
	const n = 1 << 20 // 1M elements
	hostA := make([]float32, n)
	hostB := make([]float32, n)
	for i := range hostA {
		hostA[i] = float32(i) * 0.001
		hostB[i] = float32(n-i) * 0.0005
	}

	cpu := device.CPU()
	aCPU, _ := tensor.New(hostA, []int{n}, cpu)
	bCPU, _ := tensor.New(hostB, []int{n}, cpu)

	fmt.Printf("uploading tensors... ")
	start := time.Now()
	aGPU, err := aCPU.To(gpu)
	if err != nil {
		log.Fatal(err)
	}
	bGPU, err := bCPU.To(gpu)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("done (%s)\n", time.Since(start))

	fmt.Printf("add on GPU... ")
	start = time.Now()
	sum, err := ops.Add(aGPU, bGPU)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("done (%s)\n", time.Since(start))
	if sum.Device().Kind() != device.KindCUDA {
		log.Fatalf("expected sum on CUDA, got %v", sum.Device())
	}

	fmt.Printf("silu(sum) on GPU... ")
	start = time.Now()
	activated, err := ops.SiLU(sum)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("done (%s)\n", time.Since(start))

	fmt.Printf("downloading to CPU... ")
	start = time.Now()
	resultCPU, err := activated.To(cpu)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("done (%s)\n", time.Since(start))

	result := tensor.MustSlice[float32](resultCPU)

	// Spot-check: first few and last few values against the expected formula:
	// silu(a[i] + b[i]) = (a+b) / (1 + exp(-(a+b)))
	fmt.Printf("first 5: %v\n", result[:5])
	fmt.Printf(" last 5: %v\n", result[len(result)-5:])

	// Dtype sanity.
	if activated.DType() != dtype.F32 {
		log.Fatalf("expected F32, got %v", activated.DType())
	}
	fmt.Printf("dtype=%v shape=%v total elements=%d\n", activated.DType(), activated.Shape(), activated.NumElements())
}
