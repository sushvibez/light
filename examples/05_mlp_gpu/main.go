// Example 05 — Small MLP on GPU.
//
// Builds a 3-layer MLP (Linear → ReLU → Linear → SiLU → Linear → Softmax)
// entirely on CUDA and runs a forward pass on random inputs. Demonstrates
// that `nn.Module` composition produces identical structure on GPU as CPU.
//
// This is the first example that exercises real "neural network" semantics:
// parameter-owning modules, multi-layer composition, and activation functions.
package main

import (
	"fmt"
	"log"

	"light/device"
	"light/dtype"
	"light/nn"
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

	// Build the network on GPU directly: Linear layers initialize on GPU.
	net := nn.NewSequential(
		mustLinear(8, 32, true, gpu),
		nn.ReLU{},
		mustLinear(32, 32, true, gpu),
		nn.SiLU{},
		mustLinear(32, 4, true, gpu),
	)
	params := net.Parameters()
	totalParams := 0
	for _, p := range params {
		totalParams += p.NumElements()
	}
	fmt.Printf("net: %d modules, %d trainable tensors, %d parameters\n",
		net.Len(), len(params), totalParams)

	// Input batch of 16 samples, 8 features each.
	const batch, inFeat = 16, 8
	x, err := tensor.Rand(dtype.F32, []int{batch, inFeat}, device.CPU())
	if err != nil {
		log.Fatal(err)
	}
	xG, err := x.To(gpu)
	if err != nil {
		log.Fatal(err)
	}

	// Forward pass.
	logits, err := net.Forward(xG)
	if err != nil {
		log.Fatal(err)
	}
	if logits.Device().Kind() != device.KindCUDA {
		log.Fatalf("logits not on GPU: %v", logits.Device())
	}
	if logits.Dim(0) != batch || logits.Dim(1) != 4 {
		log.Fatalf("logits shape wrong: %v (want (%d,4))", logits.Shape(), batch)
	}

	// Softmax to get class probabilities.
	probs, err := ops.Softmax(logits, -1)
	if err != nil {
		log.Fatal(err)
	}

	// Bring to CPU for printing.
	probsCPU, _ := probs.To(device.CPU())
	vals := tensor.MustSlice[float32](probsCPU)

	fmt.Printf("logits shape=%s\n", logits.Shape())
	fmt.Printf("probs  shape=%s\n", probs.Shape())
	fmt.Println("probs[sample 0]:", vals[:4])
	fmt.Println("probs[sample 1]:", vals[4:8])

	// Sanity check: every row must sum to 1.
	for i := range batch {
		sum := vals[i*4] + vals[i*4+1] + vals[i*4+2] + vals[i*4+3]
		if abs32(sum-1) > 1e-4 {
			log.Fatalf("row %d sum=%v, expected 1.0", i, sum)
		}
	}
	fmt.Printf("all %d rows sum to 1.0 ✓\n", batch)
}

func mustLinear(inF, outF int, bias bool, dev device.Device) *nn.Linear {
	l, err := nn.NewLinear(inF, outF, bias, dev)
	if err != nil {
		log.Fatal(err)
	}
	return l
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
