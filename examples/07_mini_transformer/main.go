// Example 07 — mini transformer FFN block on GPU.
//
// Assembles the real transformer forward-path primitives we now support:
//
//	token ids → Embedding → RMSNorm → Linear(gate) → SiLU → Linear(down)
//
// This is the shape of a Llama-style FFN (sans up-projection and bias, for
// brevity). All layers run on CUDA; no intermediate CPU hops.
package main

import (
	"fmt"
	"log"

	"light/device"
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
	cpu := device.CPU()
	fmt.Printf("device=%s\n", gpu)

	const (
		vocabSize    = 100
		hidden       = 32
		intermediate = 64
	)

	// 1. Token embedding.
	embed, err := nn.NewEmbedding(vocabSize, hidden, gpu)
	if err != nil {
		log.Fatal(err)
	}

	// 2. Normalization (Llama-style RMSNorm).
	norm, err := nn.NewRMSNorm(hidden, 1e-5, gpu)
	if err != nil {
		log.Fatal(err)
	}

	// 3. FFN: up-project, SiLU, down-project.
	gate, err := nn.NewLinear(hidden, intermediate, false, gpu)
	if err != nil {
		log.Fatal(err)
	}
	down, err := nn.NewLinear(intermediate, hidden, false, gpu)
	if err != nil {
		log.Fatal(err)
	}

	// 4. Assemble as a Sequential (we don't need the raw residual, just a
	//    forward-through-the-stack for this example).
	block := nn.NewSequential(norm, gate, nn.SiLU{}, down)

	// Input token IDs.
	tokens, _ := tensor.New([]int64{5, 2, 8, 99, 42}, []int{5}, cpu)
	tokensG, err := tokens.To(gpu)
	if err != nil {
		log.Fatal(err)
	}

	// 5. Forward.
	embedded, err := embed.Forward(tokensG)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("embedded shape=%s device=%s\n", embedded.Shape(), embedded.Device())
	if embedded.Device().Kind() != device.KindCUDA {
		log.Fatalf("embedding output not on GPU")
	}

	y, err := block.Forward(embedded)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("block output shape=%s device=%s\n", y.Shape(), y.Device())

	// Residual add: y = y + embedded (exercises broadcast-less same-shape add).
	withResidual, err := ops.Add(y, embedded)
	if err != nil {
		log.Fatal(err)
	}

	// Bring back to CPU for reporting stats.
	res, _ := withResidual.To(cpu)
	vals := tensor.MustSlice[float32](res)
	var minV, maxV, sum float32
	minV = vals[0]
	maxV = vals[0]
	for _, v := range vals {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
		sum += v
	}
	mean := sum / float32(len(vals))
	fmt.Printf("final  shape=%s  stats: min=%.3f  max=%.3f  mean=%.3f\n",
		withResidual.Shape(), minV, maxV, mean)

	// Parameter count summary — first time we can print something meaningful.
	totalParams := 0
	for _, p := range append(append(embed.Parameters(), norm.Parameters()...), append(gate.Parameters(), down.Parameters()...)...) {
		totalParams += p.NumElements()
	}
	fmt.Printf("total params: %d\n", totalParams)
	fmt.Printf("mini transformer FFN block forward on GPU — success.\n")
}
