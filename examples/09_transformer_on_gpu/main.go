// Example 09 — full transformer forward pass on GPU.
//
// Assembles a 2-layer causal transformer:
//
//	tokens → Embedding → [Block1 → Block2] → LayerNorm → lm_head → logits
//
// Each block is LayerNorm → SelfAttention → residual → LayerNorm →
// FFN(Linear→SiLU→Linear) → residual. Runs entirely on the GPU.
//
// This is the smallest plausibly-real GPT-style forward pass our stack has
// shipped. Loading actual trained weights (tiny-gpt2 etc.) lands next phase
// once we add a `models/gpt2.go` architecture file that maps safetensors
// parameter names to our modules.
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
		vocabSize = 200
		hidden    = 32
		numLayers = 2
		seqLen    = 8
	)

	// Token embedding.
	embed, err := nn.NewEmbedding(vocabSize, hidden, gpu)
	if err != nil {
		log.Fatal(err)
	}

	// Transformer blocks.
	blocks := make([]*nn.TransformerBlock, numLayers)
	for i := range numLayers {
		b, err := nn.NewTransformerBlock(hidden, true, gpu) // causal
		if err != nil {
			log.Fatal(err)
		}
		blocks[i] = b
	}

	// Final LayerNorm + LM head.
	finalNorm, err := nn.NewLayerNorm(hidden, 1e-5, gpu)
	if err != nil {
		log.Fatal(err)
	}
	lmHead, err := nn.NewLinear(hidden, vocabSize, false, gpu)
	if err != nil {
		log.Fatal(err)
	}

	// Count parameters.
	var totalParams int
	countParams := func(ps []*tensor.Tensor) {
		for _, p := range ps {
			totalParams += p.NumElements()
		}
	}
	countParams(embed.Parameters())
	for _, b := range blocks {
		countParams(b.Parameters())
	}
	countParams(finalNorm.Parameters())
	countParams(lmHead.Parameters())
	fmt.Printf("model: vocab=%d hidden=%d layers=%d → %d parameters\n",
		vocabSize, hidden, numLayers, totalParams)

	// Construct fake tokens.
	tokens := make([]int64, seqLen)
	for i := range seqLen {
		tokens[i] = int64(i*7+3) % vocabSize
	}
	tokCPU, _ := tensor.New(tokens, []int{seqLen}, cpu)
	tokG, err := tokCPU.To(gpu)
	if err != nil {
		log.Fatal(err)
	}

	// --- Forward pass ---
	h, err := embed.Forward(tokG)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("after embed: shape=%s device=%s\n", h.Shape(), h.Device())

	for i, b := range blocks {
		h, err = b.Forward(h)
		if err != nil {
			log.Fatalf("block %d: %v", i, err)
		}
	}
	fmt.Printf("after blocks: shape=%s\n", h.Shape())

	h, err = finalNorm.Forward(h)
	if err != nil {
		log.Fatal(err)
	}
	logits, err := lmHead.Forward(h)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("logits: shape=%s device=%s\n", logits.Shape(), logits.Device())

	// Turn into probabilities and spot-check.
	probs, err := ops.Softmax(logits, -1)
	if err != nil {
		log.Fatal(err)
	}
	probsCPU, _ := probs.To(cpu)
	vals := tensor.MustSlice[float32](probsCPU)

	// Check every row (token position) sums to 1.
	for i := range seqLen {
		sum := float32(0)
		for j := range vocabSize {
			sum += vals[i*vocabSize+j]
		}
		if abs32(sum-1) > 1e-3 {
			log.Fatalf("row %d sums to %v, expected 1.0", i, sum)
		}
	}
	fmt.Printf("all %d token positions produce valid probability distributions ✓\n", seqLen)

	// Pick argmax per position (greedy next-token preview).
	fmt.Print("greedy argmax per position: [")
	for i := range seqLen {
		best := 0
		bestP := vals[i*vocabSize]
		for j := 1; j < vocabSize; j++ {
			if vals[i*vocabSize+j] > bestP {
				bestP = vals[i*vocabSize+j]
				best = j
			}
		}
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Printf("%d", best)
	}
	fmt.Println("]")
	fmt.Printf("full %d-layer transformer forward on GPU: success.\n", numLayers)
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
