// Example 16 — train a 2-layer transformer on a synthetic sequence task.
//
// Task: given an input sequence of T=6 tokens from a vocab of V=12, predict
// one of K=4 classes. The label is the token with maximum id in the
// sequence (a task that benefits from attention being able to look across
// positions). The class is then (max_token_id * K / V) — a simple mapping
// from "biggest token" to class.
//
// Architecture:
//
//	tokens [T]         (I64)
//	 ↓   Embedding
//	hidden [T, D]
//	 ↓   TransformerBlock ×2   (LayerNorm → MHA → residual → LayerNorm → FFN → residual)
//	hidden [T, D]
//	 ↓   AxisMean(axis=0)
//	pooled [D]
//	 ↓   Reshape to [1, D]
//	 ↓   Linear (D → K)
//	logits [1, K]
//	 ↓   CrossEntropyLoss (vs target [1])
//	loss
//
// This is the first training loop that exercises every transformer primitive
// end-to-end: Embedding, LayerNorm, MultiHeadAttention (2 heads), FFN (Linear
// + SiLU + Linear), residual adds, batched MatMul, Permute, Softmax, and
// CrossEntropyLoss — every one of them autograd-routed so gradients flow back
// through the full stack.
package main

import (
	"fmt"
	"log"
	"math/rand/v2"
	"runtime"

	"light/autograd"
	"light/device"
	"light/nn"
	"light/tensor"
	"light/training"
)

const (
	vocab   = 12
	seq     = 6
	embDim  = 16
	classes = 4
	steps   = 1500
	batches = 1 // train one sequence per step (small model, no batched MHA shape yet)
)

// Generate a random sequence and its label (max token id's bucket).
func makeExample(rng *rand.Rand) ([]int64, int64) {
	tokens := make([]int64, seq)
	maxID := int64(0)
	for i := range tokens {
		tokens[i] = int64(rng.IntN(vocab))
		if tokens[i] > maxID {
			maxID = tokens[i]
		}
	}
	return tokens, maxID * int64(classes) / int64(vocab)
}

func main() {
	runtime.LockOSThread()
	dev := device.Auto()
	fmt.Printf("device: %v\n", dev)
	rng := rand.New(rand.NewPCG(7, 42))

	emb, _ := nn.NewEmbedding(vocab, embDim, dev)
	block1, _ := nn.NewTransformerBlock(embDim, false, dev)
	block2, _ := nn.NewTransformerBlock(embDim, false, dev)
	finalNorm, _ := nn.NewLayerNorm(embDim, 1e-5, dev)
	head, _ := nn.NewLinear(embDim, classes, true, dev)

	var params []*tensor.Tensor
	for _, m := range []nn.Module{emb, block1, block2, finalNorm, head} {
		params = append(params, m.Parameters()...)
	}
	total := 0
	for _, p := range params {
		total += p.NumElements()
	}
	fmt.Printf("tiny transformer: %d parameters\n", total)

	opt, err := training.NewAdamW(params, training.AdamWOptions{LR: 0.005})
	if err != nil {
		log.Fatal(err)
	}

	forward := func(tokens *tensor.Tensor) (*tensor.Tensor, error) {
		h, err := emb.Forward(tokens)
		if err != nil {
			return nil, err
		}
		h, err = block1.Forward(h)
		if err != nil {
			return nil, err
		}
		h, err = block2.Forward(h)
		if err != nil {
			return nil, err
		}
		h, err = finalNorm.Forward(h)
		if err != nil {
			return nil, err
		}
		// Pool over sequence axis.
		pooled, err := autograd.AxisMean(h, []int{0}, false) // [D]
		if err != nil {
			return nil, err
		}
		// Reshape [D] → [1, D] for the linear classification head.
		pooled2D, err := autograd.Reshape(pooled, 1, embDim)
		if err != nil {
			return nil, err
		}
		return head.Forward(pooled2D) // [1, K]
	}

	// Training loop
	correct := 0
	for step := range steps {
		tokens, label := makeExample(rng)
		tokT, _ := tensor.New(tokens, []int{seq}, dev)
		lblT, _ := tensor.New([]int64{label}, []int{1}, dev)

		opt.ZeroGrad()
		logits, err := forward(tokT)
		if err != nil {
			log.Fatal(err)
		}
		loss, err := autograd.CrossEntropyLoss(logits, lblT)
		if err != nil {
			log.Fatal(err)
		}
		if err := autograd.Backward(loss, nil); err != nil {
			log.Fatal(err)
		}
		if err := opt.Step(); err != nil {
			log.Fatal(err)
		}

		// Track rolling accuracy. Copy to CPU so MustSlice works regardless of device.
		logitsCPU, _ := logits.To(device.CPU())
		flat := tensor.MustSlice[float32](logitsCPU)
		best := 0
		for c := 1; c < classes; c++ {
			if flat[c] > flat[best] {
				best = c
			}
		}
		if int64(best) == label {
			correct++
		}

		if step%100 == 0 || step == steps-1 {
			window := step + 1
			if window > 100 {
				window = 100
			}
			acc := float32(correct) / float32(window) * 100
			if step%100 == 0 && step > 0 {
				correct = 0 // reset the 100-step window
			}
			lossCPU, _ := loss.To(device.CPU())
			fmt.Printf("step %4d  loss = %.4f  rolling-acc = %.1f%%\n",
				step, tensor.MustSlice[float32](lossCPU)[0], acc)
		}
	}

	// Final eval on 200 fresh samples.
	fmt.Println("\n--- evaluation ---")
	evalCorrect := 0
	const evalN = 200
	for i := 0; i < evalN; i++ {
		tokens, label := makeExample(rng)
		tokT, _ := tensor.New(tokens, []int{seq}, dev)
		logits, _ := forward(tokT)
		logitsCPU, _ := logits.To(device.CPU())
		flat := tensor.MustSlice[float32](logitsCPU)
		best := 0
		for c := 1; c < classes; c++ {
			if flat[c] > flat[best] {
				best = c
			}
		}
		if int64(best) == label {
			evalCorrect++
		}
	}
	fmt.Printf("held-out accuracy: %.1f%% (%d/%d)\n",
		float32(evalCorrect)/float32(evalN)*100, evalCorrect, evalN)
}
