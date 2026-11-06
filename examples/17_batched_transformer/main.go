// Example 17 — train a batched 2-layer transformer with CrossEntropyLoss.
//
// Unlike example 16 (one sequence per step), this trains in true mini-batches
// of 16 sequences at a time, shape (B, T). Matches the PyTorch idiom:
//
//	tokens [B, T]      (I64)
//	 ↓  Embedding      → [B, T, D]
//	 ↓  TransformerBlock (autograd-routed LN + MHA + FFN + residuals)
//	 ↓  TransformerBlock
//	 ↓  final LayerNorm → [B, T, D]
//	 ↓  AxisMean(axis=1 over T) → [B, D]
//	 ↓  Linear (D → K)  → [B, K]
//	 ↓  CrossEntropyLoss (target [B])
//	loss
//
// Task: label = maximum token id in the sequence, bucketed into K classes.
// Attention helps because the max is a position-invariant property.
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

// dev is set in main() so makeBatch can build batch tensors on the active
// device. Examples-only convenience; production code would thread the
// device through explicit parameters instead.
var dev device.Device

const (
	vocab   = 12
	seq     = 8
	embDim  = 24
	classes = 4
	batch   = 16
	steps   = 800
)

func makeBatch(rng *rand.Rand) (*tensor.Tensor, *tensor.Tensor) {
	toks := make([]int64, batch*seq)
	lbls := make([]int64, batch)
	for b := range batch {
		maxID := int64(0)
		for t := range seq {
			toks[b*seq+t] = int64(rng.IntN(vocab))
			if toks[b*seq+t] > maxID {
				maxID = toks[b*seq+t]
			}
		}
		lbls[b] = maxID * int64(classes) / int64(vocab)
	}
	x, _ := tensor.New(toks, []int{batch, seq}, dev)
	y, _ := tensor.New(lbls, []int{batch}, dev)
	return x, y
}

func main() {
	runtime.LockOSThread()
	dev = device.Auto()
	fmt.Printf("device: %v\n", dev)
	rng := rand.New(rand.NewPCG(3, 14))

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
	fmt.Printf("batched transformer: %d parameters\n", total)

	opt, err := training.NewAdamW(params, training.AdamWOptions{LR: 0.005})
	if err != nil {
		log.Fatal(err)
	}

	forward := func(tok *tensor.Tensor) (*tensor.Tensor, error) {
		h, err := emb.Forward(tok) // (B, T, D)
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
		pooled, err := autograd.AxisMean(h, []int{1}, false) // (B, D)
		if err != nil {
			return nil, err
		}
		return head.Forward(pooled) // (B, K)
	}

	for step := range steps {
		tok, lbl := makeBatch(rng)
		opt.ZeroGrad()
		logits, err := forward(tok)
		if err != nil {
			log.Fatal(err)
		}
		loss, err := autograd.CrossEntropyLoss(logits, lbl)
		if err != nil {
			log.Fatal(err)
		}
		if err := autograd.Backward(loss, nil); err != nil {
			log.Fatal(err)
		}
		if err := opt.Step(); err != nil {
			log.Fatal(err)
		}

		if step%50 == 0 || step == steps-1 {
			acc := accuracy(logits, lbl, classes)
			lossCPU, _ := loss.To(device.CPU())
			fmt.Printf("step %4d  loss = %.4f  batch-acc = %.2f%%\n",
				step, tensor.MustSlice[float32](lossCPU)[0], acc*100)
		}
	}

	// Held-out evaluation on 10 fresh batches.
	fmt.Println("\n--- held-out evaluation (160 samples) ---")
	correct := 0
	for range 10 {
		tok, lbl := makeBatch(rng)
		logits, _ := forward(tok)
		logitsCPU, _ := logits.To(device.CPU())
		lblCPU, _ := lbl.To(device.CPU())
		flat := tensor.MustSlice[float32](logitsCPU)
		lbls := tensor.MustSlice[int64](lblCPU)
		for b := range batch {
			best := 0
			for c := 1; c < classes; c++ {
				if flat[b*classes+c] > flat[b*classes+best] {
					best = c
				}
			}
			if int64(best) == lbls[b] {
				correct++
			}
		}
	}
	fmt.Printf("accuracy: %.1f%% (%d/160)\n", float32(correct)/160*100, correct)
}

func accuracy(logits, labels *tensor.Tensor, classes int) float32 {
	logitsCPU, _ := logits.To(device.CPU())
	labelsCPU, _ := labels.To(device.CPU())
	flat := tensor.MustSlice[float32](logitsCPU)
	lbls := tensor.MustSlice[int64](labelsCPU)
	B := len(lbls)
	correct := 0
	for b := range B {
		best := 0
		for c := 1; c < classes; c++ {
			if flat[b*classes+c] > flat[b*classes+best] {
				best = c
			}
		}
		if int64(best) == lbls[b] {
			correct++
		}
	}
	return float32(correct) / float32(B)
}
