// Example 20 — train a 2-layer batched transformer fully on GPU.
//
// Same architecture as example 17 (Embedding → 2× TransformerBlock → LayerNorm
// → mean-pool → Linear → CrossEntropyLoss) but every tensor on CUDA. This is
// the full transformer-training stack running on GPU:
//
//   - Embedding.Forward   → autograd.IndexSelect + Reshape on CUDA
//   - TransformerBlock    → LayerNorm + SelfAttention + FFN, all CUDA
//   - SelfAttention       → autograd.MatMul (rank-3 batched) + Softmax
//   - LayerNorm backward  → composed ops, CUDA
//   - CrossEntropyLoss    → composed ops, CUDA
//   - AdamW.Step          → ops.Mul/Add/Sub/Div/Sqrt + tensor.CopyFrom (DtoD)
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
	seq     = 8
	embDim  = 24
	classes = 4
	batch   = 16
	steps   = 400
)

func makeBatch(rng *rand.Rand, dev device.Device) (*tensor.Tensor, *tensor.Tensor) {
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
	gpu, err := device.CUDA(0)
	if err != nil {
		log.Fatalf("CUDA unavailable: %v", err)
	}
	fmt.Printf("training batched transformer on GPU: %v\n", gpu)
	rng := rand.New(rand.NewPCG(5, 7))

	emb, _ := nn.NewEmbedding(vocab, embDim, gpu)
	block1, _ := nn.NewTransformerBlock(embDim, false, gpu)
	block2, _ := nn.NewTransformerBlock(embDim, false, gpu)
	finalNorm, _ := nn.NewLayerNorm(embDim, 1e-5, gpu)
	head, _ := nn.NewLinear(embDim, classes, true, gpu)

	var params []*tensor.Tensor
	for _, m := range []nn.Module{emb, block1, block2, finalNorm, head} {
		params = append(params, m.Parameters()...)
	}
	total := 0
	for _, p := range params {
		total += p.NumElements()
	}
	fmt.Printf("parameters: %d\n", total)

	opt, err := training.NewAdamW(params, training.AdamWOptions{LR: 0.005})
	if err != nil {
		log.Fatal(err)
	}

	forward := func(tok *tensor.Tensor) (*tensor.Tensor, error) {
		h, err := emb.Forward(tok)
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
		pooled, err := autograd.AxisMean(h, []int{1}, false)
		if err != nil {
			return nil, err
		}
		return head.Forward(pooled)
	}

	for step := range steps {
		tok, lbl := makeBatch(rng, gpu)
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
			lossCPU, _ := loss.To(device.CPU())
			logitsCPU, _ := logits.To(device.CPU())
			labelsCPU, _ := lbl.To(device.CPU())
			fmt.Printf("step %4d  loss = %.5f  batch-acc = %.2f%%\n",
				step, tensor.MustSlice[float32](lossCPU)[0],
				batchAcc(logitsCPU, labelsCPU, classes)*100)
		}
	}

	fmt.Println("\nGPU transformer training complete.")
}

func batchAcc(logits, labels *tensor.Tensor, classes int) float32 {
	flat := tensor.MustSlice[float32](logits)
	lbl := tensor.MustSlice[int64](labels)
	B := len(lbl)
	correct := 0
	for b := range B {
		best := 0
		for c := 1; c < classes; c++ {
			if flat[b*classes+c] > flat[b*classes+best] {
				best = c
			}
		}
		if int64(best) == lbl[b] {
			correct++
		}
	}
	return float32(correct) / float32(B)
}
