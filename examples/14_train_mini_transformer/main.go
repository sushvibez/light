// Example 14 — train a tiny sequence classifier using every transformer
// primitive we have autograd for:
//
//	tokens [B, T]                (I64)
//	 ↓   Embedding
//	hidden [B, T, D]
//	 ↓   LayerNorm (last axis)
//	hidden [B, T, D]
//	 ↓   AxisMean (axis=1 over T)
//	pooled [B, D]
//	 ↓   Linear (D → K classes)
//	logits [B, K]
//	 ↓   CrossEntropyLoss
//	loss
//
// Task: predict the majority-class "bucket" of the sequence. Tokens 0..2 are
// class 0, tokens 3..5 are class 1, tokens 6..7 are class 2; the label is
// whichever bucket has the most tokens in the sequence. Mean-pooled
// embeddings can solve this trivially because bucket id is a linear function
// of the bucket-indicator embeddings.
//
// This is the first example that trains _through_ an Embedding, LayerNorm,
// and CrossEntropyLoss end-to-end — the exact ingredients a real transformer
// classifier uses.
package main

import (
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"runtime"

	"light/autograd"
	"light/device"
	"light/nn"
	"light/tensor"
	"light/training"
)

func main() {
	runtime.LockOSThread()
	dev := device.Auto()
	fmt.Printf("device: %v\n", dev)
	const (
		vocab   = 8
		seq     = 4
		embDim  = 16
		classes = 3
		batch   = 64
		steps   = 2000
	)

	// Map token id → bucket (class).
	bucket := func(id int64) int64 {
		switch {
		case id < 3:
			return 0
		case id < 6:
			return 1
		default:
			return 2
		}
	}
	tokData := make([]int64, batch*seq)
	lblData := make([]int64, batch)
	for b := range batch {
		counts := make([]int, classes)
		for t := range seq {
			tok := int64(rand.IntN(vocab))
			tokData[b*seq+t] = tok
			counts[bucket(tok)]++
		}
		best := 0
		for c := 1; c < classes; c++ {
			if counts[c] > counts[best] {
				best = c
			}
		}
		lblData[b] = int64(best)
	}
	tokens, _ := tensor.New(tokData, []int{batch, seq}, dev)
	labels, _ := tensor.New(lblData, []int{batch}, dev)

	emb, _ := nn.NewEmbedding(vocab, embDim, dev)
	norm, _ := nn.NewLayerNorm(embDim, 1e-5, dev)
	head, _ := nn.NewLinear(embDim, classes, true, dev)

	var params []*tensor.Tensor
	params = append(params, emb.Parameters()...)
	params = append(params, norm.Parameters()...)
	params = append(params, head.Parameters()...)

	opt, err := training.NewAdamW(params, training.AdamWOptions{LR: 0.02})
	if err != nil {
		log.Fatal(err)
	}

	forward := func() (*tensor.Tensor, *tensor.Tensor, error) {
		h, err := emb.Forward(tokens) // [B, T, D]
		if err != nil {
			return nil, nil, err
		}
		h, err = norm.Forward(h)
		if err != nil {
			return nil, nil, err
		}
		pooled, err := autograd.AxisMean(h, []int{1}, false) // [B, D]
		if err != nil {
			return nil, nil, err
		}
		logits, err := head.Forward(pooled) // [B, K]
		if err != nil {
			return nil, nil, err
		}
		loss, err := autograd.CrossEntropyLoss(logits, labels)
		return loss, logits, err
	}

	var first float32
	for step := range steps {
		opt.ZeroGrad()
		loss, _, err := forward()
		if err != nil {
			log.Fatal(err)
		}
		if err := autograd.Backward(loss, nil); err != nil {
			log.Fatal(err)
		}
		if err := opt.Step(); err != nil {
			log.Fatal(err)
		}
		lossCPU, _ := loss.To(device.CPU())
		val := tensor.MustSlice[float32](lossCPU)[0]
		if step == 0 {
			first = val
		}
		if step%200 == 0 || step == steps-1 {
			_, logits, _ := forward()
			fmt.Printf("step %4d  loss = %.5f  train-acc = %.2f%%\n",
				step, val, accuracy(logits, labels, classes)*100)
		}
	}
	fmt.Printf("\nstart loss %.4f (random = ln(%d) ≈ %.4f)\n",
		first, classes, math.Log(float64(classes)))
}

func accuracy(logits, labels *tensor.Tensor, classes int) float32 {
	logitsCPU, _ := logits.To(device.CPU())
	labelsCPU, _ := labels.To(device.CPU())
	flat := tensor.MustSlice[float32](logitsCPU)
	lbl := tensor.MustSlice[int64](labelsCPU)
	B := len(lbl)
	correct := 0
	for b := range B {
		best := 0
		bestVal := flat[b*classes]
		for c := 1; c < classes; c++ {
			if v := flat[b*classes+c]; v > bestVal {
				bestVal = v
				best = c
			}
		}
		if int64(best) == lbl[b] {
			correct++
		}
	}
	return float32(correct) / float32(B)
}
