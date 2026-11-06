// Example 15 — train a LeNet-5-style CNN on synthetic 28×28 images.
//
// Architecture (PyTorch-compatible):
//
//	Conv2d(1, 6, 5, padding=2) ─► ReLU ─► MaxPool(2,2)
//	Conv2d(6, 16, 5)           ─► ReLU ─► MaxPool(2,2)
//	Flatten ─► Linear(16·5·5 = 400 → 120) ─► ReLU
//	        ─► Linear(120 → 84)            ─► ReLU
//	        ─► Linear(84 → 3)              (classes)
//
// Task: 3-class shape classification on synthetic 28×28 images. Class 0 is
// a horizontal stripe, class 1 is a vertical stripe, class 2 is an X. The
// CNN has ~60k parameters and learns the task to >95% accuracy in a couple
// hundred AdamW steps.
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
	imgSize = 28
	batch   = 48
	classes = 3
	steps   = 400
)

func drawHorizontalStripe(img []float32, rng *rand.Rand) {
	row := rng.IntN(imgSize-4) + 2
	for c := range imgSize {
		img[row*imgSize+c] = 1
	}
}

func drawVerticalStripe(img []float32, rng *rand.Rand) {
	col := rng.IntN(imgSize-4) + 2
	for r := range imgSize {
		img[r*imgSize+col] = 1
	}
}

func drawX(img []float32, rng *rand.Rand) {
	for k := range imgSize {
		img[k*imgSize+k] = 1
		img[k*imgSize+(imgSize-1-k)] = 1
	}
	_ = rng
}

func makeBatch(rng *rand.Rand) (*tensor.Tensor, *tensor.Tensor) {
	imgs := make([]float32, batch*imgSize*imgSize)
	labels := make([]int64, batch)
	for b := range batch {
		cls := rng.IntN(classes)
		labels[b] = int64(cls)
		off := b * imgSize * imgSize
		switch cls {
		case 0:
			drawHorizontalStripe(imgs[off:off+imgSize*imgSize], rng)
		case 1:
			drawVerticalStripe(imgs[off:off+imgSize*imgSize], rng)
		case 2:
			drawX(imgs[off:off+imgSize*imgSize], rng)
		}
		// Add light noise.
		for i := off; i < off+imgSize*imgSize; i++ {
			imgs[i] += float32(rng.NormFloat64()) * 0.05
		}
	}
	x, _ := tensor.New(imgs, []int{batch, 1, imgSize, imgSize}, batchDevice)
	y, _ := tensor.New(labels, []int{batch}, batchDevice)
	return x, y
}

// batchDevice is set in main() so makeBatch can build batches on the right
// device without each call having to take a parameter.
var batchDevice device.Device

func main() {
	runtime.LockOSThread()
	gpu, err := device.CUDA(0)
	if err != nil {
		log.Fatal(err)
	}
	batchDevice = gpu
	rng := rand.New(rand.NewPCG(42, 0))

	// Build model.
	conv1, _ := nn.NewConv2d(1, 6, 5, 5, [2]int{1, 1}, [2]int{2, 2}, true, gpu)
	conv2, _ := nn.NewConv2d(6, 16, 5, 5, [2]int{1, 1}, [2]int{0, 0}, true, gpu)
	pool := nn.NewMaxPool2d([2]int{2, 2}, [2]int{2, 2}, [2]int{0, 0})
	fc1, _ := nn.NewLinear(16*5*5, 120, true, gpu)
	fc2, _ := nn.NewLinear(120, 84, true, gpu)
	fc3, _ := nn.NewLinear(84, classes, true, gpu)

	forward := func(x *tensor.Tensor) (*tensor.Tensor, error) {
		h, err := conv1.Forward(x)
		if err != nil {
			return nil, err
		}
		h, err = autograd.ReLU(h)
		if err != nil {
			return nil, err
		}
		h, err = pool.Forward(h)
		if err != nil {
			return nil, err
		}
		h, err = conv2.Forward(h)
		if err != nil {
			return nil, err
		}
		h, err = autograd.ReLU(h)
		if err != nil {
			return nil, err
		}
		h, err = pool.Forward(h)
		if err != nil {
			return nil, err
		}
		h, err = (nn.Flatten{}).Forward(h)
		if err != nil {
			return nil, err
		}
		h, err = fc1.Forward(h)
		if err != nil {
			return nil, err
		}
		h, err = autograd.ReLU(h)
		if err != nil {
			return nil, err
		}
		h, err = fc2.Forward(h)
		if err != nil {
			return nil, err
		}
		h, err = autograd.ReLU(h)
		if err != nil {
			return nil, err
		}
		return fc3.Forward(h)
	}

	var params []*tensor.Tensor
	for _, m := range []nn.Module{conv1, conv2, fc1, fc2, fc3} {
		params = append(params, m.Parameters()...)
	}
	paramCount := 0
	for _, p := range params {
		paramCount += p.NumElements()
	}
	fmt.Printf("LeNet-5: %d parameters\n", paramCount)

	opt, err := training.NewAdamW(params, training.AdamWOptions{LR: 0.002})
	if err != nil {
		log.Fatal(err)
	}

	for step := range steps {
		x, y := makeBatch(rng)
		opt.ZeroGrad()
		logits, err := forward(x)
		if err != nil {
			log.Fatal(err)
		}
		loss, err := autograd.CrossEntropyLoss(logits, y)
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
			acc := batchAccuracy(logits, y)
			lossCPU, _ := loss.To(device.CPU())
			fmt.Printf("step %4d  loss = %.4f  batch-acc = %.2f%%\n",
				step, tensor.MustSlice[float32](lossCPU)[0], acc*100)
		}
	}

	// Final eval on a fresh batch.
	xEval, yEval := makeBatch(rng)
	logitsEval, _ := forward(xEval)
	fmt.Printf("\nfinal held-out batch acc = %.2f%%\n",
		batchAccuracy(logitsEval, yEval)*100)
}

func batchAccuracy(logits, labels *tensor.Tensor) float32 {
	logitsCPU, _ := logits.To(device.CPU())
	labelsCPU, _ := labels.To(device.CPU())
	flat := tensor.MustSlice[float32](logitsCPU)
	lbl := tensor.MustSlice[int64](labelsCPU)
	B := len(lbl)
	C := len(flat) / B
	correct := 0
	for b := range B {
		best := 0
		bestVal := flat[b*C]
		for c := 1; c < C; c++ {
			if v := flat[b*C+c]; v > bestVal {
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
