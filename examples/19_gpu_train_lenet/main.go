// Example 19 — train LeNet-5 fully on GPU.
//
// Same architecture as example 15 (Conv2d → ReLU → MaxPool → Conv2d → ReLU →
// MaxPool → Flatten → 3× Linear with ReLU), but every tensor lives on CUDA.
// Exercises the CUDA Conv2d/MaxPool2d forward + backward kernels, plus the
// device-agnostic AdamW and CrossEntropyLoss.
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
	steps   = 200
)

func drawHorizontal(img []float32, rng *rand.Rand) {
	row := rng.IntN(imgSize-4) + 2
	for c := range imgSize {
		img[row*imgSize+c] = 1
	}
}
func drawVertical(img []float32, rng *rand.Rand) {
	col := rng.IntN(imgSize-4) + 2
	for r := range imgSize {
		img[r*imgSize+col] = 1
	}
}
func drawX(img []float32) {
	for k := range imgSize {
		img[k*imgSize+k] = 1
		img[k*imgSize+(imgSize-1-k)] = 1
	}
}

func makeBatch(rng *rand.Rand, dev device.Device) (*tensor.Tensor, *tensor.Tensor) {
	imgs := make([]float32, batch*imgSize*imgSize)
	labels := make([]int64, batch)
	for b := range batch {
		cls := rng.IntN(classes)
		labels[b] = int64(cls)
		off := b * imgSize * imgSize
		switch cls {
		case 0:
			drawHorizontal(imgs[off:off+imgSize*imgSize], rng)
		case 1:
			drawVertical(imgs[off:off+imgSize*imgSize], rng)
		case 2:
			drawX(imgs[off : off+imgSize*imgSize])
		}
		for i := off; i < off+imgSize*imgSize; i++ {
			imgs[i] += float32(rng.NormFloat64()) * 0.05
		}
	}
	x, _ := tensor.New(imgs, []int{batch, 1, imgSize, imgSize}, dev)
	y, _ := tensor.New(labels, []int{batch}, dev)
	return x, y
}

func main() {
	runtime.LockOSThread()
	gpu, err := device.CUDA(0)
	if err != nil {
		log.Fatalf("CUDA unavailable: %v", err)
	}
	fmt.Printf("training LeNet-5 on GPU: %v\n", gpu)
	rng := rand.New(rand.NewPCG(17, 23))

	conv1, _ := nn.NewConv2d(1, 6, 5, 5, [2]int{1, 1}, [2]int{2, 2}, true, gpu)
	conv2, _ := nn.NewConv2d(6, 16, 5, 5, [2]int{1, 1}, [2]int{0, 0}, true, gpu)
	pool := nn.NewMaxPool2d([2]int{2, 2}, [2]int{2, 2}, [2]int{0, 0})
	fc1, _ := nn.NewLinear(16*5*5, 120, true, gpu)
	fc2, _ := nn.NewLinear(120, 84, true, gpu)
	fc3, _ := nn.NewLinear(84, classes, true, gpu)

	forward := func(x *tensor.Tensor) (*tensor.Tensor, error) {
		for _, step := range []func(*tensor.Tensor) (*tensor.Tensor, error){
			conv1.Forward, autograd.ReLU, pool.Forward,
			conv2.Forward, autograd.ReLU, pool.Forward,
			(nn.Flatten{}).Forward,
			fc1.Forward, autograd.ReLU,
			fc2.Forward, autograd.ReLU,
			fc3.Forward,
		} {
			var err error
			x, err = step(x)
			if err != nil {
				return nil, err
			}
		}
		return x, nil
	}

	var params []*tensor.Tensor
	for _, m := range []nn.Module{conv1, conv2, fc1, fc2, fc3} {
		params = append(params, m.Parameters()...)
	}
	total := 0
	for _, p := range params {
		total += p.NumElements()
	}
	fmt.Printf("parameters: %d\n", total)

	opt, err := training.NewAdamW(params, training.AdamWOptions{LR: 0.002})
	if err != nil {
		log.Fatal(err)
	}

	for step := range steps {
		x, y := makeBatch(rng, gpu)
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
		if step%25 == 0 || step == steps-1 {
			lossCPU, _ := loss.To(device.CPU())
			logitsCPU, _ := logits.To(device.CPU())
			labelsCPU, _ := y.To(device.CPU())
			fmt.Printf("step %4d  loss = %.5f  batch-acc = %.2f%%\n",
				step, tensor.MustSlice[float32](lossCPU)[0],
				batchAcc(logitsCPU, labelsCPU)*100)
		}
	}
	fmt.Println("\nGPU CNN training complete.")
}

func batchAcc(logits, labels *tensor.Tensor) float32 {
	flat := tensor.MustSlice[float32](logits)
	lbl := tensor.MustSlice[int64](labels)
	B := len(lbl)
	C := len(flat) / B
	correct := 0
	for b := range B {
		best := 0
		for c := 1; c < C; c++ {
			if flat[b*C+c] > flat[b*C+best] {
				best = c
			}
		}
		if int64(best) == lbl[b] {
			correct++
		}
	}
	return float32(correct) / float32(B)
}
