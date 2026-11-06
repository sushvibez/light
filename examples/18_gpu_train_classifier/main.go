// Example 18 — train a 3-class MLP classifier fully on the GPU.
//
// Same task as example 13 (3 Gaussian blobs arranged at 0°/120°/240°) but
// every tensor lives on CUDA: parameters, activations, gradients, optimizer
// state. This is the first example that trains end-to-end on GPU:
//
//	nn.Linear.Forward  → autograd.MatMul/Add on CUDA
//	nn.ReLU.Forward    → autograd.ReLU on CUDA
//	CrossEntropyLoss   → composed ops all CUDA-dispatched
//	AdamW.Step         → ops.Mul/Add/Sub/Div/Sqrt all CUDA + tensor.CopyFrom via cudaMemcpyDtoD
//
// Runs on any CUDA device (`go run            ./examples/18_gpu_train_classifier`).
// Falls back with a clear error if CUDA is unavailable.
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

const (
	classes = 3
	perCls  = 80
	steps   = 1500
)

func makeBlobs(rng *rand.Rand) (xs []float32, ys []int64) {
	n := perCls * classes
	xs = make([]float32, n*2)
	ys = make([]int64, n)
	for c := range classes {
		angle := float64(c) * 2 * math.Pi / float64(classes)
		cx := float32(math.Cos(angle))
		cy := float32(math.Sin(angle))
		for i := range perCls {
			k := c*perCls + i
			xs[k*2] = cx + 0.25*float32(rng.NormFloat64())
			xs[k*2+1] = cy + 0.25*float32(rng.NormFloat64())
			ys[k] = int64(c)
		}
	}
	return xs, ys
}

func main() {
	// Pin the main goroutine to one OS thread so the CUDA driver-API
	// context stays attached throughout the training loop.
	runtime.LockOSThread()

	gpu, err := device.CUDA(0)
	if err != nil {
		log.Fatalf("CUDA unavailable: %v (build with -tags cuda and ensure a GPU is present)", err)
	}
	fmt.Printf("training on GPU: %v\n", gpu)
	rng := rand.New(rand.NewPCG(1, 2))

	xsCPU, ysCPU := makeBlobs(rng)
	xCPU, _ := tensor.New(xsCPU, []int{perCls * classes, 2}, device.CPU())
	yCPU, _ := tensor.New(ysCPU, []int{perCls * classes}, device.CPU())
	// Move data to GPU once.
	x, err := xCPU.To(gpu)
	if err != nil {
		log.Fatal(err)
	}
	y, err := yCPU.To(gpu)
	if err != nil {
		log.Fatal(err)
	}

	// Model weights initialised directly on GPU.
	l1, _ := nn.NewLinear(2, 32, true, gpu)
	l2, _ := nn.NewLinear(32, 32, true, gpu)
	l3, _ := nn.NewLinear(32, classes, true, gpu)
	model := nn.NewSequential(l1, nn.ReLU{}, l2, nn.ReLU{}, l3)

	opt, err := training.NewAdamW(model.Parameters(), training.AdamWOptions{LR: 0.01})
	if err != nil {
		log.Fatal(err)
	}

	for step := range steps {
		opt.ZeroGrad()
		logits, err := model.Forward(x)
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
		if step%200 == 0 || step == steps-1 {
			lossCPU, _ := loss.To(device.CPU())
			logitsCPU, _ := logits.To(device.CPU())
			yLabels, _ := y.To(device.CPU())
			acc := accuracy(logitsCPU, yLabels, classes)
			fmt.Printf("step %4d  loss = %.5f  train-acc = %.2f%%\n",
				step, tensor.MustSlice[float32](lossCPU)[0], acc*100)
		}
	}
	fmt.Println("\nGPU training complete — gradients flowed through CUDA kernels on every op.")
}

func accuracy(logits, labels *tensor.Tensor, classes int) float32 {
	flat := tensor.MustSlice[float32](logits)
	lbl := tensor.MustSlice[int64](labels)
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
