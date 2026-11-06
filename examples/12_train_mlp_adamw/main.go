// Example 12 — train an MLP with nn.Sequential + autograd + AdamW.
//
// This is the cleanest "training loop" example yet: the forward graph is
// built inside nn modules (Linear and Tanh), the backward pass is run via
// autograd, and parameter updates go through training.AdamW — the same
// shape you'd see in PyTorch. No hand-rolled SGD update.
//
// Task: learn y = sin(3x) on x ∈ [-1, 1] with a 2-layer MLP.
package main

import (
	"fmt"
	"log"
	"math"
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

	// Synthetic regression: y = sin(3x).
	n := 64
	xData := make([]float32, n)
	yData := make([]float32, n)
	for i := range xData {
		xData[i] = -1 + 2*float32(i)/float32(n-1)
		yData[i] = float32(math.Sin(3 * float64(xData[i])))
	}
	x, _ := tensor.New(xData, []int{n, 1}, dev)
	y, _ := tensor.New(yData, []int{n, 1}, dev)

	// Model: 1 → 32 → 32 → 1 with Tanh activations.
	l1, _ := nn.NewLinear(1, 32, true, dev)
	l2, _ := nn.NewLinear(32, 32, true, dev)
	l3, _ := nn.NewLinear(32, 1, true, dev)
	model := nn.NewSequential(l1, nn.Tanh{}, l2, nn.Tanh{}, l3)

	fmt.Printf("model parameters: %d tensors, %d total floats\n",
		len(model.Parameters()), countElems(model.Parameters()))

	opt, err := training.NewAdamW(model.Parameters(), training.AdamWOptions{
		LR:    0.01,
		Beta1: 0.9,
		Beta2: 0.999,
	})
	if err != nil {
		log.Fatal(err)
	}

	for step := range 3000 {
		opt.ZeroGrad()

		pred, err := model.Forward(x)
		if err != nil {
			log.Fatal(err)
		}
		diff, _ := autograd.Sub(pred, y)
		sq, _ := autograd.Mul(diff, diff)
		loss, err := autograd.Mean(sq)
		if err != nil {
			log.Fatal(err)
		}

		if err := autograd.Backward(loss, nil); err != nil {
			log.Fatal(err)
		}
		if err := opt.Step(); err != nil {
			log.Fatal(err)
		}

		if step%500 == 0 || step == 2999 {
			lossCPU, _ := loss.To(device.CPU())
			fmt.Printf("step %4d  loss = %.6f\n", step, tensor.MustSlice[float32](lossCPU)[0])
		}
	}

	// Evaluate at a few points.
	fmt.Println("\nEvaluation (x → predicted vs target sin(3x)):")
	pred, _ := model.Forward(x)
	predCPU, _ := pred.To(device.CPU())
	yCPU, _ := y.To(device.CPU())
	ps := tensor.MustSlice[float32](predCPU)
	ts := tensor.MustSlice[float32](yCPU)
	for _, i := range []int{0, n / 4, n / 2, 3 * n / 4, n - 1} {
		fmt.Printf("  x=%+.3f  pred=%+.4f  target=%+.4f\n", xData[i], ps[i], ts[i])
	}
}

func countElems(params []*tensor.Tensor) int {
	total := 0
	for _, p := range params {
		total += p.NumElements()
	}
	return total
}
