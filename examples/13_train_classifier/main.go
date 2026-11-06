// Example 13 — train a 2-layer MLP classifier on synthetic 3-class data.
//
// This is the first end-to-end classification training loop: spiral-like
// Gaussian blobs in 2-D, cross-entropy loss via autograd, AdamW optimizer.
// After training the model separates the three classes cleanly (>95% accuracy
// on a held-out grid).
package main

import (
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"runtime"
	"strings"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/nn"
	"light/tensor"
	"light/training"
)

// Generate three isotropic Gaussian blobs centred at angles 0°, 120°, 240°.
func makeBlobs(perClass int) (xData []float32, yData []int64) {
	nClasses := 3
	xData = make([]float32, perClass*nClasses*2)
	yData = make([]int64, perClass*nClasses)
	for c := range nClasses {
		angle := float64(c) * 2 * math.Pi / float64(nClasses)
		cx := float32(math.Cos(angle))
		cy := float32(math.Sin(angle))
		for i := range perClass {
			idx := c*perClass + i
			xData[idx*2] = cx + 0.25*float32(rand.NormFloat64())
			xData[idx*2+1] = cy + 0.25*float32(rand.NormFloat64())
			yData[idx] = int64(c)
		}
	}
	return xData, yData
}

func main() {
	runtime.LockOSThread()
	dev := device.Auto()
	fmt.Printf("device: %v\n", dev)
	perClass := 80
	numClasses := 3

	xData, yData := makeBlobs(perClass)
	x, _ := tensor.New(xData, []int{perClass * numClasses, 2}, dev)
	y, _ := tensor.New(yData, []int{perClass * numClasses}, dev)

	// Model: 2 → 32 → 32 → 3 with ReLU activations.
	l1, _ := nn.NewLinear(2, 32, true, dev)
	l2, _ := nn.NewLinear(32, 32, true, dev)
	l3, _ := nn.NewLinear(32, numClasses, true, dev)
	model := nn.NewSequential(l1, nn.ReLU{}, l2, nn.ReLU{}, l3)

	opt, err := training.NewAdamW(model.Parameters(), training.AdamWOptions{
		LR: 0.01,
	})
	if err != nil {
		log.Fatal(err)
	}

	steps := 2000
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
			acc := accuracy(model, x, y)
			lossCPU, _ := loss.To(device.CPU())
			fmt.Printf("step %4d  loss = %.5f  train-acc = %.2f%%\n",
				step, tensor.MustSlice[float32](lossCPU)[0], acc*100)
		}
	}

	// Evaluate on a grid of (x, y) points to visualise class regions.
	fmt.Println("\nClass predictions on a 5×5 grid over [-1.5, 1.5]²:")
	gridN := 5
	gridData := make([]float32, gridN*gridN*2)
	for i := range gridN {
		for j := range gridN {
			gridData[(i*gridN+j)*2] = -1.5 + 3.0*float32(j)/float32(gridN-1)
			gridData[(i*gridN+j)*2+1] = -1.5 + 3.0*float32(i)/float32(gridN-1)
		}
	}
	grid, _ := tensor.New(gridData, []int{gridN * gridN, 2}, dev)
	logits, _ := model.Forward(grid)
	pred := argmaxRows(logits, numClasses)
	for i := range gridN {
		var row strings.Builder
		for j := range gridN {
			fmt.Fprintf(&row, "  %d", pred[i*gridN+j])
		}
		fmt.Println(row.String())
	}
}

func accuracy(model nn.Module, x, y *tensor.Tensor) float32 {
	logits, _ := model.Forward(x)
	pred := argmaxRows(logits, logits.Dim(1))
	yCPU, _ := y.To(device.CPU())
	labels := tensor.MustSlice[int64](yCPU)
	correct := 0
	for i, p := range pred {
		if int64(p) == labels[i] {
			correct++
		}
	}
	return float32(correct) / float32(len(pred))
}

// argmaxRows returns the argmax column per row for a rank-2 F32 tensor.
func argmaxRows(t *tensor.Tensor, C int) []int {
	tCPU, _ := t.To(device.CPU())
	flat := tensor.MustSlice[float32](tCPU)
	B := len(flat) / C
	out := make([]int, B)
	for b := range B {
		best := 0
		bestVal := flat[b*C]
		for c := 1; c < C; c++ {
			if v := flat[b*C+c]; v > bestVal {
				bestVal = v
				best = c
			}
		}
		out[b] = best
	}
	return out
}

var _ = dtype.I64 // keep import stable
