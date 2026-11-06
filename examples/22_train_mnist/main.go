// Example 22 — train a LeNet-style CNN on real MNIST.
//
// On first run the 4 MNIST files are downloaded into ./mnist-cache/ and
// then cached. All subsequent runs read from disk.
//
// Runs on CPU by default; pass `-device gpu` (requires `-tags cuda`) for
// CUDA. Expected: ≥97% test accuracy after ~2 epochs on GPU.
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"runtime"
	"time"

	"light/autograd"
	"light/data"
	"light/device"
	"light/nn"
	"light/tensor"
	"light/training"
)

const (
	classes = 10
	batch   = 64
)

// LeNet-style net tailored to MNIST 28×28.
type LeNet struct {
	conv1, conv2 *nn.Conv2d
	pool         *nn.MaxPool2d
	fc1, fc2     *nn.Linear
}

func newLeNet(dev device.Device) *LeNet {
	c1, _ := nn.NewConv2d(1, 16, 5, 5, [2]int{1, 1}, [2]int{2, 2}, true, dev)  // 28→28
	c2, _ := nn.NewConv2d(16, 32, 5, 5, [2]int{1, 1}, [2]int{0, 0}, true, dev) // 14→10
	fc1, _ := nn.NewLinear(32*5*5, 128, true, dev)
	fc2, _ := nn.NewLinear(128, classes, true, dev)
	return &LeNet{
		conv1: c1, conv2: c2,
		pool: nn.NewMaxPool2d([2]int{2, 2}, [2]int{2, 2}, [2]int{0, 0}),
		fc1:  fc1, fc2: fc2,
	}
}

func (l *LeNet) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, m := range []nn.Module{l.conv1, l.conv2, l.fc1, l.fc2} {
		ps = append(ps, m.Parameters()...)
	}
	return ps
}

func (l *LeNet) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	for _, step := range []func(*tensor.Tensor) (*tensor.Tensor, error){
		l.conv1.Forward, autograd.ReLU, l.pool.Forward,
		l.conv2.Forward, autograd.ReLU, l.pool.Forward,
		(nn.Flatten{}).Forward,
		l.fc1.Forward, autograd.ReLU, l.fc2.Forward,
	} {
		var err error
		x, err = step(x)
		if err != nil {
			return nil, err
		}
	}
	return x, nil
}

func main() {
	deviceFlag := flag.String("device", "gpu", "cpu or gpu")
	epochs := flag.Int("epochs", 1, "number of epochs over the full train set")
	flag.Parse()

	var dev device.Device
	if *deviceFlag == "gpu" {
		runtime.LockOSThread()
		g, err := device.CUDA(0)
		if err != nil {
			log.Fatalf("CUDA unavailable: %v", err)
		}
		dev = g
	} else {
		dev = device.CPU()
	}
	fmt.Printf("device: %v\n", dev)

	// Always load data onto CPU first — for batching we slice per step.
	mnist, err := data.LoadMNIST("./mnist-cache", device.CPU())
	if err != nil {
		log.Fatalf("load MNIST: %v", err)
	}
	trainN := mnist.TrainImages.Dim(0)
	testN := mnist.TestImages.Dim(0)
	fmt.Printf("MNIST: %d train, %d test\n", trainN, testN)

	net := newLeNet(dev)
	total := 0
	for _, p := range net.Parameters() {
		total += p.NumElements()
	}
	fmt.Printf("LeNet: %d params\n", total)
	opt, err := training.NewAdamW(net.Parameters(), training.AdamWOptions{LR: 0.001})
	if err != nil {
		log.Fatal(err)
	}

	// Get raw CPU buffers for cheap batch slicing.
	trainImgs := tensor.MustSlice[float32](mnist.TrainImages)
	trainLbls := tensor.MustSlice[int64](mnist.TrainLabels)
	testImgs := tensor.MustSlice[float32](mnist.TestImages)
	testLbls := tensor.MustSlice[int64](mnist.TestLabels)
	const imgPx = 28 * 28

	rng := rand.New(rand.NewPCG(1, 1))
	start := time.Now()
	stepsPerEpoch := trainN / batch
	for epoch := 1; epoch <= *epochs; epoch++ {
		// Shuffle indices.
		perm := make([]int, trainN)
		for i := range perm {
			perm[i] = i
		}
		rng.Shuffle(trainN, func(i, j int) { perm[i], perm[j] = perm[j], perm[i] })

		var lastLoss float32
		for step := 0; step < stepsPerEpoch; step++ {
			imgs := make([]float32, batch*imgPx)
			lbls := make([]int64, batch)
			for b := range batch {
				idx := perm[step*batch+b]
				copy(imgs[b*imgPx:(b+1)*imgPx], trainImgs[idx*imgPx:(idx+1)*imgPx])
				lbls[b] = trainLbls[idx]
			}
			x, _ := tensor.New(imgs, []int{batch, 1, 28, 28}, dev)
			y, _ := tensor.New(lbls, []int{batch}, dev)

			opt.ZeroGrad()
			logits, err := net.Forward(x)
			if err != nil {
				log.Fatalf("step %d forward: %v", step, err)
			}
			loss, err := autograd.CrossEntropyLoss(logits, y)
			if err != nil {
				log.Fatalf("step %d loss: %v", step, err)
			}
			if err := autograd.Backward(loss, nil); err != nil {
				log.Fatalf("step %d backward: %v", step, err)
			}
			if err := opt.Step(); err != nil {
				log.Fatalf("step %d optim: %v", step, err)
			}
			if step%100 == 0 || step == stepsPerEpoch-1 {
				lossCPU, _ := loss.To(device.CPU())
				lastLoss = tensor.MustSlice[float32](lossCPU)[0]
				fmt.Printf("epoch %d step %4d/%d  loss = %.4f\n",
					epoch, step, stepsPerEpoch, lastLoss)
			}
			// Force finalizers so GPU memory is freed before the next
			// allocations. Without this, Go's GC can lag behind CUDA
			// allocations and we OOM after a few hundred steps.
			if step%20 == 0 {
				runtime.GC()
			}
		}
		fmt.Printf("epoch %d done in %v, final loss %.4f\n",
			epoch, time.Since(start).Round(time.Millisecond), lastLoss)
	}

	// Test-set accuracy.
	correct := 0
	for off := 0; off+batch <= testN; off += batch {
		imgs := make([]float32, batch*imgPx)
		lbls := make([]int64, batch)
		for b := range batch {
			copy(imgs[b*imgPx:(b+1)*imgPx], testImgs[(off+b)*imgPx:(off+b+1)*imgPx])
			lbls[b] = testLbls[off+b]
		}
		x, _ := tensor.New(imgs, []int{batch, 1, 28, 28}, dev)
		logits, _ := net.Forward(x)
		logitsCPU, _ := logits.To(device.CPU())
		flat := tensor.MustSlice[float32](logitsCPU)
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
	evalN := (testN / batch) * batch
	fmt.Printf("\ntest accuracy: %.2f%% (%d/%d)\n",
		float32(correct)/float32(evalN)*100, correct, evalN)
}
