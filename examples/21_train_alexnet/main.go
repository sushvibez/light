// Example 21 — train AlexNet on synthetic 64×64 images, 4 classes.
//
// Runs on CPU by default; pass `-device gpu` (requires `-tags cuda`) to run
// on CUDA. The architecture follows the 2012 paper with modern tweaks: no
// LocalResponseNorm (replaced by BatchNorm2d which we support), 64×64
// inputs scaled down from 224×224, and half the channel counts so the
// network fits comfortably on a small GPU:
//
//	Conv2d(3, 48, k=11, s=4, p=2)    → ReLU → MaxPool2d(3, 2)
//	Conv2d(48, 128, k=5, s=1, p=2)   → ReLU → MaxPool2d(3, 2)
//	Conv2d(128, 192, k=3, s=1, p=1)  → ReLU
//	Conv2d(192, 192, k=3, s=1, p=1)  → ReLU
//	Conv2d(192, 128, k=3, s=1, p=1)  → ReLU → MaxPool2d(3, 2)
//	Flatten → Dropout → Linear(→2048) → ReLU
//	       → Dropout → Linear(→2048) → ReLU
//	       → Linear(→classes) → CrossEntropyLoss
//
// Task: 4-class synthetic image classification — class 0 has a horizontal
// stripe, 1 vertical, 2 diagonal, 3 random noise only. Exercises the full
// CNN stack: conv, pool, ReLU, Flatten, Dropout, Linear, CE, AdamW.
package main

import (
	"flag"
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
	imgH    = 64
	imgW    = 64
	batch   = 16
	classes = 4
	steps   = 200
)

func drawHoriz(img []float32, rng *rand.Rand) {
	row := rng.IntN(imgH-8) + 4
	for c := 0; c < 3; c++ {
		for w := range imgW {
			img[c*imgH*imgW+row*imgW+w] = 1
		}
	}
}
func drawVert(img []float32, rng *rand.Rand) {
	col := rng.IntN(imgW-8) + 4
	for c := 0; c < 3; c++ {
		for h := range imgH {
			img[c*imgH*imgW+h*imgW+col] = 1
		}
	}
}
func drawDiag(img []float32) {
	for c := 0; c < 3; c++ {
		for k := range imgH {
			w := k * imgW / imgH
			if w < imgW {
				img[c*imgH*imgW+k*imgW+w] = 1
			}
		}
	}
}

func makeBatch(rng *rand.Rand, dev device.Device) (*tensor.Tensor, *tensor.Tensor) {
	imgs := make([]float32, batch*3*imgH*imgW)
	labels := make([]int64, batch)
	for b := range batch {
		cls := rng.IntN(classes)
		labels[b] = int64(cls)
		off := b * 3 * imgH * imgW
		switch cls {
		case 0:
			drawHoriz(imgs[off:off+3*imgH*imgW], rng)
		case 1:
			drawVert(imgs[off:off+3*imgH*imgW], rng)
		case 2:
			drawDiag(imgs[off : off+3*imgH*imgW])
			// class 3: pure noise (no shape).
		}
		for i := off; i < off+3*imgH*imgW; i++ {
			imgs[i] += float32(rng.NormFloat64()) * 0.05
		}
	}
	x, _ := tensor.New(imgs, []int{batch, 3, imgH, imgW}, dev)
	y, _ := tensor.New(labels, []int{batch}, dev)
	return x, y
}

// AlexNet holds the full module tree.
type AlexNet struct {
	conv1, conv2, conv3, conv4, conv5 *nn.Conv2d
	pool                              *nn.MaxPool2d
	drop1, drop2                      *nn.Dropout
	fc1, fc2, fc3                     *nn.Linear
}

func newAlexNet(dev device.Device) *AlexNet {
	c1, _ := nn.NewConv2d(3, 48, 11, 11, [2]int{4, 4}, [2]int{2, 2}, true, dev)
	c2, _ := nn.NewConv2d(48, 128, 5, 5, [2]int{1, 1}, [2]int{2, 2}, true, dev)
	c3, _ := nn.NewConv2d(128, 192, 3, 3, [2]int{1, 1}, [2]int{1, 1}, true, dev)
	c4, _ := nn.NewConv2d(192, 192, 3, 3, [2]int{1, 1}, [2]int{1, 1}, true, dev)
	c5, _ := nn.NewConv2d(192, 128, 3, 3, [2]int{1, 1}, [2]int{1, 1}, true, dev)

	// Compute flattened spatial size after conv+pool stack on 64×64 input.
	// conv1 (k=11, s=4, p=2): (64+4-11)/4+1 = 15
	// pool (k=3, s=2):        (15-3)/2+1   = 7
	// conv2 (k=5, s=1, p=2):  (7+4-5)/1+1  = 7
	// pool:                   (7-3)/2+1    = 3
	// conv3/4/5: stride=1 padding=1 keep size 3
	// pool:                   (3-3)/2+1    = 1
	// ⇒ features = 128 * 1 * 1 = 128
	fc1, _ := nn.NewLinear(128, 2048, true, dev)
	fc2, _ := nn.NewLinear(2048, 2048, true, dev)
	fc3, _ := nn.NewLinear(2048, classes, true, dev)

	return &AlexNet{
		conv1: c1, conv2: c2, conv3: c3, conv4: c4, conv5: c5,
		pool:  nn.NewMaxPool2d([2]int{3, 3}, [2]int{2, 2}, [2]int{0, 0}),
		drop1: nn.NewDropout(0.5),
		drop2: nn.NewDropout(0.5),
		fc1:   fc1, fc2: fc2, fc3: fc3,
	}
}

func (a *AlexNet) SetTraining(v bool) {
	a.drop1.SetTraining(v)
	a.drop2.SetTraining(v)
}

func (a *AlexNet) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, m := range []nn.Module{a.conv1, a.conv2, a.conv3, a.conv4, a.conv5, a.fc1, a.fc2, a.fc3} {
		ps = append(ps, m.Parameters()...)
	}
	return ps
}

func (a *AlexNet) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	for _, step := range []func(*tensor.Tensor) (*tensor.Tensor, error){
		a.conv1.Forward, autograd.ReLU, a.pool.Forward,
		a.conv2.Forward, autograd.ReLU, a.pool.Forward,
		a.conv3.Forward, autograd.ReLU,
		a.conv4.Forward, autograd.ReLU,
		a.conv5.Forward, autograd.ReLU, a.pool.Forward,
		(nn.Flatten{}).Forward,
		a.drop1.Forward, a.fc1.Forward, autograd.ReLU,
		a.drop2.Forward, a.fc2.Forward, autograd.ReLU,
		a.fc3.Forward,
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
	fmt.Printf("training AlexNet on %v\n", dev)
	rng := rand.New(rand.NewPCG(9, 99))

	net := newAlexNet(dev)
	net.SetTraining(true)

	total := 0
	for _, p := range net.Parameters() {
		total += p.NumElements()
	}
	fmt.Printf("parameters: %d (≈ %.1fM)\n", total, float64(total)/1e6)

	opt, err := training.NewAdamW(net.Parameters(), training.AdamWOptions{LR: 0.0005})
	if err != nil {
		log.Fatal(err)
	}

	for step := range steps {
		x, y := makeBatch(rng, dev)
		opt.ZeroGrad()
		logits, err := net.Forward(x)
		if err != nil {
			log.Fatalf("forward: %v", err)
		}
		loss, err := autograd.CrossEntropyLoss(logits, y)
		if err != nil {
			log.Fatalf("loss: %v", err)
		}
		if err := autograd.Backward(loss, nil); err != nil {
			log.Fatalf("backward: %v", err)
		}
		if err := opt.Step(); err != nil {
			log.Fatalf("step: %v", err)
		}
		if step%10 == 0 || step == steps-1 {
			lossCPU, _ := loss.To(device.CPU())
			logitsCPU, _ := logits.To(device.CPU())
			labelsCPU, _ := y.To(device.CPU())
			fmt.Printf("step %4d  loss = %.4f  batch-acc = %.2f%%\n",
				step, tensor.MustSlice[float32](lossCPU)[0],
				batchAcc(logitsCPU, labelsCPU)*100)
		}
	}

	// Evaluate with Dropout off.
	net.SetTraining(false)
	fmt.Println("\n--- evaluation (Dropout off, 4 fresh batches) ---")
	correct, total2 := 0, 0
	for range 4 {
		x, y := makeBatch(rng, dev)
		logits, _ := net.Forward(x)
		logitsCPU, _ := logits.To(device.CPU())
		labelsCPU, _ := y.To(device.CPU())
		correct += int(batchAcc(logitsCPU, labelsCPU) * float32(batch))
		total2 += batch
	}
	fmt.Printf("eval accuracy: %.1f%% (%d/%d)\n", float32(correct)/float32(total2)*100, correct, total2)
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
