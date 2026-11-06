// Train VGG-11 (10-class head) on CIFAR-10 — smoke test.
// VGG-16 fits in 8GB at batch=2 but is wasteful on a smoke run; default
// is VGG-11 with batch=4 to keep memory + time reasonable.
//
//	go run            ./examples/28_vgg16/train
package main

import (
	"flag"
	"fmt"
	"log"
	"runtime"
	"time"

	"light/autograd"
	"light/data"
	"light/device"
	"light/models"
	"light/ops"
	"light/tensor"
	"light/training"
)

func main() {
	deviceFlag := flag.String("device", "gpu", "gpu or cpu")
	steps := flag.Int("steps", 30, "training steps")
	batch := flag.Int("batch", 4, "batch size (small — VGG is memory-heavy)")
	depth := flag.Int("depth", 11, "VGG depth: 11, 13, 16, 19")
	cacheDir := flag.String("cache", "./cifar10-cache", "CIFAR-10 cache dir")
	flag.Parse()

	dev := pickDevice(*deviceFlag)
	fmt.Printf("device: %v\n", dev)

	cifar, err := data.LoadCIFAR10(*cacheDir, device.CPU())
	if err != nil {
		log.Fatal(err)
	}
	subset := 256
	xCPU, _ := cifar.TrainImages.Narrow(0, 0, subset)
	xCPU, _ = xCPU.Contiguous()
	xAll, _ := xCPU.To(dev)
	yCPU, _ := cifar.TrainLabels.Narrow(0, 0, subset)
	yCPU, _ = yCPU.Contiguous()
	allLabels := tensor.MustSlice[int64](yCPU)

	var (
		model interface {
			Forward(*tensor.Tensor) (*tensor.Tensor, error)
			Parameters() []*tensor.Tensor
			ParamCount() int
			SetTraining(bool)
		}
	)
	switch *depth {
	case 11:
		model, err = models.NewVGG11(10, dev)
	case 13:
		model, err = models.NewVGG13(10, dev)
	case 16:
		model, err = models.NewVGG16(10, dev)
	case 19:
		model, err = models.NewVGG19(10, dev)
	default:
		log.Fatalf("unsupported depth %d", *depth)
	}
	if err != nil {
		log.Fatal(err)
	}
	model.SetTraining(true)
	fmt.Printf("VGG-%d (10-class) — %.2fM params\n", *depth, float64(model.ParamCount())/1e6)

	opt, _ := training.NewAdamW(model.Parameters(), training.AdamWOptions{LR: 1e-4, WeightDecay: 1e-4})

	for step := range *steps {
		t1 := time.Now()
		idx := make([]int64, *batch)
		for i := range idx {
			idx[i] = int64((step**batch + i) % subset)
		}
		idxT, _ := tensor.New(idx, []int{*batch}, dev)
		xb, _ := ops.IndexSelect(xAll, idxT)
		batchLabels := make([]int64, *batch)
		for i, j := range idx {
			batchLabels[i] = allLabels[j]
		}
		yb, _ := tensor.New(batchLabels, []int{*batch}, dev)
		xUp, _ := ops.UpsampleNearest(xb, [2]int{7, 7})

		opt.ZeroGrad()
		logits, err := model.Forward(xUp)
		if err != nil {
			log.Fatal(err)
		}
		loss, err := autograd.CrossEntropyLoss(logits, yb)
		if err != nil {
			log.Fatal(err)
		}
		if err := autograd.Backward(loss, nil); err != nil {
			log.Fatal(err)
		}
		_ = opt.Step()

		if step%5 == 0 || step == *steps-1 {
			lc, _ := loss.To(device.CPU())
			fmt.Printf("step %3d  loss=%.4f  (%v)\n", step,
				tensor.MustSlice[float32](lc)[0], time.Since(t1).Round(time.Millisecond))
		}
		runtime.GC()
	}
}

func pickDevice(name string) device.Device {
	if name == "gpu" && device.CUDAAvailable() {
		runtime.LockOSThread()
		g, err := device.CUDA(0)
		if err == nil {
			return g
		}
		fmt.Printf("CUDA unavailable (%v); falling back to CPU\n", err)
	}
	return device.CPU()
}
