// Train MobileNetV2 (10-class head) on CIFAR-10 — smoke test.
//
//	go run            ./examples/27_mobilenetv2/train
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
	steps := flag.Int("steps", 50, "training steps")
	batch := flag.Int("batch", 8, "batch size")
	width := flag.Float64("width", 0.5, "MobileNetV2 width multiplier")
	cacheDir := flag.String("cache", "./cifar10-cache", "CIFAR-10 cache dir")
	flag.Parse()

	dev := pickDevice(*deviceFlag)
	fmt.Printf("device: %v\n", dev)

	t0 := time.Now()
	cifar, err := data.LoadCIFAR10(*cacheDir, device.CPU())
	if err != nil {
		log.Fatal(err)
	}
	xCPU, _ := cifar.TrainImages.Narrow(0, 0, 1024)
	xCPU, _ = xCPU.Contiguous()
	xAll, _ := xCPU.To(dev)
	yCPU, _ := cifar.TrainLabels.Narrow(0, 0, 1024)
	yCPU, _ = yCPU.Contiguous()
	allLabels := tensor.MustSlice[int64](yCPU)

	model, err := models.NewMobileNetV2WithMult(10, *width, dev)
	if err != nil {
		log.Fatal(err)
	}
	model.SetTraining(true)
	fmt.Printf("MobileNetV2 (×%.2f, 10-class) — %.2fM params (loaded in %v)\n",
		*width, float64(model.ParamCount())/1e6, time.Since(t0).Round(time.Millisecond))

	opt, _ := training.NewAdamW(model.Parameters(), training.AdamWOptions{LR: 1e-3, WeightDecay: 1e-4})

	for step := range *steps {
		t1 := time.Now()
		idx := make([]int64, *batch)
		for i := range idx {
			idx[i] = int64((step**batch + i) % 1024)
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
