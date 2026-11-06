// Train a 10-class ResNet-50 on CIFAR-10 (32×32 → upsampled 7× to 224×224).
//
// This is a smoke test, not a convergence run — it trains on a 1024-image
// subset for ~50 AdamW steps so the whole thing finishes in <1 min on a
// modern GPU. To reproduce the full ImageNet recipe you'd swap CIFAR-10
// for ImageNet, drop the upsample, and train for 90+ epochs.
//
//	go run            ./examples/26_resnet50/train
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
	cacheDir := flag.String("cache", "./cifar10-cache", "CIFAR-10 cache dir")
	flag.Parse()

	dev := pickDevice(*deviceFlag)
	fmt.Printf("device: %v\n", dev)

	t0 := time.Now()
	cifar, err := data.LoadCIFAR10(*cacheDir, device.CPU())
	if err != nil {
		log.Fatalf("load CIFAR-10: %v", err)
	}
	fmt.Printf("CIFAR-10 loaded in %v\n", time.Since(t0).Round(time.Millisecond))

	// Use a small subset for the smoke test (faster to copy to GPU).
	subset := 1024
	xCPU, _ := cifar.TrainImages.Narrow(0, 0, subset)
	xCPU, _ = xCPU.Contiguous()
	xAll, err := xCPU.To(dev)
	if err != nil {
		log.Fatal(err)
	}
	// Labels stay on CPU as a plain []int64 — we don't have an I64 IndexSelect
	// kernel on CUDA, so per-batch we just construct a fresh GPU label tensor.
	yCPU, _ := cifar.TrainLabels.Narrow(0, 0, subset)
	yCPU, _ = yCPU.Contiguous()
	allLabels := tensor.MustSlice[int64](yCPU)

	// Model + optimiser.
	model, err := models.NewResNet50(10, dev)
	if err != nil {
		log.Fatal(err)
	}
	model.SetTraining(true)
	fmt.Printf("ResNet-50 with 10-class head — %.2fM params\n", float64(model.ParamCount())/1e6)

	opt, err := training.NewAdamW(model.Parameters(), training.AdamWOptions{
		LR: 1e-3, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, WeightDecay: 1e-4,
	})
	if err != nil {
		log.Fatal(err)
	}

	for step := range *steps {
		t1 := time.Now()
		// Random batch indices.
		idx := make([]int64, *batch)
		for i := range idx {
			idx[i] = int64((step**batch + i) % subset)
		}
		idxT, _ := tensor.New(idx, []int{*batch}, dev)
		xb, err := ops.IndexSelect(xAll, idxT)
		if err != nil {
			log.Fatal(err)
		}
		// Build the batch labels directly on dev — no IndexSelect on I64.
		batchLabels := make([]int64, *batch)
		for i, j := range idx {
			batchLabels[i] = allLabels[j]
		}
		yb, err := tensor.New(batchLabels, []int{*batch}, dev)
		if err != nil {
			log.Fatal(err)
		}
		// Upsample 32 → 224 (nearest-neighbour 7×).
		xUp, err := ops.UpsampleNearest(xb, [2]int{7, 7})
		if err != nil {
			log.Fatal(err)
		}

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
		if err := opt.Step(); err != nil {
			log.Fatal(err)
		}

		if step%5 == 0 || step == *steps-1 {
			lossCPU, _ := loss.To(device.CPU())
			fmt.Printf("step %3d  loss=%.4f  (%v)\n",
				step, tensor.MustSlice[float32](lossCPU)[0], time.Since(t1).Round(time.Millisecond))
		}
		runtime.GC()
	}
	fmt.Printf("\ndone in %v — model is in memory, no checkpoint saved.\n", time.Since(t0).Round(time.Millisecond))
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
