// Example 34 — ResNet9 on CIFAR-10, target 90%+ test accuracy.
//
// Recipe (matches the kmldas Kaggle notebook within the parts our framework
// supports today):
//
//	architecture       : ResNet9 (~6.5M params)
//	pre-processing     : per-channel normalize with CIFAR-10 train stats
//	augmentation       : RandomCrop(32, padding=4) + RandomHorizontalFlip
//	optimiser          : AdamW (lr=1e-3, wd=1e-4)
//	schedule           : CosineLR with 1-epoch linear warmup
//	batch size         : 400
//	default epochs     : 8
//
//	go run            ./examples/34_resnet9_cifar10 -epochs 8
//	go run            ./examples/34_resnet9_cifar10 -epochs 12 -lr 0.002
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
	"light/models"
	"light/tensor"
	"light/training"
)

const classes = 10

func main() {
	deviceFlag := flag.String("device", "gpu", "cpu or gpu")
	epochs := flag.Int("epochs", 8, "number of epochs")
	batch := flag.Int("batch", 400, "batch size")
	lrFlag := flag.Float64("lr", 1e-3, "peak learning rate (cosine)")
	wd := flag.Float64("wd", 1e-4, "weight decay")
	flag.Parse()

	dev := pickDevice(*deviceFlag)
	fmt.Printf("device: %v\n", dev)

	// ─── Load + normalise the CIFAR-10 split (CPU-side once at startup).
	cifar, err := data.LoadCIFAR10("./cifar10-cache", device.CPU())
	if err != nil {
		log.Fatalf("load CIFAR-10: %v", err)
	}
	trainN := cifar.TrainImages.Dim(0)
	testN := cifar.TestImages.Dim(0)
	trainImgs := tensor.MustSlice[float32](cifar.TrainImages)
	trainLbls := tensor.MustSlice[int64](cifar.TrainLabels)
	testImgs := tensor.MustSlice[float32](cifar.TestImages)
	testLbls := tensor.MustSlice[int64](cifar.TestLabels)
	data.NormalizeCIFAR10(trainImgs, trainN, 32, 32)
	data.NormalizeCIFAR10(testImgs, testN, 32, 32)
	fmt.Printf("CIFAR-10: %d train, %d test  (per-channel normalised)\n", trainN, testN)

	// ─── Model + optimiser.
	net, err := models.NewResNet9(classes, dev)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("ResNet9: %.2fM params\n", float64(net.ParamCount())/1e6)

	opt, err := training.NewAdamW(net.Parameters(), training.AdamWOptions{
		LR:          float32(*lrFlag),
		WeightDecay: float32(*wd),
	})
	if err != nil {
		log.Fatal(err)
	}
	stepsPerEpoch := trainN / *batch
	totalSteps := stepsPerEpoch * *epochs
	sched := training.NewCosineLR(opt, float32(*lrFlag), 1e-5, stepsPerEpoch /* 1 epoch warmup */, totalSteps)

	const imgPx = 3 * 32 * 32
	rng := rand.New(rand.NewPCG(11, 13))
	start := time.Now()

	for epoch := 1; epoch <= *epochs; epoch++ {
		net.SetTraining(true)
		perm := make([]int, trainN)
		for i := range perm {
			perm[i] = i
		}
		rng.Shuffle(trainN, func(i, j int) { perm[i], perm[j] = perm[j], perm[i] })

		var lastLoss float32
		for step := 0; step < stepsPerEpoch; step++ {
			imgs := make([]float32, *batch*imgPx)
			lbls := make([]int64, *batch)
			for b := range *batch {
				idx := perm[step**batch+b]
				copy(imgs[b*imgPx:(b+1)*imgPx], trainImgs[idx*imgPx:(idx+1)*imgPx])
				lbls[b] = trainLbls[idx]
			}
			// Augment on the CPU before transfer to GPU.
			data.RandomCropPad(imgs, *batch, 32, 32, 4, rng)
			data.RandomHorizontalFlip(imgs, *batch, 32, 32, 0.5, rng)

			x, _ := tensor.New(imgs, []int{*batch, 3, 32, 32}, dev)
			y, _ := tensor.New(lbls, []int{*batch}, dev)

			opt.ZeroGrad()
			logits, err := net.Forward(x)
			if err != nil {
				log.Fatalf("step %d fwd: %v", step, err)
			}
			loss, err := autograd.CrossEntropyLoss(logits, y)
			if err != nil {
				log.Fatalf("step %d loss: %v", step, err)
			}
			if err := autograd.Backward(loss, nil); err != nil {
				log.Fatalf("step %d bwd: %v", step, err)
			}
			if err := opt.Step(); err != nil {
				log.Fatalf("step %d opt: %v", step, err)
			}
			sched.Step()

			if step%25 == 0 || step == stepsPerEpoch-1 {
				lossCPU, _ := loss.To(device.CPU())
				lastLoss = tensor.MustSlice[float32](lossCPU)[0]
				fmt.Printf("ep %d  step %3d/%d  loss=%.4f  lr=%.5f\n",
					epoch, step, stepsPerEpoch, lastLoss, sched.LR())
			}
			runtime.GC()
		}
		acc := evalAccuracy(net, testImgs, testLbls, testN, *batch, dev)
		fmt.Printf("epoch %d done in %v  loss %.4f  test-acc = %.2f%%\n\n",
			epoch, time.Since(start).Round(time.Millisecond), lastLoss, acc*100)
	}
}

func evalAccuracy(net *models.ResNet9, xs []float32, ys []int64, n, batch int, dev device.Device) float32 {
	net.SetTraining(false)
	const imgPx = 3 * 32 * 32
	correct := 0
	for off := 0; off+batch <= n; off += batch {
		imgs := make([]float32, batch*imgPx)
		lbls := make([]int64, batch)
		for b := range batch {
			copy(imgs[b*imgPx:(b+1)*imgPx], xs[(off+b)*imgPx:(off+b+1)*imgPx])
			lbls[b] = ys[off+b]
		}
		x, _ := tensor.New(imgs, []int{batch, 3, 32, 32}, dev)
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
		if off%2048 == 0 {
			runtime.GC()
		}
	}
	total := (n / batch) * batch
	return float32(correct) / float32(total)
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
