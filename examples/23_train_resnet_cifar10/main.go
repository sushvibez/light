// Example 23 — train ResNet-20 on CIFAR-10 with a cosine LR schedule.
//
// First run downloads the CIFAR-10 binary tarball (163 MB) into
// ./cifar10-cache/ and extracts it. Subsequent runs load from disk.
//
// Usage:
//
//	go run            ./examples/23_train_resnet_cifar10 -epochs 5
//	go run            ./examples/23_train_resnet_cifar10 -device cpu -epochs 1
//
// ResNet-20 (He et al. 2015 §4.2): 6n+2 layers with n=3 ⇒ 20 layers.
// 0.27M parameters. Expected ≥78% test accuracy after 5 epochs on GPU.
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

const (
	batch   = 128
	classes = 10
)

func main() {
	deviceFlag := flag.String("device", "gpu", "cpu or gpu")
	epochs := flag.Int("epochs", 5, "number of epochs")
	lrFlag := flag.Float64("lr", 0.05, "peak learning rate (cosine)")
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

	cifar, err := data.LoadCIFAR10("./cifar10-cache", device.CPU())
	if err != nil {
		log.Fatalf("load CIFAR-10: %v", err)
	}
	trainN := cifar.TrainImages.Dim(0)
	testN := cifar.TestImages.Dim(0)
	fmt.Printf("CIFAR-10: %d train, %d test\n", trainN, testN)

	net, err := models.NewResNetCIFAR(3 /* n=3 ⇒ ResNet-20 */, classes, dev)
	if err != nil {
		log.Fatal(err)
	}
	total := 0
	for _, p := range net.Parameters() {
		total += p.NumElements()
	}
	fmt.Printf("ResNet-20: %d params (≈ %.2fK)\n", total, float64(total)/1e3)

	opt, err := training.NewAdamW(net.Parameters(), training.AdamWOptions{
		LR:          float32(*lrFlag),
		WeightDecay: 5e-4,
	})
	if err != nil {
		log.Fatal(err)
	}
	stepsPerEpoch := trainN / batch
	totalSteps := stepsPerEpoch * *epochs
	sched := training.NewCosineLR(opt, float32(*lrFlag), 1e-5, stepsPerEpoch /* 1 epoch warmup */, totalSteps)

	trainImgs := tensor.MustSlice[float32](cifar.TrainImages)
	trainLbls := tensor.MustSlice[int64](cifar.TrainLabels)
	testImgs := tensor.MustSlice[float32](cifar.TestImages)
	testLbls := tensor.MustSlice[int64](cifar.TestLabels)
	const imgPx = 3 * 32 * 32

	rng := rand.New(rand.NewPCG(11, 13))
	start := time.Now()
	globalStep := 0

	for epoch := 1; epoch <= *epochs; epoch++ {
		net.SetTraining(true)
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
			x, _ := tensor.New(imgs, []int{batch, 3, 32, 32}, dev)
			y, _ := tensor.New(lbls, []int{batch}, dev)

			opt.ZeroGrad()
			logits, err := net.Forward(x)
			if err != nil {
				log.Fatalf("step %d fwd: %v", globalStep, err)
			}
			loss, err := autograd.CrossEntropyLoss(logits, y)
			if err != nil {
				log.Fatalf("step %d loss: %v", globalStep, err)
			}
			if err := autograd.Backward(loss, nil); err != nil {
				log.Fatalf("step %d bwd: %v", globalStep, err)
			}
			if err := opt.Step(); err != nil {
				log.Fatalf("step %d opt: %v", globalStep, err)
			}
			sched.Step()
			globalStep++

			if step%50 == 0 || step == stepsPerEpoch-1 {
				lossCPU, _ := loss.To(device.CPU())
				lastLoss = tensor.MustSlice[float32](lossCPU)[0]
				fmt.Printf("ep %d step %4d/%d  loss = %.4f  lr = %.5f\n",
					epoch, step, stepsPerEpoch, lastLoss, sched.LR())
			}
			// Keep GPU memory bounded — ResNet creates many BN intermediates
			// per step, so we force finalizers every iteration.
			runtime.GC()
		}
		acc := evalAccuracy(net, testImgs, testLbls, testN, dev)
		fmt.Printf("epoch %d done in %v  final loss %.4f  test-acc = %.2f%%\n",
			epoch, time.Since(start).Round(time.Millisecond), lastLoss, acc*100)
	}
}

func evalAccuracy(net *models.ResNetCIFAR, xs []float32, ys []int64, n int, dev device.Device) float32 {
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
