// Inference for ResNet-50 on CIFAR-10 test samples.
//
// Loads N samples from the CIFAR-10 test set, upsamples 32×32 → 224×224,
// runs forward through a (random-init) ResNet-50, and prints
// predicted-class vs. ground-truth-class for each sample. With random
// weights the accuracy is ~10% (chance); train with the sibling
// `examples/26_resnet50/train` and load a checkpoint to do better.
//
//	go run            ./examples/26_resnet50/infer
package main

import (
	"flag"
	"fmt"
	"log"
	"runtime"
	"sort"
	"time"

	"light/data"
	"light/device"
	"light/models"
	"light/ops"
	"light/tensor"
)

func main() {
	deviceFlag := flag.String("device", "gpu", "gpu or cpu")
	n := flag.Int("n", 8, "number of test samples to classify")
	cacheDir := flag.String("cache", "./cifar10-cache", "CIFAR-10 cache dir")
	flag.Parse()

	dev := pickDevice(*deviceFlag)
	fmt.Printf("device: %v\n", dev)

	t0 := time.Now()
	imgs, lbls, err := data.CIFAR10TestSamples(*cacheDir, *n, dev)
	if err != nil {
		log.Fatalf("load samples: %v", err)
	}
	fmt.Printf("loaded %d test samples in %v\n", *n, time.Since(t0).Round(time.Millisecond))

	model, err := models.NewResNet50(10, dev)
	if err != nil {
		log.Fatal(err)
	}
	model.SetTraining(false) // eval mode — uses BN running stats
	fmt.Printf("ResNet-50 (random init) — %.2fM params\n", float64(model.ParamCount())/1e6)

	// Upsample 32→224 for the ImageNet-shape ResNet.
	t1 := time.Now()
	imgsUp, err := ops.UpsampleNearest(imgs, [2]int{7, 7})
	if err != nil {
		log.Fatal(err)
	}
	logits, err := model.Forward(imgsUp)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("forward (%v) → %v in %v\n",
		imgsUp.Shape(), logits.Shape(), time.Since(t1).Round(time.Millisecond))

	// Argmax per sample.
	logitsCPU, _ := logits.To(device.CPU())
	flat := tensor.MustSlice[float32](logitsCPU)
	lblsCPU, _ := lbls.To(device.CPU())
	gold := tensor.MustSlice[int64](lblsCPU)

	V := 10
	correct := 0
	fmt.Println("\nsample | predicted        | actual           | top-3 logits")
	fmt.Println("-------|------------------|------------------|----------------------------------------")
	for i := range *n {
		row := flat[i*V : (i+1)*V]
		type lp struct {
			idx int
			val float32
		}
		all := make([]lp, V)
		for j, v := range row {
			all[j] = lp{j, v}
		}
		sort.Slice(all, func(a, b int) bool { return all[a].val > all[b].val })
		pred := all[0].idx
		if pred == int(gold[i]) {
			correct++
		}
		fmt.Printf("  %2d   | %-16s | %-16s | (%d)%.2f  (%d)%.2f  (%d)%.2f\n",
			i, data.CIFAR10ClassNames[pred], data.CIFAR10ClassNames[gold[i]],
			all[0].idx, all[0].val, all[1].idx, all[1].val, all[2].idx, all[2].val)
	}
	fmt.Printf("\naccuracy: %d / %d = %.1f%%   (random init ≈ 10%% expected)\n",
		correct, *n, 100*float64(correct)/float64(*n))
	runtime.GC()
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
