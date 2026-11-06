// Inference for MobileNetV2 on CIFAR-10 test samples.
//
//	go run            ./examples/27_mobilenetv2/infer
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
	n := flag.Int("n", 8, "number of test samples")
	width := flag.Float64("width", 0.5, "MobileNetV2 width multiplier")
	cacheDir := flag.String("cache", "./cifar10-cache", "CIFAR-10 cache dir")
	flag.Parse()

	dev := pickDevice(*deviceFlag)
	fmt.Printf("device: %v\n", dev)

	imgs, lbls, err := data.CIFAR10TestSamples(*cacheDir, *n, dev)
	if err != nil {
		log.Fatal(err)
	}
	model, err := models.NewMobileNetV2WithMult(10, *width, dev)
	if err != nil {
		log.Fatal(err)
	}
	model.SetTraining(false)
	fmt.Printf("MobileNetV2 (×%.2f, random init) — %.2fM params\n", *width, float64(model.ParamCount())/1e6)

	t0 := time.Now()
	imgsUp, _ := ops.UpsampleNearest(imgs, [2]int{7, 7})
	logits, err := model.Forward(imgsUp)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("forward (%v) → %v in %v\n",
		imgsUp.Shape(), logits.Shape(), time.Since(t0).Round(time.Millisecond))

	logitsCPU, _ := logits.To(device.CPU())
	flat := tensor.MustSlice[float32](logitsCPU)
	lblsCPU, _ := lbls.To(device.CPU())
	gold := tensor.MustSlice[int64](lblsCPU)

	V := 10
	correct := 0
	fmt.Println("\nsample | predicted        | actual           | top-3 logits")
	fmt.Println("-------|------------------|------------------|---------------------------")
	for i := range *n {
		row := flat[i*V : (i+1)*V]
		idx := topK(row, 3)
		if idx[0] == int(gold[i]) {
			correct++
		}
		fmt.Printf("  %2d   | %-16s | %-16s | (%d)%.2f  (%d)%.2f  (%d)%.2f\n",
			i, data.CIFAR10ClassNames[idx[0]], data.CIFAR10ClassNames[gold[i]],
			idx[0], row[idx[0]], idx[1], row[idx[1]], idx[2], row[idx[2]])
	}
	fmt.Printf("\naccuracy: %d / %d  (random init ≈ 10%% expected)\n", correct, *n)
	runtime.GC()
}

func topK(row []float32, k int) []int {
	type lp struct {
		i int
		v float32
	}
	arr := make([]lp, len(row))
	for i, v := range row {
		arr[i] = lp{i, v}
	}
	sort.Slice(arr, func(a, b int) bool { return arr[a].v > arr[b].v })
	out := make([]int, k)
	for i := range k {
		out[i] = arr[i].i
	}
	return out
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
