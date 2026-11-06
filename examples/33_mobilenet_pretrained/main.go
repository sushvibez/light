// Example 33 — MobileNetV2 inference with pretrained HuggingFace weights.
//
// Downloads `google/mobilenet_v2_1.0_224` (or another HF MobileNetV2 repo)
// on first run, copies the safetensors weights into our `*MobileNetV2`,
// and classifies a few CIFAR-10 test samples (upsampled 32→224). The
// predictions are ImageNet-1k indices and won't match CIFAR-10 classes —
// the point is to validate the loader against
// `transformers.MobileNetV2ForImageClassification`.
//
//	go run ./examples/33_mobilenet_pretrained
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
	repo := flag.String("repo", "google/mobilenet_v2_1.0_224", "HuggingFace MobileNetV2 repo")
	n := flag.Int("n", 4, "number of CIFAR-10 test samples to classify")
	cacheDir := flag.String("cache", "./cifar10-cache", "CIFAR-10 cache dir")
	flag.Parse()

	dev := pickDevice(*deviceFlag)
	fmt.Printf("device: %v\nrepo:   %s\n", dev, *repo)

	t0 := time.Now()
	model, err := models.LoadMobileNetV2FromHF(*repo, dev)
	if err != nil {
		log.Fatalf("load model: %v", err)
	}
	model.SetTraining(false)
	fmt.Printf("loaded in %v — %.2fM params\n",
		time.Since(t0).Round(time.Millisecond), float64(model.ParamCount())/1e6)

	imgs, lbls, err := data.CIFAR10TestSamples(*cacheDir, *n, dev)
	if err != nil {
		log.Fatal(err)
	}
	imgsUp, _ := ops.UpsampleNearest(imgs, [2]int{7, 7})

	t1 := time.Now()
	logits, err := model.Forward(imgsUp)
	if err != nil {
		log.Fatalf("forward: %v", err)
	}
	runtime.GC()
	fmt.Printf("forward (%v) → %v in %v\n",
		imgsUp.Shape(), logits.Shape(), time.Since(t1).Round(time.Millisecond))

	logitsCPU, _ := logits.To(device.CPU())
	flat := tensor.MustSlice[float32](logitsCPU)
	V := logits.Dim(1)
	lblsCPU, _ := lbls.To(device.CPU())
	gold := tensor.MustSlice[int64](lblsCPU)

	fmt.Printf("\nsample (CIFAR class) | top-5 ImageNet class indices\n")
	fmt.Println("---------------------|------------------------------")
	for i := range *n {
		row := flat[i*V : (i+1)*V]
		idx := topK(row, 5)
		fmt.Printf("  %d (%-10s) | (%d)%.2f  (%d)%.2f  (%d)%.2f  (%d)%.2f  (%d)%.2f\n",
			gold[i], data.CIFAR10ClassNames[gold[i]],
			idx[0], row[idx[0]], idx[1], row[idx[1]], idx[2], row[idx[2]],
			idx[3], row[idx[3]], idx[4], row[idx[4]])
	}
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
