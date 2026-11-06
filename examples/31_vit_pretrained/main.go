// Example 31 — ViT-B/16 inference with pretrained HuggingFace weights.
//
// Downloads `google/vit-base-patch16-224` (or another HF ViT repo) on first
// run, loads safetensors weights into our `models.ViT`, runs a deterministic
// synthetic input, and prints the top-5 predicted ImageNet class indices.
// Use this to compare logits against PyTorch's
// `transformers.ViTForImageClassification`.
//
//	go run            ./examples/31_vit_pretrained
//	go run            ./examples/31_vit_pretrained -device cpu
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"runtime"
	"sort"
	"time"

	"light/device"
	"light/models"
	"light/tensor"
)

func main() {
	deviceFlag := flag.String("device", "gpu", "gpu or cpu (default gpu)")
	repo := flag.String("repo", "google/vit-base-patch16-224", "HuggingFace ViT repo")
	seed := flag.Uint64("seed", 42, "RNG seed for the synthetic image (so PyTorch can match it)")
	flag.Parse()

	dev := pickDevice(*deviceFlag)
	fmt.Printf("device: %v\n", dev)
	fmt.Printf("repo:   %s\n", *repo)

	t0 := time.Now()
	vit, err := models.LoadViTFromHF(*repo, dev)
	if err != nil {
		log.Fatalf("load: %v", err)
	}
	fmt.Printf("loaded in %v — %.2fM params, ImgSize=%d Patch=%d Layers=%d D=%d Heads=%d\n",
		time.Since(t0).Round(time.Millisecond), float64(vit.ParamCount())/1e6,
		vit.Cfg.ImgSize, vit.Cfg.PatchSize, vit.Cfg.NLayer, vit.Cfg.NEmbd, vit.Cfg.NHead)

	// Deterministic synthetic image, normalised to [-1, 1] (the standard ViT
	// preprocessor scales pixel/255 → /2−1).
	rng := rand.New(rand.NewPCG(*seed, *seed^0xdeadbeef))
	imgSize := vit.Cfg.ImgSize
	pixels := make([]float32, 3*imgSize*imgSize)
	for i := range pixels {
		pixels[i] = rng.Float32()*2 - 1
	}
	x, err := tensor.New(pixels, []int{1, 3, imgSize, imgSize}, dev)
	if err != nil {
		log.Fatal(err)
	}

	t1 := time.Now()
	logits, err := vit.Forward(x)
	if err != nil {
		log.Fatalf("forward: %v", err)
	}
	runtime.GC()
	fmt.Printf("forward (1, 3, %d, %d) → %v in %v\n",
		imgSize, imgSize, logits.Shape(), time.Since(t1).Round(time.Millisecond))

	// Top-5 class indices.
	logitsCPU, _ := logits.To(device.CPU())
	flat := tensor.MustSlice[float32](logitsCPU)
	type lp struct {
		idx int
		val float32
	}
	all := make([]lp, len(flat))
	for i, v := range flat {
		all[i] = lp{i, v}
	}
	sort.Slice(all, func(i, j int) bool { return all[i].val > all[j].val })
	fmt.Printf("\ntop-5 (idx, logit):\n")
	for i := range 5 {
		fmt.Printf("  %d  %.4f\n", all[i].idx, all[i].val)
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
