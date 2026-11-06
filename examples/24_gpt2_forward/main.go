// Example 24 — GPT-2 forward pass on CPU or GPU.
//
// Runs a random-initialised GPT-2 through a few token sequences and prints
// the shape + argmax-per-position of the logits. With trained weights, the
// same code would produce next-token predictions for autoregressive text
// generation.
//
// Usage:
//
//	go run            ./examples/24_gpt2_forward -size tiny
//	go run            ./examples/24_gpt2_forward -size small
//
// Sizes:
//
//	tiny   — 2-layer, 64-dim, vocab 256 — smoke test
//	small  — GPT-2 gpt2 (124M params) — real architecture, random weights
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"runtime"
	"time"

	"light/autograd"
	"light/device"
	"light/models"
	"light/tensor"
)

func main() {
	deviceFlag := flag.String("device", "gpu", "gpu or cpu (default gpu)")
	sizeFlag := flag.String("size", "tiny", "tiny|small|medium")
	batch := flag.Int("batch", 2, "batch size")
	seqLen := flag.Int("seq", 16, "sequence length")
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

	var cfg models.GPT2Config
	switch *sizeFlag {
	case "tiny":
		cfg = models.GPT2Tiny()
	case "small":
		cfg = models.GPT2Small()
	case "medium":
		cfg = models.GPT2Medium()
	default:
		log.Fatalf("unknown -size %q", *sizeFlag)
	}
	fmt.Printf("device: %v  size: %s  config: vocab=%d npos=%d d=%d layers=%d heads=%d\n",
		dev, *sizeFlag, cfg.VocabSize, cfg.NPositions, cfg.NEmbd, cfg.NLayer, cfg.NHead)

	t0 := time.Now()
	gpt, err := models.NewGPT2(cfg, dev)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("constructed in %v  — %.2fM parameters\n",
		time.Since(t0).Round(time.Millisecond), float64(gpt.ParamCount())/1e6)

	// Random token sequences (in [0, VocabSize)).
	rng := rand.New(rand.NewPCG(4, 8))
	toks := make([]int64, *batch**seqLen)
	for i := range toks {
		toks[i] = int64(rng.IntN(cfg.VocabSize))
	}
	x, err := tensor.New(toks, []int{*batch, *seqLen}, dev)
	if err != nil {
		log.Fatal(err)
	}

	t1 := time.Now()
	logits, err := gpt.Forward(x)
	if err != nil {
		log.Fatalf("forward: %v", err)
	}
	runtime.GC()
	fmt.Printf("forward [%d × %d] → %v in %v\n",
		*batch, *seqLen, logits.Shape(), time.Since(t1).Round(time.Millisecond))

	// Argmax next-token prediction per position.
	logitsCPU, _ := logits.To(device.CPU())
	flat := tensor.MustSlice[float32](logitsCPU)
	B := *batch
	T := *seqLen
	V := cfg.VocabSize
	fmt.Printf("\nsample argmax predictions (batch 0):\n  input : %v\n  pred  : [", toks[:T])
	for t := range T {
		best := 0
		for v := 1; v < V; v++ {
			if flat[t*V+v] > flat[t*V+best] {
				best = v
			}
		}
		if t > 0 {
			fmt.Print(" ")
		}
		fmt.Printf("%d", best)
	}
	fmt.Println("]")

	_ = autograd.Sum // keep import warm in case the example expands to training
	_ = B            // unused beyond argmax loop
}
