// Example 25 — GPT-2 inference with pretrained HuggingFace weights.
//
// Downloads `gpt2` (or another variant) from the HF Hub on first run, loads
// the safetensors checkpoint into our `models.GPT2`, tokenizes a prompt with
// the byte-level BPE, and runs greedy generation for a few tokens. Use this
// to compare logits / completions against the reference PyTorch run.
//
// Usage:
//
//	go run            ./examples/25_gpt2_pretrained
//	go run            ./examples/25_gpt2_pretrained
//	go run            ./examples/25_gpt2_pretrained -prompt "The capital of France is" -tokens 12
//	go run ./examples/25_gpt2_pretrained -device webgpu -prompt "The capital of France is" -tokens 8
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"time"

	"light/device"
	"light/hub"
	"light/models"
	"light/tensor"
	"light/tokenizer"
)

func main() {
	deviceFlag := flag.String("device", "gpu", "gpu (CUDA), webgpu, or cpu (default gpu)")
	repo := flag.String("repo", "gpt2", "HuggingFace repo (gpt2, gpt2-medium, ...)")
	prompt := flag.String("prompt", "The quick brown fox jumps over the lazy", "prompt text")
	nGen := flag.Int("tokens", 8, "number of tokens to generate (greedy)")
	flag.Parse()

	var dev device.Device
	switch *deviceFlag {
	case "gpu", "cuda":
		runtime.LockOSThread()
		g, err := device.CUDA(0)
		if err != nil {
			log.Fatalf("CUDA unavailable: %v", err)
		}
		dev = g
	case "webgpu":
		g, err := device.WebGPU(0)
		if err != nil {
			log.Fatalf("WebGPU unavailable (scripts/fetch-wgpu.sh fetches wgpu-native): %v", err)
		}
		dev = g
	case "cpu":
		dev = device.CPU()
	default:
		log.Fatalf("unknown -device %q (want gpu, webgpu or cpu)", *deviceFlag)
	}
	fmt.Printf("device:  %v\n", dev)
	fmt.Printf("repo:    %s\n", *repo)

	// Tokenizer — fetch tokenizer.json from the same repo.
	tokPath, err := hub.Download(*repo, "main", "tokenizer.json")
	if err != nil {
		log.Fatalf("download tokenizer: %v", err)
	}
	tokFile, err := os.Open(tokPath)
	if err != nil {
		log.Fatal(err)
	}
	bpe, err := tokenizer.LoadBPE(tokFile)
	tokFile.Close()
	if err != nil {
		log.Fatalf("load BPE: %v", err)
	}
	fmt.Printf("tokenizer: %d tokens\n", bpe.VocabSize())

	// Model.
	t0 := time.Now()
	gpt, err := models.LoadGPT2FromHF(*repo, dev)
	if err != nil {
		log.Fatalf("load model: %v", err)
	}
	fmt.Printf("loaded in %v — %.2fM params, vocab=%d ctx=%d d=%d L=%d H=%d\n",
		time.Since(t0).Round(time.Millisecond), float64(gpt.ParamCount())/1e6,
		gpt.Cfg.VocabSize, gpt.Cfg.NPositions, gpt.Cfg.NEmbd, gpt.Cfg.NLayer, gpt.Cfg.NHead)

	// Encode prompt.
	ids32 := bpe.Encode(*prompt)
	if len(ids32) == 0 {
		log.Fatal("prompt encoded to 0 tokens")
	}
	fmt.Printf("\nprompt:  %q\nids:     %v\n", *prompt, ids32)

	// Greedy generation: feed full prefix each step, take argmax of last position.
	tokens := make([]int64, len(ids32))
	for i, v := range ids32 {
		tokens[i] = int64(v)
	}
	V := gpt.Cfg.VocabSize
	for step := range *nGen {
		x, err := tensor.New(tokens, []int{1, len(tokens)}, dev)
		if err != nil {
			log.Fatal(err)
		}
		logits, err := gpt.Forward(x)
		if err != nil {
			log.Fatalf("forward: %v", err)
		}
		// (1, T, V) → take last position.
		logitsCPU, _ := logits.To(device.CPU())
		flat := tensor.MustSlice[float32](logitsCPU)
		lastOff := (len(tokens) - 1) * V
		best := 0
		for v := 1; v < V; v++ {
			if flat[lastOff+v] > flat[lastOff+best] {
				best = v
			}
		}
		tokens = append(tokens, int64(best))
		fmt.Printf("  step %d → token %d (%q)\n", step, best, bpe.Decode([]int32{int32(best)}))
		runtime.GC()
	}
	// Decode the full sequence so the user sees the generated continuation.
	finalIDs := make([]int32, len(tokens))
	for i, v := range tokens {
		finalIDs[i] = int32(v)
	}
	fmt.Printf("\noutput:  %q\n", bpe.Decode(finalIDs))
}
