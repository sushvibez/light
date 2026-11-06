// Example 35 — LLM inference with pretrained HuggingFace weights.
//
// Downloads `Qwen/Qwen2-0.5B-Instruct` (or any Qwen2/2.5, Llama-3.x, Phi-3.x
// or Mistral repo) on first run, loads its safetensors checkpoint into our
// `models.Llama`, tokenizes a prompt with the repo's BPE (byte-level or
// SentencePiece-style), and runs greedy generation.
//
//	go run            ./examples/35_qwen_inference
//	go run            ./examples/35_qwen_inference -tokens 32 -prompt "Q: What is 2+2? A:"
//	go run            ./examples/35_qwen_inference -repo meta-llama/Llama-3.2-1B-Instruct -prompt "The capital of France is"
//	go run            ./examples/35_qwen_inference -repo microsoft/Phi-3.5-mini-instruct -bf16 -prompt $'<|user|>\nWhat is 2+2?<|end|>\n<|assistant|>\n'
//
// Pass -logits to print the top-5 next-token logits after prefill and compare
// them with scripts/verify/ref_transformers.py; that comparison on a chat
// prompt of 15+ tokens is the validation bar for a new model.
//
// First run downloads ~1 GB into ~/.cache/huggingface/hub/. Subsequent runs
// reuse the cached files. Default batch is 1; sequence stays in O(N²) memory
// since a KV cache is not yet wired in.
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"runtime"
	"sort"
	"time"

	"light/device"
	"light/dtype"
	"light/hub"
	"light/internal/cudart"
	"light/models"
	"light/tensor"
	"light/tokenizer"
)

func main() {
	deviceFlag := flag.String("device", "gpu", "gpu or cpu (default gpu)")
	repo := flag.String("repo", "Qwen/Qwen2-0.5B-Instruct", "HuggingFace repo")
	prompt := flag.String("prompt", "The quick brown fox jumps over the lazy", "prompt text")
	nGen := flag.Int("tokens", 16, "number of tokens to generate")
	temp := flag.Float64("temp", 0, "sampling temperature (0 = greedy)")
	topK := flag.Int("top-k", 0, "top-k cap (0 = disabled)")
	topP := flag.Float64("top-p", 0, "top-p / nucleus cap (0 = disabled, e.g. 0.9)")
	seed := flag.Uint64("seed", 0, "RNG seed (0 = nondeterministic)")
	stopOnEOS := flag.Bool("stop-on-eos", true, "halt generation on <|im_end|>/<|endoftext|>")
	mixed := flag.Bool("bf16", false, "full BF16 inference (weights + activations + matmul output) — half the GPU memory and tensor-core throughput; F32 path otherwise")
	addBOS := flag.Bool("bos", true, "prepend the model's BOS token (config.json bos_token_id) when it has one; Llama-3 needs it, Qwen2 has none")
	ropeScaling := flag.Bool("rope-scaling", true, "apply the config's rope_scaling (debug toggle)")
	showLogits := flag.Bool("logits", false, "print the top-5 next-token logits after prefill (compare with scripts/verify/ref_transformers.py)")
	ggufPath := flag.String("gguf", "", "load weights from a llama.cpp GGUF file (quantized Linear weights stay quantized); -repo still supplies the tokenizer and EOS ids")
	flag.Parse()

	dev := pickDevice(*deviceFlag)
	fmt.Printf("device:  %v\n", dev)
	fmt.Printf("repo:    %s\n", *repo)

	// EOS-class tokens from generation_config.json / config.json: Qwen2 has
	// <|im_end|> + <|endoftext|>; Llama-3 has end_of_text + eom + eot.
	eosIDs := map[int64]bool{}
	if ids, err := models.EOSTokenIDs(*repo); err == nil {
		for _, id := range ids {
			eosIDs[id] = true
		}
	}

	// Tokenizer.
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

	// Model — downloads + loads in one call; the loader is chosen from the
	// config's model_type (qwen2, llama, mistral). BF16 weights are upcast
	// to F32 by default; pass -bf16 to keep them in BF16 (half the GPU memory).
	t0 := time.Now()
	wdt := dtype.F32
	if *mixed {
		wdt = dtype.BF16
	}
	var model *models.Llama
	if *ggufPath != "" {
		model, err = models.LoadLlamaGGUF(*ggufPath, dev)
	} else {
		model, err = models.LoadCausalLMFromHF(*repo, dev, wdt)
	}
	if err != nil {
		log.Fatalf("load model: %v", err)
	}
	if !*ropeScaling {
		model.Cfg.RopeScaling = nil
	}
	cfg := model.Cfg
	fmt.Printf("loaded in %v — %.2fM params, vocab=%d ctx=%d d=%d L=%d H=%d/KV=%d rope_scaling=%v\n",
		time.Since(t0).Round(time.Millisecond), float64(model.ParamCount())/1e6,
		cfg.VocabSize, cfg.MaxPosition, cfg.HiddenSize, cfg.NumLayers, cfg.NumHeads, cfg.NumKVHeads, cfg.RopeScaling != nil)
	runtime.GC()
	printGPUMem("after model load")

	// Encode prompt (+ BOS when the model defines one and the prompt does
	// not already start with it).
	ids32 := bpe.Encode(*prompt)
	if len(ids32) == 0 {
		log.Fatal("prompt encoded to 0 tokens")
	}
	if *addBOS {
		if bos, ok := models.BOSTokenID(*repo); ok && (len(ids32) == 0 || int64(ids32[0]) != bos) {
			ids32 = append([]int32{int32(bos)}, ids32...)
		}
	}
	fmt.Printf("\nprompt:  %q\nids:     %v (n=%d)\n", *prompt, ids32, len(ids32))

	tokens := make([]int64, len(ids32))
	for i, v := range ids32 {
		tokens[i] = int64(v)
	}
	V := cfg.VocabSize
	cache := models.NewKVCache(cfg.NumLayers)

	// Sampler.
	var rng *rand.Rand
	if *seed != 0 {
		rng = rand.New(rand.NewPCG(*seed, *seed^0x9E3779B97F4A7C15))
	} else {
		rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}
	sampleOpts := models.SampleOpts{
		Temperature: float32(*temp),
		TopK:        *topK,
		TopP:        float32(*topP),
		Rng:         rng,
	}

	// Prefill: feed the whole prompt once. logits[..., -1, :] gives the score
	// distribution for the *next* token; subsequent decode steps feed only
	// that newly-sampled token thanks to the KV cache.
	prefill0 := time.Now()
	x, err := tensor.New(tokens, []int{1, len(tokens)}, dev)
	if err != nil {
		log.Fatal(err)
	}
	logits, err := model.ForwardWithCache(x, cache)
	if err != nil {
		log.Fatalf("prefill: %v", err)
	}
	logitsCPU, _ := logits.To(device.CPU())
	flat := tensor.MustSlice[float32](logitsCPU)
	prefillDur := time.Since(prefill0)

	if *showLogits {
		printTop5(flat[(len(tokens)-1)*V:len(tokens)*V], bpe)
	}
	first := models.SampleLogits(flat[(len(tokens)-1)*V:len(tokens)*V], sampleOpts)
	tokens = append(tokens, int64(first))
	fmt.Printf("  prefill in %v  → token %6d  %q\n",
		prefillDur.Round(time.Millisecond), first, bpe.Decode([]int32{int32(first)}))
	stopped := *stopOnEOS && eosIDs[int64(first)]

	gen0 := time.Now()
	for step := 1; step < *nGen && !stopped; step++ {
		// Decode: feed only the newest token; cache provides past context.
		newTok := []int64{tokens[len(tokens)-1]}
		x, err := tensor.New(newTok, []int{1, 1}, dev)
		if err != nil {
			log.Fatal(err)
		}
		logits, err := model.ForwardWithCache(x, cache)
		if err != nil {
			log.Fatalf("decode step %d: %v", step, err)
		}
		logitsCPU, _ := logits.To(device.CPU())
		flat := tensor.MustSlice[float32](logitsCPU)
		// Output is (1, 1, V).
		best := models.SampleLogits(flat[:V], sampleOpts)
		tokens = append(tokens, int64(best))
		fmt.Printf("  step %2d  cache=%d  → token %6d  %q\n",
			step, cache.Len, best, bpe.Decode([]int32{int32(best)}))
		runtime.GC()
		if *stopOnEOS && eosIDs[int64(best)] {
			stopped = true
		}
	}
	totalDur := time.Since(gen0)

	finalIDs := make([]int32, len(tokens))
	for i, v := range tokens {
		finalIDs[i] = int32(v)
	}
	fmt.Printf("\noutput:  %q\n", bpe.Decode(finalIDs))
	if *nGen > 1 {
		fmt.Printf("\nprefill %v + %d decode steps in %v  (%.1f tok/s decode)\n",
			prefillDur.Round(time.Millisecond), *nGen-1, totalDur.Round(time.Millisecond),
			float64(*nGen-1)/totalDur.Seconds())
	} else {
		fmt.Printf("\nprefill %v\n", prefillDur.Round(time.Millisecond))
	}
}

// printTop5 lists the five largest logits in the same format as
// scripts/verify/ref_transformers.py so the two can be diffed by eye.
func printTop5(row []float32, tk *tokenizer.BPE) {
	idx := make([]int, len(row))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return row[idx[a]] > row[idx[b]] })
	fmt.Println("  top5 (light):")
	for _, i := range idx[:5] {
		fmt.Printf("    id=%-7d %-12q logit=%.4f\n", i, tk.Decode([]int32{int32(i)}), row[i])
	}
}

func printGPUMem(label string) {
	free, total, err := cudart.MemInfo()
	if err != nil {
		return
	}
	fmt.Printf("[gpu mem %s] free=%d MB used=%d MB total=%d MB\n",
		label, free>>20, (total-free)>>20, total>>20)
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
