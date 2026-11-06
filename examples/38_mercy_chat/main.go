// Example 38 — MercyLM inference / training demo.
//
// MercyLM is a 15.4 M-parameter GPT-style causal LM from
// https://github.com/ravipatib/mercylm — trained for conversational
// response with a 4096-token BPE vocabulary and a 128-token context window.
//
// Without a checkpoint the model produces random output (useful for
// verifying the forward pass compiles and runs).  With a safetensors
// checkpoint produced by `tools/export_to_hf.py` from the reference
// training run you get real responses.
//
// Usage:
//
//	# Random-weight smoke test (no file needed)
//	go run ./examples/38_mercy_chat
//
//	# Load a trained checkpoint
//	go run ./examples/38_mercy_chat -weights /path/to/model.safetensors
//
//	# Override vocab size (read from vocab_size.txt in the original repo)
//	go run ./examples/38_mercy_chat -vocab 4096 -weights /path/to/model.safetensors
//
// Conversation format (mirrors the reference training data):
//
//	[USER] <input> [MERCY] <response> [END]
//
// Special token IDs are assumed to be the last four entries of the BPE
// vocabulary in training order:  [UNK]=0, [USER]=vocabSize-3,
// [MERCY]=vocabSize-2, [END]=vocabSize-1.  Adjust -user/-mercy/-end flags
// if your tokenizer placed them elsewhere.
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"time"

	"light/device"
	"light/models"
	"light/tensor"
)

func main() {
	weightsPath := flag.String("weights", "", "path to safetensors checkpoint (optional)")
	vocabSize := flag.Int("vocab", 4096, "vocabulary size (match vocab_size.txt from training)")
	maxTokens := flag.Int("tokens", 80, "max new tokens to generate")
	temp := flag.Float64("temp", 0.75, "sampling temperature (0 = greedy)")
	topK := flag.Int("topk", 40, "top-k cutoff (0 = disabled)")
	userTok := flag.Int("user", -1, "[USER] token id (-1 = vocabSize-3)")
	mercyTok := flag.Int("mercy", -1, "[MERCY] token id (-1 = vocabSize-2)")
	endTok := flag.Int("end", -1, "[END] token id (-1 = vocabSize-1)")
	devFlag := flag.String("device", "auto", "cpu | cuda | auto")
	flag.Parse()

	// Resolve special token IDs.
	if *userTok < 0 {
		*userTok = *vocabSize - 3
	}
	if *mercyTok < 0 {
		*mercyTok = *vocabSize - 2
	}
	if *endTok < 0 {
		*endTok = *vocabSize - 1
	}

	var (
		dev device.Device
		err error
	)
	switch *devFlag {
	case "cpu":
		dev = device.CPU()
	case "cuda":
		dev, err = device.CUDA(0)
		if err != nil {
			log.Fatalf("CUDA: %v", err)
		}
	default:
		dev = device.Auto()
	}

	cfg := models.DefaultMercyConfig(*vocabSize)

	var m *models.MercyLM
	if *weightsPath != "" {
		fmt.Printf("Loading MercyLM from %s ...\n", *weightsPath)
		t0 := time.Now()
		m, err = models.LoadMercyFromSafetensors(*weightsPath, cfg, dev)
		if err != nil {
			log.Fatalf("load: %v", err)
		}
		fmt.Printf("Loaded in %.2fs  params=%d\n", time.Since(t0).Seconds(), m.ParamCount())
	} else {
		fmt.Println("No checkpoint provided — running with random weights (output will be nonsense).")
		m, err = models.NewMercyLM(cfg, dev)
		if err != nil {
			log.Fatalf("new model: %v", err)
		}
		fmt.Printf("MercyLM  params=%d  device=%v\n", m.ParamCount(), dev)
	}

	// Minimal inference loop using raw token IDs.
	// In a real deployment you would tokenize with the BPE tokenizer from the
	// reference repo (tokenizer.json) via light/tokenizer.
	//
	// Here we demonstrate with a hand-crafted prompt:
	//   [USER] hello [MERCY]
	// represented as raw integer IDs — sufficient to prove the forward pass.
	rng := rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	opts := models.SampleOpts{
		Temperature: float32(*temp),
		TopK:        *topK,
		Rng:         rng,
	}

	// Seed prompt: [USER] token + some placeholder tokens + [MERCY] token.
	// Replace with real BPE-encoded IDs when using a trained checkpoint.
	prompt := []int64{int64(*userTok), 42, 100, 200, int64(*mercyTok)}
	fmt.Printf("\nPrompt token IDs: %v\n", prompt)
	fmt.Printf("Generating up to %d tokens (temp=%.2f, top-k=%d)...\n\n",
		*maxTokens, *temp, *topK)

	generated, err := mercyGenerate(m, prompt, *maxTokens, int64(*endTok), opts, dev)
	if err != nil {
		log.Fatalf("generate: %v", err)
	}
	fmt.Printf("Generated token IDs: %v\n", generated)
	fmt.Println("\n(Decode with the BPE tokenizer from the reference repo to get text.)")
}

// mercyGenerate runs autoregressive generation from a prompt until [END] is
// sampled or maxNewTokens is exhausted. Returns the newly generated IDs (not
// including the prompt).
func mercyGenerate(
	m *models.MercyLM,
	prompt []int64,
	maxNewTokens int,
	endID int64,
	opts models.SampleOpts,
	dev device.Device,
) ([]int64, error) {
	ctx := make([]int64, len(prompt))
	copy(ctx, prompt)
	generated := make([]int64, 0, maxNewTokens)

	for range maxNewTokens {
		// Crop to context window.
		start := 0
		if len(ctx) > m.Cfg.ContextLen {
			start = len(ctx) - m.Cfg.ContextLen
		}
		window := ctx[start:]

		ids, err := tensor.New(window, []int{1, len(window)}, dev)
		if err != nil {
			return nil, err
		}

		logits, err := m.Forward(ids)
		if err != nil {
			return nil, err
		}
		// logits: (1, T, V) — take the last time step.
		T := logits.Dim(1)
		V := logits.Dim(2)
		lastLogits, err := logits.Narrow(1, T-1, 1)
		if err != nil {
			return nil, err
		}
		lastC, err := lastLogits.Contiguous()
		if err != nil {
			return nil, err
		}
		flat, err := lastC.Reshape(V)
		if err != nil {
			return nil, err
		}
		// Pull to CPU for sampling.
		var cpuFlat *tensor.Tensor
		if flat.Device().Kind() != 0 { // 0 = CPU
			cpuFlat, err = flat.To(device.CPU())
			if err != nil {
				return nil, err
			}
		} else {
			cpuFlat = flat
		}
		logSlice := tensor.MustSlice[float32](cpuFlat)
		next := int64(models.SampleLogits(logSlice, opts))

		generated = append(generated, next)
		ctx = append(ctx, next)
		if next == endID {
			break
		}
	}
	return generated, nil
}
