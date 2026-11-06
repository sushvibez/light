// Example 36 — Qwen2 multi-turn chat with KV-cache reuse.
//
// Same model and tokenizer as `examples/35_qwen_inference`, but interactive:
// each user line is wrapped in the Qwen chat template, only the *new* tokens
// are fed through the prefill, and the KV cache carries the full history
// across turns. So the second turn only pays for the new user message + the
// assistant reply, not for re-encoding the entire conversation.
//
// Type your message and hit enter. `/exit` quits, `/reset` clears the
// conversation and starts a new system prompt.
//
//	go run            ./examples/36_qwen_chat
//	go run            ./examples/36_qwen_chat -repo Qwen/Qwen2.5-3B-Instruct -bf16
//	go run            ./examples/36_qwen_chat -system "You are a terse Go expert."
package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"runtime"
	"strings"
	"time"

	"light/device"
	"light/hub"
	"light/internal/cudart"
	"light/models"
	"light/tensor"
	"light/tokenizer"
)

// Qwen2 / Qwen2.5 EOS-class tokens. <|im_end|> ends a turn; <|endoftext|>
// closes the document. Either one halts generation.
var eosIDs = map[int64]bool{151643: true, 151645: true}

func main() {
	deviceFlag := flag.String("device", "gpu", "gpu or cpu")
	repo := flag.String("repo", "Qwen/Qwen2-0.5B-Instruct", "HuggingFace repo")
	system := flag.String("system", "You are a helpful assistant.", "system prompt")
	maxNew := flag.Int("max-new", 256, "max tokens per assistant turn")
	temp := flag.Float64("temp", 0.7, "sampling temperature (0 = greedy)")
	topK := flag.Int("top-k", 0, "top-k cap (0 = off)")
	topP := flag.Float64("top-p", 0.9, "top-p cap (0 = off)")
	seed := flag.Uint64("seed", 0, "RNG seed (0 = nondeterministic)")
	bf16 := flag.Bool("bf16", false, "load weights at BF16 (half the GPU memory)")
	flag.Parse()

	dev := pickDevice(*deviceFlag)
	fmt.Printf("device:  %v\nrepo:    %s\n", dev, *repo)

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

	// Model.
	t0 := time.Now()
	var model *models.Llama
	if *bf16 {
		model, err = models.LoadQwen2FromHFMixed(*repo, dev)
	} else {
		model, err = models.LoadQwen2FromHF(*repo, dev)
	}
	if err != nil {
		log.Fatalf("load model: %v", err)
	}
	cfg := model.Cfg
	fmt.Printf("loaded in %v — %.2fM params\n",
		time.Since(t0).Round(time.Millisecond), float64(model.ParamCount())/1e6)
	if free, total, err := cudart.MemInfo(); err == nil {
		fmt.Printf("gpu:     %d MB used / %d MB total\n\n",
			(total-free)>>20, total>>20)
	}

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

	cache := models.NewKVCache(cfg.NumLayers)
	primeSystem(model, cache, bpe, *system, dev)

	in := bufio.NewReader(os.Stdin)
	turn := 0
	for {
		fmt.Print("you> ")
		line, err := in.ReadString('\n')
		if err != nil {
			fmt.Println()
			return
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			continue
		}
		switch line {
		case "/exit", "/quit":
			return
		case "/reset":
			cache.Reset()
			primeSystem(model, cache, bpe, *system, dev)
			turn = 0
			fmt.Println("(history cleared)")
			continue
		}
		turn++
		fmt.Printf("ai>  ")
		t1 := time.Now()
		nGen, err := chatTurn(model, cache, bpe, line, *maxNew, sampleOpts, dev)
		dur := time.Since(t1).Round(time.Millisecond)
		if err != nil {
			fmt.Printf("\n[error] %v\n", err)
			continue
		}
		fmt.Printf("\n     (turn %d  %d tok in %v  %.1f tok/s  cache=%d)\n\n",
			turn, nGen, dur, float64(nGen)/dur.Seconds(), cache.Len)
	}
}

// primeSystem feeds the system prompt through the model so the cache holds
// the system tokens for the rest of the session. We don't need the logits.
func primeSystem(m *models.Llama, cache *models.KVCache, bpe *tokenizer.BPE, system string, dev device.Device) {
	header := "<|im_start|>system\n" + system + "<|im_end|>\n"
	ids := bpe.Encode(header)
	if len(ids) == 0 {
		return
	}
	tokens := toI64(ids)
	x, err := tensor.New(tokens, []int{1, len(tokens)}, dev)
	if err != nil {
		log.Fatalf("system prompt tensor: %v", err)
	}
	if _, err := m.ForwardWithCache(x, cache); err != nil {
		log.Fatalf("system prompt forward: %v", err)
	}
	runtime.GC()
}

// chatTurn appends one user message + the assistant prefix to the cache,
// then samples tokens until EOS or `maxNew`. Returns the number of tokens
// generated. Decoded text streams to stdout as it's produced.
func chatTurn(m *models.Llama, cache *models.KVCache, bpe *tokenizer.BPE, user string, maxNew int, opts models.SampleOpts, dev device.Device) (int, error) {
	cfg := m.Cfg
	// Prefill: <|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
	prefix := "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"
	pre := bpe.Encode(prefix)
	if len(pre) == 0 {
		return 0, fmt.Errorf("user message tokenized to 0 tokens")
	}
	if os.Getenv("LIGHT_CHAT_DEBUG") != "" {
		fmt.Printf("\n[debug] prefix ids (n=%d): %v\n", len(pre), pre)
	}
	x, err := tensor.New(toI64(pre), []int{1, len(pre)}, dev)
	if err != nil {
		return 0, fmt.Errorf("prefill tensor: %w", err)
	}
	logits, err := m.ForwardWithCache(x, cache)
	if err != nil {
		return 0, fmt.Errorf("prefill: %w", err)
	}
	flat, err := lastRow(logits, cfg.VocabSize, len(pre))
	if err != nil {
		return 0, err
	}
	first := models.SampleLogits(flat, opts)
	if eosIDs[int64(first)] {
		// Empty reply — the model went straight to <|im_end|>. Don't print
		// anything, but still finalize the turn so the cache stays consistent.
		appendTurnSeparator(m, cache, bpe, dev)
		return 1, nil
	}
	streamToken(bpe, first)
	tokens := []int64{int64(first)}

	for step := 1; step < maxNew; step++ {
		newTok := []int64{tokens[len(tokens)-1]}
		x, err := tensor.New(newTok, []int{1, 1}, dev)
		if err != nil {
			return step, err
		}
		logits, err := m.ForwardWithCache(x, cache)
		if err != nil {
			return step, fmt.Errorf("decode step %d: %w", step, err)
		}
		flat, err := lastRow(logits, cfg.VocabSize, 1)
		if err != nil {
			return step, err
		}
		next := models.SampleLogits(flat, opts)
		tokens = append(tokens, int64(next))
		if eosIDs[int64(next)] {
			appendTurnSeparator(m, cache, bpe, dev)
			return step + 1, nil
		}
		streamToken(bpe, next)
		runtime.GC()
	}
	// Hit max-new without EOS — close the turn ourselves so the cache has
	// the canonical separator before the next turn.
	appendTurnSeparator(m, cache, bpe, dev)
	return maxNew, nil
}

// appendTurnSeparator pushes "<|im_end|>\n" through the cache so the next
// user turn starts at the canonical boundary. The EOS we just sampled was
// *not* fed through ForwardWithCache (we stop on it before the next forward),
// so the cache is missing both the `<|im_end|>` and the trailing newline —
// without this the next turn's tokens get attached to a half-closed
// assistant block and the model drifts into role-token gibberish.
//
// If the loop instead aborted at maxNew without seeing EOS, we still inject
// the canonical separator so the conversation stays well-formed.
func appendTurnSeparator(m *models.Llama, cache *models.KVCache, bpe *tokenizer.BPE, dev device.Device) {
	sep := bpe.Encode("<|im_end|>\n")
	if len(sep) == 0 {
		return
	}
	x, err := tensor.New(toI64(sep), []int{1, len(sep)}, dev)
	if err != nil {
		return
	}
	_, _ = m.ForwardWithCache(x, cache)
}

func streamToken(bpe *tokenizer.BPE, id int) {
	piece := bpe.Decode([]int32{int32(id)})
	fmt.Print(piece)
}

func lastRow(logits *tensor.Tensor, V, T int) ([]float32, error) {
	cpu, err := logits.To(device.CPU())
	if err != nil {
		return nil, err
	}
	flat := tensor.MustSlice[float32](cpu)
	off := (T - 1) * V
	return flat[off : off+V], nil
}

func toI64(ids []int32) []int64 {
	out := make([]int64, len(ids))
	for i, v := range ids {
		out[i] = int64(v)
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
