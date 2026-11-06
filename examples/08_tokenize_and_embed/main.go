// Example 08 — Tokenize text → embed on GPU.
//
// Demonstrates the full input path of a transformer:
//
//	"hello world" → BPE → token IDs → Embedding (on GPU) → (seq, hidden) tensor
//
// The vocab is a tiny hand-crafted one; loading a real HF tokenizer.json works
// the same way via `tokenizer.LoadBPE`.
package main

import (
	"fmt"
	"log"
	"strings"

	"light/device"
	"light/nn"
	"light/tensor"
	"light/tokenizer"
)

func main() {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("device=%s\n", gpu)

	// Build a tiny BPE tokenizer capable of encoding "hello world".
	vocab := map[string]int32{
		"h": 0, "e": 1, "l": 2, "o": 3,
		"w": 4, "r": 5, "d": 6,
		"he": 7, "hel": 8, "hell": 9, "hello": 10,
		gpt2Space("w"): 11, gpt2Space("wo"): 12, gpt2Space("wor"): 13,
		gpt2Space("worl"): 14, gpt2Space("world"): 15,
	}
	merges := []string{
		"h e", "he l", "hel l", "hell o",
		"\u0120 w", "\u0120w o", "\u0120wo r", "\u0120wor l", "\u0120worl d",
	}
	bpe, err := tokenizer.NewBPE(vocab, merges)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("vocab size: %d\n", bpe.VocabSize())

	// Encode a sentence.
	text := "hello world"
	ids := bpe.Encode(text)
	fmt.Printf("tokenized %q → ids=%v → tokens=%v\n", text, ids, idsToTokens(bpe, ids))

	// Decode to verify round-trip.
	back := bpe.Decode(ids)
	fmt.Printf("decoded: %q\n", back)

	// Embed tokens on GPU.
	embed, err := nn.NewEmbedding(bpe.VocabSize(), 16, gpu)
	if err != nil {
		log.Fatal(err)
	}

	// Put ids into an I64 tensor on CPU, then upload to GPU and look up.
	idsI64 := make([]int64, len(ids))
	for i, v := range ids {
		idsI64[i] = int64(v)
	}
	idxCPU, err := tensor.New(idsI64, []int{len(ids)}, cpu)
	if err != nil {
		log.Fatal(err)
	}
	idxG, _ := idxCPU.To(gpu)

	embedded, err := embed.Forward(idxG)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("embedded shape=%s device=%s\n", embedded.Shape(), embedded.Device())

	// Bring back for stats.
	eCPU, _ := embedded.To(cpu)
	vals := tensor.MustSlice[float32](eCPU)
	var minV, maxV, sum float32
	minV = vals[0]
	maxV = vals[0]
	for _, v := range vals {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
		sum += v
	}
	mean := sum / float32(len(vals))
	fmt.Printf("embedding stats: min=%.3f max=%.3f mean=%.3f\n", minV, maxV, mean)
	fmt.Println("text → tokens → GPU embedding: success.")
}

// gpt2Space prefixes a token with Ġ (U+0120), which is the GPT-2 byte-encoder
// mapping of the ASCII space character.
func gpt2Space(s string) string { return "\u0120" + s }

// idsToTokens maps a slice of IDs back to their printable forms (with Ġ
// rendered as "·" for readability).
func idsToTokens(bpe *tokenizer.BPE, ids []int32) []string {
	out := make([]string, len(ids))
	for i, id := range ids {
		out[i] = strings.ReplaceAll(bpe.IDToToken(id), "\u0120", "·")
	}
	return out
}
