// Example 40 — BERT sentence embeddings with pretrained HuggingFace weights.
//
// Loads `unsloth/bge-small-en-v1.5` (a BERT encoder exported by
// sentence-transformers; resolved from the HF cache when present), tokenizes
// a few sentences with [CLS] ... [SEP] padded to the longest, runs the
// encoder with a key-padding mask, and prints the cosine similarities of
// the CLS-pooled, L2-normalised embeddings. The numbers match PyTorch's
// `transformers.AutoModel` (see scripts/verify/ref_bert.py).
//
//	go run ./examples/40_bert_embeddings
//	go run ./examples/40_bert_embeddings -device cpu
package main

import (
	"flag"
	"fmt"
	"log"
	"strings"
	"time"

	"light/device"
	"light/hub"
	"light/models"
	"light/tensor"
	"light/tokenizer"
)

func main() {
	deviceFlag := flag.String("device", "gpu", "gpu or cpu (default gpu)")
	repo := flag.String("repo", "unsloth/bge-small-en-v1.5", "HuggingFace BERT repo")
	flag.Parse()

	sentences := []string{
		"The cat sits on the mat.",
		"A feline rests on a rug.",
		"Stock markets fell sharply today.",
		"Go is a statically typed language.",
	}
	if args := flag.Args(); len(args) > 0 {
		sentences = args
	}

	dev := pickDevice(*deviceFlag)
	fmt.Printf("device: %v\nrepo:   %s\n", dev, *repo)

	t0 := time.Now()
	m, err := models.LoadBertFromHF(*repo, dev)
	if err != nil {
		log.Fatalf("load: %v", err)
	}
	fmt.Printf("loaded in %v — %.2fM params, Layers=%d H=%d Heads=%d act=%s pooler=%v\n",
		time.Since(t0).Round(time.Millisecond), float64(m.ParamCount())/1e6,
		m.Cfg.NumLayers, m.Cfg.HiddenSize, m.Cfg.NumHeads, m.Cfg.HiddenAct, m.Pooler != nil)

	tokPath, err := hub.Download(*repo, "main", "tokenizer.json")
	if err != nil {
		log.Fatalf("tokenizer: %v", err)
	}
	tk, err := tokenizer.Load(tokPath)
	if err != nil {
		log.Fatalf("tokenizer: %v", err)
	}
	te, ok := tk.(tokenizer.TemplateEncoder)
	if !ok {
		log.Fatalf("tokenizer %T cannot add [CLS]/[SEP]", tk)
	}

	// Tokenize with specials, pad to the longest sequence.
	B := len(sentences)
	seqs := make([][]int32, B)
	T := 0
	for i, s := range sentences {
		seqs[i] = te.EncodeWithSpecial(s)
		T = max(T, len(seqs[i]))
	}
	ids := make([]int64, B*T)
	mask := make([]int64, B*T)
	for b, s := range seqs {
		for j := range T {
			ids[b*T+j] = int64(m.Cfg.PadTokenID)
		}
		for j, id := range s {
			ids[b*T+j] = int64(id)
			mask[b*T+j] = 1
		}
		fmt.Printf("[%d] %-40q → %d tokens\n", b, sentences[b], len(s))
	}
	idsT, err := tensor.New(ids, []int{B, T}, dev)
	if err != nil {
		log.Fatal(err)
	}
	maskT, err := tensor.New(mask, []int{B, T}, dev)
	if err != nil {
		log.Fatal(err)
	}

	t1 := time.Now()
	hidden, err := m.Forward(idsT, maskT)
	if err != nil {
		log.Fatalf("forward: %v", err)
	}
	cls, err := models.CLSPooled(hidden)
	if err != nil {
		log.Fatal(err)
	}
	emb, err := models.Normalize(cls)
	if err != nil {
		log.Fatal(err)
	}
	embCPU, err := emb.To(device.CPU())
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("forward (%d, %d) → %v in %v\n", B, T, hidden.Shape(), time.Since(t1).Round(time.Millisecond))

	H := m.Cfg.HiddenSize
	vals := tensor.MustSlice[float32](embCPU)
	for b := range B {
		fmt.Printf("emb[%d][:8] = %v\n", b, vals[b*H:b*H+8])
	}
	fmt.Println("\ncosine similarity (CLS-pooled, L2-normalised):")
	for i := range B {
		row := make([]string, B)
		for j := range B {
			s := float32(0)
			for k := range H {
				s += vals[i*H+k] * vals[j*H+k]
			}
			row[j] = fmt.Sprintf("%.4f", s)
		}
		fmt.Printf("[%d]  %s\n", i, strings.Join(row, "  "))
	}
}

func pickDevice(name string) device.Device {
	switch strings.ToLower(name) {
	case "cpu":
		return device.CPU()
	case "gpu", "cuda":
		d, err := device.CUDA(0)
		if err != nil {
			log.Printf("CUDA unavailable (%v); falling back to CPU", err)
			return device.CPU()
		}
		return d
	default:
		log.Fatalf("unknown device %q (gpu|cpu)", name)
		return nil
	}
}
