package models

import (
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/tensor"
)

// Local GGUF files produced with llama.cpp from the HF checkpoints (see
// CHECKPOINTS.md 2026-10-27). Tests skip when they are absent.
func localGGUF(name string) string {
	home, _ := os.UserHomeDir()
	p := filepath.Join(home, ".cache/light/gguf", name)
	if _, err := os.Stat(p); err != nil {
		return ""
	}
	return p
}

type ggufCase struct {
	file     string
	ids      []int64 // chat-template prompt "What is 2+2?" pre-tokenized
	wantTop  int     // transformers F32 top-1 on the same ids
	wantLog  float32 // transformers F32 top-1 logit
	tolLogit float32 // allowed deviation for this quant level
}

func runGGUFCase(t *testing.T, c ggufCase, dev device.Device) {
	t.Helper()
	m, err := LoadLlamaGGUF(c.file, dev)
	if err != nil {
		t.Fatal(err)
	}
	x, err := tensor.New(c.ids, []int{1, len(c.ids)}, dev)
	if err != nil {
		t.Fatal(err)
	}
	logits, err := m.ForwardWithCache(x, NewKVCache(m.Cfg.NumLayers))
	if err != nil {
		t.Fatal(err)
	}
	cpu, _ := logits.To(device.CPU())
	flat := tensor.MustSlice[float32](cpu)
	V := m.Cfg.VocabSize
	row := flat[(len(c.ids)-1)*V : len(c.ids)*V]
	best := 0
	for i, v := range row {
		if v > row[best] {
			best = i
		}
	}
	if best != c.wantTop {
		t.Errorf("%s: top-1 = %d (%.3f), want %d", filepath.Base(c.file), best, row[best], c.wantTop)
	}
	if d := row[c.wantTop] - c.wantLog; d > c.tolLogit || d < -c.tolLogit {
		t.Errorf("%s: logit[%d] = %.3f, want %.3f ± %.2f", filepath.Base(c.file), c.wantTop, row[c.wantTop], c.wantLog, c.tolLogit)
	}
}

var ggufCases = []ggufCase{
	{"qwen2-0.5b-instruct-q8_0.gguf",
		[]int64{151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198}, 17, 20.024, 0.15},
	{"qwen2-0.5b-instruct-q4_k_m.gguf",
		[]int64{151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198}, 17, 20.024, 0.8},
	{"llama-3.2-1b-instruct-q4_k_m.gguf", // llama arch: Q/K un-permute + rope_freqs
		[]int64{128000, 128006, 882, 128007, 271, 3923, 374, 220, 17, 10, 17, 30, 128009, 128006, 78191, 128007, 271}, 17, 24.877, 2.0},
}

func TestLoadLlamaGGUFCUDA(t *testing.T) {
	if !device.CUDAAvailable() {
		t.Skip("CUDA not available")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Skip(err)
	}
	ran := false
	for _, c := range ggufCases {
		if p := localGGUF(c.file); p != "" {
			c.file = p
			runGGUFCase(t, c, dev)
			ran = true
		}
	}
	if !ran {
		t.Skip("no local GGUF checkpoints in ~/.cache/light/gguf")
	}
}

// CPU path uses the dequant fallback; the 0.5B model is small enough.
func TestLoadLlamaGGUFCPU(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	c := ggufCases[0]
	p := localGGUF(c.file)
	if p == "" {
		t.Skip("no local GGUF checkpoint")
	}
	c.file = p
	runGGUFCase(t, c, device.CPU())
}
