package models

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
)

// Gemma 3 4B: the multimodal checkpoint (Gemma3ForConditionalGeneration)
// loaded text-only — nested text_config, language_model.model.* keys, linear
// rope_scaling ×8 on the global layers. 8.6 GB BF16, so gated.
// Reference: HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify/ref_gemma.py unsloth/gemma-3-4b-it bf16
const gemma4BRepo = "unsloth/gemma-3-4b-it"

var gemma4BRef = []struct {
	id    int
	logit float32
}{{236778, 56.0}, {6372, 34.0}, {818, 33.0}, {1018, 32.5}, {11634, 32.0}}

var gemma4BGreedy = []int64{236778, 900, 236743, 236778, 578, 236743, 236812, 107, 106}

func TestGemma3_4BMatchesTransformersBF16(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (8.6 GB BF16 checkpoint)")
	}
	requireHostMemGB(t, 14)
	releaseModelMemory(t)
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--unsloth--gemma-3-4b-it/snapshots/*/model.safetensors.index.json")); len(m) == 0 {
		t.Skip(gemma4BRepo + " not in HF cache")
	}
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	os.Setenv("HF_HUB_OFFLINE", "1")
	m, err := LoadGemma3FromHF(gemma4BRepo, dev, dtype.BF16)
	if err != nil {
		t.Fatal(err)
	}
	if m.Cfg.RopeScale != 8 || m.Cfg.NumLayers != 34 || m.Cfg.SlidingWindow != 1024 || m.Cfg.isSliding(5) || !m.Cfg.isSliding(4) {
		t.Fatalf("config %+v", m.Cfg)
	}
	if got := m.ParamCount(); got < 3_850_000_000 || got > 3_900_000_000 {
		t.Fatalf("text param count %d", got)
	}
	row := lastLogits(t, m, gemmaChatIDs, NewKVCache(m.NumLayers()), dev)
	got := top5(row)
	for i, w := range gemma4BRef {
		if i < 3 && got[i] != w.id {
			t.Errorf("top-%d: got id %d (%.3f), want %d (%.3f)", i+1, got[i], row[got[i]], w.id, w.logit)
		}
		if d := math.Abs(float64(row[w.id] - w.logit)); d > 1.0 {
			t.Errorf("logit[%d] = %.3f, want %.3f (BF16 tol 1.0)", w.id, row[w.id], w.logit)
		}
	}
	t.Logf("top-5 %v logits %.3f %.3f %.3f %.3f %.3f", got, row[got[0]], row[got[1]], row[got[2]], row[got[3]], row[got[4]])

	// Greedy continuation through the cache must reproduce transformers.
	cache := NewKVCache(m.NumLayers())
	ids := append([]int64(nil), gemmaChatIDs...)
	var out []int64
	step := lastLogits(t, m, ids, cache, dev)
	for len(out) < len(gemma4BGreedy) {
		next := int64(top5(step)[0])
		out = append(out, next)
		if next == 106 || next == 1 { // <end_of_turn> / <eos>
			break
		}
		step = lastLogits(t, m, []int64{next}, cache, dev)
	}
	t.Logf("greedy %v", out)
	for i := range gemma4BGreedy {
		if i >= len(out) || out[i] != gemma4BGreedy[i] {
			t.Fatalf("greedy ids %v, want %v", out, gemma4BGreedy)
		}
	}
}
