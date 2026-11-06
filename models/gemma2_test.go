package models

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
)

// Gemma 2 2B: no QK-norm, attention softcap 50, final logit softcap 30,
// alternating local (4096) / global layers, RoPE base 1e4 everywhere.
// Reference: HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify/ref_gemma.py unsloth/gemma-2-2b-it f32
const gemma2Repo = "unsloth/gemma-2-2b-it"

var gemma2ChatIDs = []int64{2, 106, 1645, 108, 1841, 603, 235248, 235284, 235340, 235284, 235336, 107, 108, 106, 2516, 108}

var gemma2Ref = []struct {
	id    int
	logit float32
}{{235284, 21.2943}, {651, 12.4847}, {235248, 11.2188}, {235310, 10.6975}, {9069, 10.6583}}

var gemma2Greedy = []int64{235284, 963, 235248, 235284, 589, 235248, 235310, 235248, 108, 107}

func TestGemma2MatchesTransformersF32(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (2.6B params, 5 GB host staging)")
	}
	requireHostMemGB(t, 12)
	releaseModelMemory(t)
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--unsloth--gemma-2-2b-it/snapshots/*/model*.safetensors")); len(m) == 0 {
		t.Skip(gemma2Repo + " not in HF cache")
	}
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	os.Setenv("HF_HUB_OFFLINE", "1")
	m, err := LoadGemma3FromHF(gemma2Repo, dev, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	if !m.Cfg.NoQKNorm || m.Cfg.AttnLogitSoftcap != 50 || m.Cfg.FinalLogitSoftcap != 30 || !m.Cfg.isSliding(0) || m.Cfg.isSliding(1) || m.Blocks[0].QNorm != nil {
		t.Fatalf("config %+v", m.Cfg)
	}
	if got := m.ParamCount(); got < 2_600_000_000 || got > 2_630_000_000 {
		t.Fatalf("param count %d", got)
	}
	row := lastLogits(t, m, gemma2ChatIDs, NewKVCache(m.NumLayers()), dev)
	got := top5(row)
	for i, w := range gemma2Ref {
		if got[i] != w.id {
			t.Errorf("top-%d: got id %d (%.4f), want %d (%.4f)", i+1, got[i], row[got[i]], w.id, w.logit)
		}
		if d := math.Abs(float64(row[w.id] - w.logit)); d > 0.05 {
			t.Errorf("logit[%d] = %.4f, want %.4f", w.id, row[w.id], w.logit)
		}
	}
	t.Logf("top-5 %v logits %.4f %.4f %.4f %.4f %.4f", got, row[got[0]], row[got[1]], row[got[2]], row[got[3]], row[got[4]])

	cache := NewKVCache(m.NumLayers())
	var out []int64
	step := lastLogits(t, m, gemma2ChatIDs, cache, dev)
	for len(out) < len(gemma2Greedy) {
		next := int64(top5(step)[0])
		out = append(out, next)
		if next == 107 || next == 1 { // <end_of_turn> / <eos> in the Gemma 2 vocab
			break
		}
		step = lastLogits(t, m, []int64{next}, cache, dev)
	}
	t.Logf("greedy %v", out)
	for i := range gemma2Greedy {
		if i >= len(out) || out[i] != gemma2Greedy[i] {
			t.Fatalf("greedy ids %v, want %v", out, gemma2Greedy)
		}
	}
}
