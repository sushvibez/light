package models

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
)

// Gemma 4 E2B: PLE, 20 KV-shared layers with double-wide MLPs, global
// head dim 512 with proportional RoPE, unscaled attention, final softcap 30.
// Multimodal checkpoint loaded text-only; 10 GB BF16 single file, so gated.
// Reference: HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify/ref_gemma4.py unsloth/gemma-4-E2B-it bf16
const gemma4Repo = "unsloth/gemma-4-E2B-it"

var gemma4ChatIDs = []int64{2, 105, 2364, 107, 3689, 563, 236743, 236778, 236862, 236778, 236881, 106, 107, 105, 4368, 107}

var gemma4Ref = []struct {
	id    int
	logit float32
}{{236778, 22.125}, {236795, 18.0}, {818, 17.0}, {7925, 12.75}, {100, 9.875}}

var gemma4Greedy = []int64{236778, 900, 236743, 236778, 578, 5213, 236812, 1018, 106}

func TestGemma4E2BMatchesTransformersBF16(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (10 GB BF16 checkpoint)")
	}
	requireHostMemGB(t, 10)
	releaseModelMemory(t)
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--unsloth--gemma-4-E2B-it/snapshots/*/model*.safetensors")); len(m) == 0 {
		t.Skip(gemma4Repo + " not in HF cache")
	}
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	os.Setenv("HF_HUB_OFFLINE", "1")
	m, err := LoadGemma4FromHF(gemma4Repo, dev, dtype.BF16)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cfg
	if c.NumLayers != 35 || c.NumKVSharedLayers != 20 || !c.DoubleWideMLP || c.PLEDim != 256 || c.GlobalHeadDim != 512 ||
		c.PartialRotary != 0.25 || c.FinalLogitSoftcap != 30 || c.SlidingWindow != 512 || c.isSliding(4) || !c.isSliding(3) ||
		m.Blocks[14].KProj == nil || m.Blocks[15].KProj != nil || c.kvSourceLayer(15) != 13 || c.kvSourceLayer(19) != 14 ||
		m.Blocks[15].GateProj.W.Dim(0) != 2*c.IntermediateSize {
		t.Fatalf("config %+v", c)
	}
	t.Logf("text params %.3f B", float64(m.ParamCount())/1e9)
	row := lastLogits(t, m, gemma4ChatIDs, NewKVCache(m.NumLayers()), dev)
	got := top5(row)
	for i, w := range gemma4Ref {
		if i < 3 && got[i] != w.id {
			t.Errorf("top-%d: got id %d (%.3f), want %d (%.3f)", i+1, got[i], row[got[i]], w.id, w.logit)
		}
		if d := math.Abs(float64(row[w.id] - w.logit)); d > 1.0 {
			t.Errorf("logit[%d] = %.3f, want %.3f (BF16 tol 1.0)", w.id, row[w.id], w.logit)
		}
	}
	t.Logf("top-5 %v logits %.3f %.3f %.3f %.3f %.3f", got, row[got[0]], row[got[1]], row[got[2]], row[got[3]], row[got[4]])

	cache := NewKVCache(m.NumLayers())
	var out []int64
	step := lastLogits(t, m, gemma4ChatIDs, cache, dev)
	for len(out) < len(gemma4Greedy) {
		next := int64(top5(step)[0])
		out = append(out, next)
		if next == 106 || next == 1 || next == 50 { // <turn|> / <eos> / generation_config eos
			break
		}
		step = lastLogits(t, m, []int64{next}, cache, dev)
	}
	t.Logf("greedy %v", out)
	for i := range gemma4Greedy {
		if i >= len(out) || out[i] != gemma4Greedy[i] {
			t.Fatalf("greedy ids %v, want %v", out, gemma4Greedy)
		}
	}
}
