package models

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
)

// Phi-4-mini-instruct: model_type phi3 with partial_rotary_factor 0.75
// (96 of 128 head dims rotate; 48 longrope factors), tied embeddings, GPT-4o
// vocabulary (200064), fused qkv/gate_up weights. 3.8B params, 7.7 GB BF16,
// so gated. Reference: HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify/ref_phi4.py
const phi4MiniRepo = "microsoft/Phi-4-mini-instruct"

var phi4MiniChatIDs = []int64{200022, 3575, 553, 261, 82463, 29186, 13, 200020, 200021, 176289, 306, 1001, 21872, 4436, 290, 17307, 382, 9861, 13, 200020, 200019}

var phi4MiniRef = []struct {
	id    int
	logit float32
}{{976, 47.75}, {37513, 42.5}, {15957, 40.5}, {15952, 40.5}, {30112, 40.25}}

var phi4MiniGreedy = []int64{976, 17307, 382, 9861, 5192, 316, 19781, 104153, 109445, 11, 1919, 37168, 9861, 183230, 328, 55414}

func TestPhi4MiniMatchesTransformersBF16(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (7.7 GB BF16 checkpoint)")
	}
	requireHostMemGB(t, 12)
	releaseModelMemory(t)
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--microsoft--Phi-4-mini-instruct/snapshots/*/model-*.safetensors")); len(m) < 2 {
		t.Skip(phi4MiniRepo + " not in HF cache")
	}
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	os.Setenv("HF_HUB_OFFLINE", "1")
	m, err := LoadCausalLMFromHF(phi4MiniRepo, dev, dtype.BF16)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cfg
	if c.NumLayers != 32 || c.HeadDim != 128 || c.PartialRotary != 0.75 || m.rotaryDim() != 96 || !c.TieEmbed || c.VocabSize != 200064 ||
		c.RopeScaling == nil || c.RopeScaling.Type != "longrope" || len(c.RopeScaling.ShortFactor) != 48 || c.RopeScaling.OriginalMaxPositions != 4096 {
		t.Fatalf("config %+v", c)
	}
	t.Logf("params %.3f B", float64(m.ParamCount())/1e9)
	row := lastLogits(t, m, phi4MiniChatIDs, NewKVCache(m.NumLayers()), dev)
	got := top5(row)
	// Reference top-3/4 ('Blue' / 'Light') tie at 40.5, so only the top-2
	// order is pinned; every top-5 logit is checked.
	for i, w := range phi4MiniRef {
		if i < 2 && got[i] != w.id {
			t.Errorf("top-%d: got id %d (%.3f), want %d (%.3f)", i+1, got[i], row[got[i]], w.id, w.logit)
		}
		if d := math.Abs(float64(row[w.id] - w.logit)); d > 1.0 {
			t.Errorf("logit[%d] = %.3f, want %.3f (BF16 tol 1.0)", w.id, row[w.id], w.logit)
		}
	}
	t.Logf("top-5 %v logits %.3f %.3f %.3f %.3f %.3f", got, row[got[0]], row[got[1]], row[got[2]], row[got[3]], row[got[4]])

	cache := NewKVCache(m.NumLayers())
	var out []int64
	step := lastLogits(t, m, phi4MiniChatIDs, cache, dev)
	for len(out) < len(phi4MiniGreedy) {
		next := int64(top5(step)[0])
		out = append(out, next)
		if next == 200020 || next == 199999 { // <|end|> / <|endoftext|>
			break
		}
		step = lastLogits(t, m, []int64{next}, cache, dev)
	}
	t.Logf("greedy %v", out)
	for i := range phi4MiniGreedy {
		if i >= len(out) || out[i] != phi4MiniGreedy[i] {
			t.Fatalf("greedy ids %v, want %v", out, phi4MiniGreedy)
		}
	}
}
