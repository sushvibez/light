package models

import (
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
)

// Mistral-7B-Instruct-v0.3 (llama-arch `mistral`, GQA 32 / 8, no sliding
// window in v0.3, rope θ 1e6, vocab 32768) in BF16 against transformers
// 5.21 BF16 (`ref_mellum.py mistralai/Mistral-7B-Instruct-v0.3 bf16 --chat
// --cpu`, eager): the validation-bar chat prompt rendered by the repo's
// template, top-5 last-position logits and 16 greedy tokens ("The sky
// appears blue because molecules in the Earth's atmosphere scatter sunlight
// in"). Runs on CUDA when present, else on the CPU (14.5 GB of BF16
// weights; the F32 reference does not fit a 31 GB laptop).
var (
	mistral7BChatIDs = []int64{1, 3, 1763, 1228, 1032, 3846, 1632, 14660, 29491, 781, 781, 1734, 20225, 1065, 1392, 13039, 2847, 1040, 7980, 1117, 5813, 29491, 4}
	mistral7BRef     = []struct {
		id    int
		logit float32
	}{{1183, 21.3750}, {10299, 16.0000}, {3290, 15.3125}, {7825, 14.1250}, {6286, 13.1250}}
	mistral7BGreedy = []int64{1183, 7980, 8813, 5813, 1864, 12928, 22417, 1065, 1040, 9367, 29510, 29481, 14557, 11958, 23718, 1065}
)

func TestMistral7BMatchesTransformersBF16(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (14.5 GB BF16 checkpoint)")
	}
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/*/model-*.safetensors")); len(m) < 3 {
		t.Skip("mistralai/Mistral-7B-Instruct-v0.3 not in HF cache")
	}
	requireHostMemGB(t, 18)
	releaseModelMemory(t)
	dev := device.CPU()
	if device.CUDAAvailable() {
		requireGPUMemGB(t, 0, 16)
		g, err := device.CUDA(0)
		if err != nil {
			t.Fatal(err)
		}
		dev = g
	}
	m, err := LoadCausalLMFromHF("mistralai/Mistral-7B-Instruct-v0.3", dev, dtype.BF16)
	if err != nil {
		t.Fatal(err)
	}
	if c := m.Cfg; c.NumKVHeads != 8 || c.RopeBase != 1e6 || c.VocabSize != 32768 || c.SlidingWindow != 0 {
		t.Fatalf("config %+v", c)
	}
	cache := NewKVCache(m.NumLayers())
	row := lastLogits(t, m, mistral7BChatIDs, cache, dev)
	got := top5(row)
	for i, w := range mistral7BRef {
		if i < 2 && got[i] != w.id {
			t.Errorf("top-%d: got %d (%.3f), want %d (%.3f)", i+1, got[i], row[got[i]], w.id, w.logit)
		}
		if d := row[w.id] - w.logit; d > 0.5 || d < -0.5 {
			t.Errorf("logit[%d] = %.3f, want %.3f (transformers BF16; tol 0.5)", w.id, row[w.id], w.logit)
		}
	}
	var greedy []int64
	for len(greedy) < len(mistral7BGreedy) {
		next := int64(top5(row)[0])
		greedy = append(greedy, next)
		row = lastLogits(t, m, []int64{next}, cache, dev)
	}
	agree := 0
	for agree < len(greedy) && greedy[agree] == mistral7BGreedy[agree] {
		agree++
	}
	t.Logf("greedy %v (first %d of %d agree with transformers BF16)", greedy, agree, len(greedy))
	if agree < 12 {
		t.Errorf("greedy %v diverges from transformers BF16 %v after %d tokens", greedy, mistral7BGreedy, agree)
	}
}
