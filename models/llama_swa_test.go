package models

import (
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"light/device"
	"light/tensor"
)

// qwen2ChatIDs is "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n".
var qwen2ChatIDs = []int64{151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198}

func qwen2Cached(t *testing.T) {
	t.Helper()
	home, _ := os.UserHomeDir()
	if _, err := os.Stat(filepath.Join(home, ".cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct")); err != nil {
		t.Skip("Qwen2-0.5B-Instruct not in HF cache")
	}
}

func lastLogits(t *testing.T, m CausalLM, ids []int64, cache *KVCache, dev device.Device) []float32 {
	t.Helper()
	x, err := tensor.New(ids, []int{1, len(ids)}, dev)
	if err != nil {
		t.Fatal(err)
	}
	logits, err := m.ForwardWithCache(x, cache)
	if err != nil {
		t.Fatal(err)
	}
	cpu, _ := logits.To(device.CPU())
	flat := tensor.MustSlice[float32](cpu)
	V := m.VocabSize()
	return flat[(len(ids)-1)*V : len(ids)*V]
}

func top5(row []float32) []int {
	idx := make([]int, len(row))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return row[idx[a]] > row[idx[b]] })
	return idx[:5]
}

// Reference: scripts/verify/ref_swa.py 8 (transformers F32, eager,
// use_sliding_window=True, sliding_window=8, max_window_layers=0).
func TestSlidingWindowMatchesTransformers(t *testing.T) {
	qwen2Cached(t)
	dev := device.CPU()
	m, err := LoadQwen2FromHF("Qwen/Qwen2-0.5B-Instruct", dev)
	if err != nil {
		t.Fatal(err)
	}
	m.Cfg.SlidingWindow = 8
	row := lastLogits(t, m, qwen2ChatIDs, NewKVCache(m.Cfg.NumLayers), dev)
	want := []struct {
		id    int
		logit float32
	}{{16, 13.7325}, {198, 13.5228}, {17, 13.2803}, {59, 13.1755}, {102349, 13.1741}}
	got := top5(row)
	for i, w := range want {
		if got[i] != w.id {
			t.Errorf("top-%d: got id %d (%.4f), want %d", i+1, got[i], row[got[i]], w.id)
		}
		if d := float64(row[w.id] - w.logit); math.Abs(d) > 5e-3 {
			t.Errorf("logit[%d] = %.4f, want %.4f", w.id, row[w.id], w.logit)
		}
	}
}

// The rotating cache must reproduce the single-shot sliding-window result
// when the prompt arrives in chunks, including one-token decode steps, and
// must never hold more than `window` keys per layer.
func TestSlidingWindowRotatingCache(t *testing.T) {
	qwen2Cached(t)
	testSlidingWindowRotatingCache(t, device.CPU())
}

func TestSlidingWindowRotatingCacheCUDA(t *testing.T) {
	qwen2Cached(t)
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	testSlidingWindowRotatingCache(t, dev)
}

func testSlidingWindowRotatingCache(t *testing.T, dev device.Device) {
	m, err := LoadQwen2FromHF("Qwen/Qwen2-0.5B-Instruct", dev)
	if err != nil {
		t.Fatal(err)
	}
	m.Cfg.SlidingWindow = 6
	full := lastLogits(t, m, qwen2ChatIDs, NewKVCache(m.Cfg.NumLayers), dev)

	cache := NewKVCache(m.Cfg.NumLayers)
	ids := qwen2ChatIDs
	lastLogits(t, m, ids[:5], cache, dev)
	lastLogits(t, m, ids[5:9], cache, dev)
	var step []float32
	for i := 9; i < len(ids); i++ {
		step = lastLogits(t, m, ids[i:i+1], cache, dev)
	}
	if cache.Len != len(ids) {
		t.Fatalf("cache.Len = %d, want %d", cache.Len, len(ids))
	}
	for l, k := range cache.K {
		if k.Dim(2) > 6 {
			t.Fatalf("layer %d holds %d keys, window is 6", l, k.Dim(2))
		}
	}
	for i := range full {
		if d := math.Abs(float64(full[i] - step[i])); d > 2e-3 {
			t.Fatalf("logit %d: full %.4f vs chunked %.4f", i, full[i], step[i])
		}
	}
	// And the window must actually bite: full causal attention differs.
	m.Cfg.SlidingWindow = 0
	causal := lastLogits(t, m, qwen2ChatIDs, NewKVCache(m.Cfg.NumLayers), dev)
	diff := 0.0
	for i := range causal {
		diff += math.Abs(float64(causal[i] - full[i]))
	}
	if diff/float64(len(causal)) < 1e-2 {
		t.Fatalf("sliding window had no effect (mean |Δ| = %.2e)", diff/float64(len(causal)))
	}
}
