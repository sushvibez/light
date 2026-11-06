package models

import (
	"math"
	"math/rand"
	"testing"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// A tiny random Gemma 4 with every structural feature on (PLE, KV-shared
// layers, double-wide MLP, mixed head dims, partial RoPE, softcap): chunked
// decoding through the cache must reproduce the single-shot forward, also
// when the sliding window trims the cache — this exercises the shared-KV
// bookkeeping (source layer states, geometry) without real weights.
func TestGemma4CacheConsistency(t *testing.T) {
	for _, v := range []struct {
		name        string
		shared, ple int
		window      int
		partial     float64
	}{
		{"plain", 0, 0, 0, 1}, {"window", 0, 0, 5, 1}, {"partial", 0, 0, 0, 0.25}, {"ple", 0, 4, 0, 1},
		{"shared", 2, 0, 0, 1}, {"shared+window", 2, 0, 5, 1}, {"all", 2, 4, 5, 0.25},
	} {
		t.Run(v.name, func(t *testing.T) { gemma4Consistency(t, v.shared, v.ple, v.window, v.partial) })
	}
}

func gemma4Consistency(t *testing.T, shared, ple, window int, partial float64) {
	cfg := Gemma4Config{
		VocabSize: 97, HiddenSize: 32, IntermediateSize: 48, NumLayers: 6, NumHeads: 4, NumKVHeads: 2,
		HeadDim: 8, GlobalHeadDim: 16, MaxPosition: 128, RopeBase: 1e6, RopeLocalBase: 1e4, PartialRotary: 0.25,
		RmsEps: 1e-6, SlidingWindow: window, SlidingLayers: []bool{true, true, false, true, true, false},
		FinalLogitSoftcap: 30, PLEDim: ple, PLEVocab: 97, NumKVSharedLayers: shared, DoubleWideMLP: true,
	}
	cfg.PartialRotary = partial
	dev := device.CPU()
	m, err := NewGemma4(cfg, dev)
	if err != nil {
		t.Fatal(err)
	}
	if shared == 2 && (m.Cfg.kvSourceLayer(4) != 3 || m.Cfg.kvSourceLayer(5) != 2 || m.Blocks[4].KProj != nil || m.Blocks[3].KProj == nil) {
		t.Fatalf("kv sharing wiring wrong")
	}
	rng := rand.New(rand.NewSource(3))
	for _, p := range m.Parameters() {
		n := p.NumElements()
		data := make([]float32, n)
		scale := float32(0.2)
		if p.Rank() == 1 { // norm weights and layer scalars around 1
			for i := range data {
				data[i] = 1 + 0.1*float32(rng.NormFloat64())
			}
		} else {
			for i := range data {
				data[i] = scale * float32(rng.NormFloat64()) / float32(math.Sqrt(float64(p.Dim(p.Rank()-1))))
			}
		}
		src, _ := tensor.New(data, p.Shape().Dims(), dev)
		if err := p.CopyFrom(src); err != nil {
			t.Fatal(err)
		}
	}
	ids := []int64{5, 17, 42, 3, 96, 8, 8, 61, 22, 30, 1, 77}
	full := lastLogits(t, m, ids, NewKVCache(m.NumLayers()), dev)
	cache := NewKVCache(m.NumLayers())
	lastLogits(t, m, ids[:4], cache, dev)
	lastLogits(t, m, ids[4:9], cache, dev)
	var step []float32
	for i := 9; i < len(ids); i++ {
		step = lastLogits(t, m, ids[i:i+1], cache, dev)
	}
	for l, k := range cache.K {
		if m.Cfg.isKVShared(l) {
			if k != nil {
				t.Fatalf("shared layer %d must not own cache", l)
			}
			continue
		}
		if window > 0 && m.Cfg.isSliding(l) && k.Dim(2) > cfg.SlidingWindow {
			t.Fatalf("layer %d holds %d keys, window %d", l, k.Dim(2), cfg.SlidingWindow)
		}
		if !m.Cfg.isSliding(l) && k.Dim(2) != len(ids) {
			t.Fatalf("global layer %d holds %d keys", l, k.Dim(2))
		}
	}
	maxd := 0.0
	for i := range full {
		if d := math.Abs(float64(full[i] - step[i])); d > maxd {
			maxd = d
		}
	}
	if maxd > 1e-4 {
		t.Fatalf("chunked vs single-shot logits differ by %g", maxd)
	}
	if got := m.ParamCount(); got == 0 {
		t.Fatal("no parameters")
	}
	_ = dtype.F32
}
