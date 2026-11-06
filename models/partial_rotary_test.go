package models

import (
	"testing"

	"light/device"
	"light/dtype"
)

// Tiny random Llama with Phi-4-mini style RoPE (partial rotary 0.5 +
// longrope factor tables over the rotated dims; the prompt stays under
// original_max_position_embeddings, since crossing it re-tables RoPE in
// transformers too): chunked
// decode through the cache must equal the single-shot forward, CUDA must
// match CPU, and the partial setting must actually change the output.
func TestPartialRotaryConsistency(t *testing.T) {
	cfg := LlamaConfig{
		VocabSize: 64, HiddenSize: 32, IntermediateSize: 48, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		MaxPosition: 64, RopeBase: 1e4, RmsEps: 1e-5, PartialRotary: 0.5,
		RopeScaling: &RopeScaling{Type: "longrope", ShortFactor: []float64{1, 1.5}, LongFactor: []float64{2, 3}, OriginalMaxPositions: 32, MaxPositions: 64},
	}
	cpu := device.CPU()
	m, err := NewLlama(cfg, cpu)
	if err != nil {
		t.Fatal(err)
	}
	ids := []int64{3, 9, 27, 4, 8, 15, 16, 23, 42, 7, 11, 5}
	full := lastLogits(t, m, ids, NewKVCache(m.NumLayers()), cpu)
	cache := NewKVCache(m.NumLayers())
	lastLogits(t, m, ids[:5], cache, cpu)
	var step []float32
	for i := 5; i < len(ids); i++ {
		step = lastLogits(t, m, ids[i:i+1], cache, cpu)
	}
	if d := maxAbsDiff(full, step); d > 1e-4 {
		t.Fatalf("chunked vs single-shot differ by %.2e", d)
	}
	// Flipping to full rotary must change the logits (the pass-through dims
	// are really left alone).
	m.Cfg.PartialRotary = 0
	m.resetRoPETables()
	m.Cfg.RopeScaling = &RopeScaling{Type: "longrope", ShortFactor: []float64{1, 1.5, 1, 1}, LongFactor: []float64{2, 3, 1, 1}, OriginalMaxPositions: 32, MaxPositions: 64}
	if d := maxAbsDiff(full, lastLogits(t, m, ids, NewKVCache(m.NumLayers()), cpu)); d < 1e-3 {
		t.Fatalf("partial and full rotary agree to %.2e; partial rotary is not applied", d)
	}
	m.Cfg.PartialRotary, m.Cfg.RopeScaling = cfg.PartialRotary, cfg.RopeScaling
	m.resetRoPETables()
	if !device.CUDAAvailable() {
		return
	}
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	mg, err := NewLlama(LlamaConfig{VocabSize: cfg.VocabSize, HiddenSize: cfg.HiddenSize, IntermediateSize: cfg.IntermediateSize,
		NumLayers: cfg.NumLayers, NumHeads: cfg.NumHeads, NumKVHeads: cfg.NumKVHeads, HeadDim: cfg.HeadDim, MaxPosition: cfg.MaxPosition,
		RopeBase: cfg.RopeBase, RmsEps: cfg.RmsEps, PartialRotary: cfg.PartialRotary, RopeScaling: cfg.RopeScaling, SkipInit: true, WeightDType: dtype.F32}, gpu)
	if err != nil {
		t.Fatal(err)
	}
	pc, pg := m.Parameters(), mg.Parameters()
	for i := range pc {
		src, err := pc[i].To(gpu)
		if err != nil {
			t.Fatal(err)
		}
		if err := pg[i].CopyFrom(src); err != nil {
			t.Fatal(err)
		}
	}
	g := lastLogits(t, mg, ids, NewKVCache(mg.NumLayers()), gpu)
	if d := maxAbsDiff(full, g); d > 1e-3 {
		t.Fatalf("CPU vs CUDA differ by %.2e", d)
	}
}
