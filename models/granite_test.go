package models

import (
	"slices"
	"testing"

	"light/device"
	"light/dtype"
)

// Tiny-random Granite 4 (granitemoehybrid, attention-only, no routed
// experts) from `scripts/verify/ref_granite.py --make-tiny`; reference from
// `ref_granite.py testdata/granite/tiny f32 --cpu --ids …` (transformers
// 5.21, eager): embedding × 12, attention scale 1/16 (not 1/√16), residual
// branches × 0.22, logits ÷ 10, fused shared MLP, RoPE θ 1e7, GQA 4 / 2,
// perturbed norms, tied head.
func TestGraniteTinyMatchesTransformers(t *testing.T) {
	ids := []int64{5, 17, 300, 42, 8, 499, 13, 77, 256, 3, 64, 250, 19, 6, 400, 41, 2, 90}
	ref := []struct {
		id    int
		logit float32
	}{{90, 0.1594}, {457, 0.0445}, {477, 0.0410}, {342, 0.0406}, {341, 0.0372}}
	greedyRef := []int64{90, 90, 90, 90, 90, 90, 90, 90}
	cpu := device.CPU()
	m, err := LoadCausalLMFromHF("testdata/granite/tiny", cpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cfg
	if c.EmbedMultiplier != 12 || c.AttnScale != 0.0625 || c.ResidualMultiplier != 0.22 || c.LogitsScaling != 10 ||
		c.IntermediateSize != 96 || c.RopeBase != 1e7 || !c.TieEmbed {
		t.Fatalf("config %+v", c)
	}
	full := lastLogits(t, m, ids, NewKVCache(m.NumLayers()), cpu)
	got := top5(full)
	for i, w := range ref {
		if i < 3 && got[i] != w.id {
			t.Errorf("top-%d: got %d (%.4f), want %d (%.4f)", i+1, got[i], full[got[i]], w.id, w.logit)
		}
		if d := full[w.id] - w.logit; d > 2e-4 || d < -2e-4 {
			t.Errorf("logit[%d] = %.5f, want %.4f", w.id, full[w.id], w.logit)
		}
	}
	cache := NewKVCache(m.NumLayers())
	step := lastLogits(t, m, ids, cache, cpu)
	var greedy []int64
	for len(greedy) < len(greedyRef) {
		next := int64(top5(step)[0])
		greedy = append(greedy, next)
		step = lastLogits(t, m, []int64{next}, cache, cpu)
	}
	if !slices.Equal(greedy, greedyRef) {
		t.Errorf("greedy %v, want %v", greedy, greedyRef)
	}
	if !device.CUDAAvailable() {
		if g, err := device.WebGPU(0); err == nil {
			mg, err := LoadCausalLMFromHF("testdata/granite/tiny", g, dtype.F32)
			if err != nil {
				t.Fatal(err)
			}
			if d := maxAbsDiff(full, lastLogits(t, mg, ids, NewKVCache(mg.NumLayers()), g)); d > 1e-4 {
				t.Errorf("WebGPU vs CPU logits differ by %.2e", d)
			}
		}
		return
	}
	g, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	mg, err := LoadCausalLMFromHF("testdata/granite/tiny", g, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	if d := maxAbsDiff(full, lastLogits(t, mg, ids, NewKVCache(mg.NumLayers()), g)); d > 1e-4 {
		t.Errorf("CUDA vs CPU logits differ by %.2e", d)
	}
}
