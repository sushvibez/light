package models

import (
	"slices"
	"testing"

	"light/device"
	"light/dtype"
)

// Tiny-random Mellum 2.1 checkpoint written by
// `scripts/verify/ref_mellum.py --make-tiny`, with transformers F32 eager
// reference values from the same script (CPU). It mixes sliding (window 6)
// and full layers, YaRN on the full layers with a derived attention factor,
// a different RoPE base on the sliding layers and a dense MLP layer between
// sparse ones, so every Mellum-specific path is exercised by an 18-token
// prompt and an 8-token greedy continuation.
const mellumTinyDir = "testdata/mellum/tiny"

var (
	mellumTinyIDs = []int64{5, 17, 300, 42, 8, 499, 13, 77, 256, 3, 64, 250, 19, 6, 400, 41, 2, 90}
	mellumTinyRef = []struct {
		id    int
		logit float32
	}{{225, 0.4841}, {10, 0.4689}, {190, 0.4524}, {104, 0.4511}, {159, 0.4444}}
	mellumTinyGreedy = []int64{225, 351, 504, 254, 254, 377, 255, 28}
)

func TestMellumTinyMatchesTransformers(t *testing.T) {
	cpu := device.CPU()
	m, err := LoadCausalLMFromHF(mellumTinyDir, cpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cfg
	if !c.QKNorm || c.NumExperts != 8 || c.NumExpertsPerTok != 2 || !c.NormTopKProb ||
		c.SlidingWindow != 6 || !slices.Equal(c.SlidingLayers, []bool{true, false, true, false}) ||
		!slices.Equal(c.MLPOnlyLayers, []int{1}) || c.IsMoELayer(1) || !c.IsMoELayer(0) ||
		c.RopeBase != 500000 || c.RopeScaling == nil || c.RopeScaling.Type != "yarn" || c.RopeScaling.Factor != 4 ||
		c.RopeScaling.OriginalMaxPositions != 64 || c.SlidingRopeBase != 10000 || c.SlidingRopeScaling != nil {
		t.Fatalf("config %+v (rope scaling %+v)", c, c.RopeScaling)
	}

	full := lastLogits(t, m, mellumTinyIDs, NewKVCache(m.NumLayers()), cpu)
	got := top5(full)
	for i, w := range mellumTinyRef {
		if i < 3 && got[i] != w.id {
			t.Errorf("top-%d: got %d (%.4f), want %d (%.4f)", i+1, got[i], full[got[i]], w.id, w.logit)
		}
		if d := full[w.id] - w.logit; d > 2e-3 || d < -2e-3 {
			t.Errorf("logit[%d] = %.4f, want %.4f", w.id, full[w.id], w.logit)
		}
	}

	// Greedy decode through the KV cache runs well past the sliding window.
	cache := NewKVCache(m.NumLayers())
	step := lastLogits(t, m, mellumTinyIDs, cache, cpu)
	var greedy []int64
	for len(greedy) < len(mellumTinyGreedy) {
		next := int64(top5(step)[0])
		greedy = append(greedy, next)
		step = lastLogits(t, m, []int64{next}, cache, cpu)
	}
	if !slices.Equal(greedy, mellumTinyGreedy) {
		t.Errorf("greedy %v, want %v", greedy, mellumTinyGreedy)
	}

	// Chunked prefill + single-token steps == single shot.
	cache = NewKVCache(m.NumLayers())
	lastLogits(t, m, mellumTinyIDs[:5], cache, cpu)
	var chunked []float32
	for i := 5; i < len(mellumTinyIDs); i++ {
		chunked = lastLogits(t, m, mellumTinyIDs[i:i+1], cache, cpu)
	}
	if d := maxAbsDiff(full, chunked); d > 1e-4 {
		t.Errorf("chunked vs single-shot differ by %.2e", d)
	}

	// The per-layer-type RoPE and YaRN must actually be applied: dropping
	// either changes the logits.
	for _, tc := range []struct {
		name  string
		apply func()
	}{
		{"sliding layers on the full-layer base", func() { m.Cfg.SlidingRopeBase = 0 }},
		{"no YaRN on full layers", func() { m.Cfg.RopeScaling = nil }},
	} {
		saved := m.Cfg
		tc.apply()
		m.resetRoPETables()
		if d := maxAbsDiff(full, lastLogits(t, m, mellumTinyIDs, NewKVCache(m.NumLayers()), cpu)); d < 1e-4 {
			t.Errorf("%s: logits unchanged (max diff %.2e); the scheme is not applied", tc.name, d)
		}
		m.Cfg = saved
		m.resetRoPETables()
	}

	if !device.CUDAAvailable() {
		return
	}
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	mg, err := LoadCausalLMFromHF(mellumTinyDir, gpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	if d := maxAbsDiff(full, lastLogits(t, mg, mellumTinyIDs, NewKVCache(mg.NumLayers()), gpu)); d > 1e-3 {
		t.Errorf("CUDA vs CPU logits differ by %.2e", d)
	}
}

// transformers 5.21 ROPE_INIT_FUNCTIONS["yarn"] on the tiny checkpoint's
// config, layer_type="full_attention".
const (
	yarnRef0    = 1.0
	yarnRef1    = 0.12120172381401062
	yarnRef2    = 0.009401507675647736
	yarnRef3    = 0.001823166268877685
	yarnRef4    = 0.0003535533614922315
	yarnRef5    = 6.856204709038138e-05
	yarnRef6    = 1.3295739336172119e-05
	yarnRef7    = 2.578346311565838e-06
	yarnRefAttn = 1.138629436111989
)

// TestYarnInvFreqMatchesTransformers pins YarnInvFreq and the derived
// attention factor to transformers' _compute_yarn_parameters for the tiny
// checkpoint's full-attention layers (head_dim 16, θ 5e5, factor 4,
// original context 64).
func TestYarnInvFreqMatchesTransformers(t *testing.T) {
	s := RopeScaling{Type: "yarn", Factor: 4, OriginalMaxPositions: 64, BetaFast: 32, BetaSlow: 1}
	got := YarnInvFreq(16, 500000, s)
	want := []float32{yarnRef0, yarnRef1, yarnRef2, yarnRef3, yarnRef4, yarnRef5, yarnRef6, yarnRef7}
	for i := range want {
		if d := float64(got[i]-want[i]) / float64(want[i]); d > 1e-6 || d < -1e-6 {
			t.Errorf("invFreq[%d] = %.9g, want %.9g", i, got[i], want[i])
		}
	}
	if af := s.AttentionFactor(); af < yarnRefAttn-1e-9 || af > yarnRefAttn+1e-9 {
		t.Errorf("attention factor %.12f, want %.12f", af, yarnRefAttn)
	}
	// An explicit attention_factor (Mellum 2.1's real config) wins.
	s.AttnFactor = 1.2772588722239782
	if af := s.AttentionFactor(); af != 1.2772588722239782 {
		t.Errorf("explicit attention factor %.12f ignored", af)
	}
}
