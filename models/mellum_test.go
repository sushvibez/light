package models

import (
	"os"
	"path/filepath"
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
// prompt and an 8-token greedy continuation. Heads are 4 × 24 = 96 wide on
// a 64-wide model: like the real 12B (32 × 128 on 2304), the attention
// width differs from the hidden size.
const mellumTinyDir = "testdata/mellum/tiny"

var (
	mellumTinyIDs = []int64{5, 17, 300, 42, 8, 499, 13, 77, 256, 3, 64, 250, 19, 6, 400, 41, 2, 90}
	mellumTinyRef = []struct {
		id    int
		logit float32
	}{{420, 0.4712}, {187, 0.4264}, {451, 0.3898}, {282, 0.3868}, {262, 0.3738}}
	mellumTinyGreedy = []int64{420, 462, 93, 511, 15, 53, 367, 183}
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

// Mellum 2.1 12B-A2.5B (Thinking), BF16 on CUDA, against transformers 5.21
// BF16 eager on an A100: `ref_mellum.py JetBrains/Mellum2.1-12B-A2.5B-Thinking
// bf16 --chat` (system "You are a concise assistant.", user "Explain in one
// sentence why the sky is blue.", generation prompt). The model opens with
// <think> by a margin of 28 logits, so the top-1 check is easy; the other
// four logits (near -20, BF16 step 0.125) and the 16-token greedy
// continuation are what pin the sliding / YaRN / MoE paths down. 24.3 GB
// of weights: needs an A100-class GPU (an L4's 22 GiB is too small).
const mellum12BRepo = "JetBrains/Mellum2.1-12B-A2.5B-Thinking"

var (
	mellum12BChatIDs = []int64{27, 5217, 233, 2591, 445, 255, 99, 22810, 16833, 48, 28, 233, 27, 1397, 233,
		41770, 325, 768, 7505, 3306, 302, 11734, 359, 5786, 48, 28, 233, 27, 8091, 233}
	mellum12BRef = []struct {
		id    int
		logit float32
	}{{23, 8.4375}, {629, -19.75}, {15114, -20.25}, {91657, -21.125}, {28, -21.5}}
	// "<think>\nWe need to respond with one sentence explaining why sky is blue. Concise"
	mellum12BGreedy = []int64{23, 233, 2116, 895, 333, 6246, 434, 768, 7505, 18469, 3306, 11734, 359, 5786, 48, 94113}
)

func TestMellum12BMatchesTransformersBF16(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (24.3 GB BF16 checkpoint)")
	}
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--JetBrains--Mellum2.1-12B-A2.5B-Thinking/snapshots/*/model*.safetensors")); len(m) == 0 {
		t.Skip(mellum12BRepo + " not in HF cache")
	}
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	requireHostMemGB(t, 30)
	requireGPUMemGB(t, 0, 27)
	releaseModelMemory(t)
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	os.Setenv("HF_HUB_OFFLINE", "1")
	m, err := LoadCausalLMFromHF(mellum12BRepo, dev, dtype.BF16)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cfg
	sliding := 0
	for _, s := range c.SlidingLayers {
		if s {
			sliding++
		}
	}
	if !c.QKNorm || c.NumExperts == 0 || c.NumExpertsPerTok == 0 || c.SlidingWindow == 0 || sliding == 0 ||
		sliding == len(c.SlidingLayers) || c.RopeScaling == nil || c.RopeScaling.Type != "yarn" {
		t.Fatalf("config %+v (rope scaling %+v)", c, c.RopeScaling)
	}
	t.Logf("params %.2f B, %d layers (%d sliding, window %d), %d experts top-%d, dense MLP layers %v",
		float64(m.ParamCount())/1e9, len(c.SlidingLayers), sliding, c.SlidingWindow, c.NumExperts, c.NumExpertsPerTok, c.MLPOnlyLayers)

	cache := NewKVCache(m.NumLayers())
	row := lastLogits(t, m, mellum12BChatIDs, cache, dev)
	got := top5(row)
	if got[0] != mellum12BRef[0].id {
		t.Errorf("top-1: got %d (%.3f), want %d", got[0], row[got[0]], mellum12BRef[0].id)
	}
	for _, w := range mellum12BRef {
		if d := row[w.id] - w.logit; d > 0.5 || d < -0.5 {
			t.Errorf("logit[%d] = %.3f, want %.3f (BF16 tol 0.5)", w.id, row[w.id], w.logit)
		}
	}
	t.Logf("top-5 %v logits %.3f %.3f %.3f %.3f %.3f", got, row[got[0]], row[got[1]], row[got[2]], row[got[3]], row[got[4]])

	var greedy []int64
	for len(greedy) < len(mellum12BGreedy) {
		next := int64(top5(row)[0])
		greedy = append(greedy, next)
		row = lastLogits(t, m, []int64{next}, cache, dev)
	}
	agree := 0
	for agree < len(greedy) && greedy[agree] == mellum12BGreedy[agree] {
		agree++
	}
	t.Logf("greedy %v (first %d of %d agree with transformers BF16)", greedy, agree, len(greedy))
	if agree < 8 {
		t.Errorf("greedy %v diverges from transformers BF16 %v after %d tokens", greedy, mellum12BGreedy, agree)
	}
}
