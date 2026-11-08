package models

import (
	"slices"
	"testing"

	"light/device"
	"light/dtype"
)

// Tiny-random Muse Glimmer written by `scripts/verify/ref_muse_glimmer.py
// --make-tiny`, with transformers 5.21 F32 eager reference values from
// `ref_muse_glimmer.py testdata/muse_glimmer/tiny f32 --ids …` (CPU). Layers
// sliding, sliding, sliding, full (NoPE); window 6 on an 18-token prompt;
// 4 × 24 heads on a 64-wide model; norm weights perturbed so the (1 + w)
// sandwich norms, the plain-w final norm and the weightless norms all matter.
const museGlimmerTinyDir = "testdata/muse_glimmer/tiny"

var (
	museGlimmerTinyIDs = []int64{5, 17, 300, 42, 8, 499, 13, 77, 256, 3, 64, 250, 19, 6, 400, 41, 2, 90}
	museGlimmerTinyRef = []struct {
		id    int
		logit float32
	}{{293, 0.1064}, {453, 0.0990}, {353, 0.0822}, {27, 0.0808}, {175, 0.0788}}
	museGlimmerTinyGreedy = []int64{293, 444, 229, 325, 409, 72, 7, 263}
)

func TestMuseGlimmerTinyMatchesTransformers(t *testing.T) {
	cpu := device.CPU()
	m, err := LoadMuseGlimmerFromHF(museGlimmerTinyDir, cpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cfg
	if !c.EmbedNorm || !c.AttnOutputGate || !c.SiLUMLP || !c.UntiedHead || c.QScale != 3.87 ||
		c.PostNormEps != 1e-8 || c.FinalLogitSoftcap != 20 || c.OutputMultiplier == 0 ||
		!slices.Equal(c.LayerRopeBase, []float64{10000, 10000, 10000, 0}) ||
		!slices.Equal(c.SlidingLayers, []bool{true, true, true, false}) || c.SlidingWindow != 6 ||
		c.NumHeads*c.HeadDim == c.HiddenSize {
		t.Fatalf("config %+v", c)
	}

	full := lastLogits(t, m, museGlimmerTinyIDs, NewKVCache(m.NumLayers()), cpu)
	got := top5(full)
	for i, w := range museGlimmerTinyRef {
		if i < 2 && got[i] != w.id {
			t.Errorf("top-%d: got %d (%.4f), want %d (%.4f)", i+1, got[i], full[got[i]], w.id, w.logit)
		}
		if d := full[w.id] - w.logit; d > 2e-4 || d < -2e-4 {
			t.Errorf("logit[%d] = %.5f, want %.4f", w.id, full[w.id], w.logit)
		}
	}

	// Greedy decode through the KV cache: the sliding layers trim to the
	// window, the NoPE layer keeps every key.
	cache := NewKVCache(m.NumLayers())
	step := lastLogits(t, m, museGlimmerTinyIDs, cache, cpu)
	var greedy []int64
	for len(greedy) < len(museGlimmerTinyGreedy) {
		next := int64(top5(step)[0])
		greedy = append(greedy, next)
		step = lastLogits(t, m, []int64{next}, cache, cpu)
	}
	if !slices.Equal(greedy, museGlimmerTinyGreedy) {
		t.Errorf("greedy %v, want %v", greedy, museGlimmerTinyGreedy)
	}

	// Chunked prefill + single steps == single shot.
	cache = NewKVCache(m.NumLayers())
	lastLogits(t, m, museGlimmerTinyIDs[:5], cache, cpu)
	var chunked []float32
	for i := 5; i < len(museGlimmerTinyIDs); i++ {
		chunked = lastLogits(t, m, museGlimmerTinyIDs[i:i+1], cache, cpu)
	}
	if d := maxAbsDiff(full, chunked); d > 1e-5 {
		t.Errorf("chunked vs single-shot differ by %.2e", d)
	}

	if !device.CUDAAvailable() {
		return
	}
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	mg, err := LoadMuseGlimmerFromHF(museGlimmerTinyDir, gpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	if d := maxAbsDiff(full, lastLogits(t, mg, museGlimmerTinyIDs, NewKVCache(mg.NumLayers()), gpu)); d > 1e-4 {
		t.Errorf("CUDA vs CPU logits differ by %.2e", d)
	}
}
