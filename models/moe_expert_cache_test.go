package models

import (
	"math"
	"testing"

	"light/device"
	"light/nn"
	"light/tensor"
)

// streamedPrompt is 14 tokens (every tiny vocab has ≥ 256 ids): a 9-token
// prefill touches every routed expert, decode steps cycle 2 slots.
var streamedPrompt = []int64{3, 77, 150, 9, 201, 42, 18, 250, 5, 120, 66, 199, 31, 88}

// checkStreamedMoE runs the same tokens through a resident model (ref) and
// one with streamed experts (str): the whole prompt without a cache, then
// a 9-token prefill and single-token decode steps through each model's
// cache. Returns the largest logit difference (streaming changes only where
// the expert weights come from, so it should be 0).
func checkStreamedMoE[C any](t *testing.T, ref, str func(*tensor.Tensor, C) (*tensor.Tensor, error), newRef, newStr func() C) float64 {
	t.Helper()
	run := func(f func(*tensor.Tensor, C) (*tensor.Tensor, error), ids []int64, c C) []float32 {
		x, err := tensor.New(ids, []int{1, len(ids)}, device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		out, err := f(x, c)
		if err != nil {
			t.Fatal(err)
		}
		return tensor.MustSlice[float32](out)
	}
	md := 0.0
	cmp := func(a, b []float32) {
		if len(a) != len(b) {
			t.Fatalf("%d logits vs %d", len(a), len(b))
		}
		for i := range a {
			d := math.Abs(float64(a[i] - b[i]))
			if d != d {
				d = math.Inf(1)
			}
			md = math.Max(md, d)
		}
	}
	var none C
	cmp(run(ref, streamedPrompt, none), run(str, streamedPrompt, none))
	const prefill = 9
	cr, cs := newRef(), newStr()
	cmp(run(ref, streamedPrompt[:prefill], cr), run(str, streamedPrompt[:prefill], cs))
	for i := prefill; i < len(streamedPrompt); i++ {
		cmp(run(ref, streamedPrompt[i:i+1], cr), run(str, streamedPrompt[i:i+1], cs))
	}
	return md
}

// checkExpertCaches sums the streamed layers' counters and checks that no
// layer holds more than slots experts and that decode prefetched.
func checkExpertCaches[V any](t *testing.T, caches []*nn.ExpertCache[V], slots int, md float64) {
	t.Helper()
	if len(caches) == 0 {
		t.Fatal("no streamed layers")
	}
	var loads, hits, pf, used int64
	for i, c := range caches {
		if n := len(c.Resident()); n > slots {
			t.Errorf("streamed layer %d holds %d experts over %d slots", i, n, slots)
		}
		loads, hits, pf, used = loads+c.Loads, hits+c.Hits, pf+c.Prefetched, used+c.PrefetchUsed
	}
	t.Logf("%d streamed layers, %d slots: max diff %.3g; %d loads, %d hits, %d prefetched, %d of them used", len(caches), slots, md, loads, hits, pf, used)
	if md != 0 {
		t.Errorf("streamed logits differ by %.3g", md)
	}
	if pf == 0 || used > pf {
		t.Errorf("prefetch counters: %d prefetched, %d used", pf, used)
	}
}

// GLM-5.3 with its routed experts streamed through 2 slots per layer
// (LIGHT_EXPERT_SLOTS) equals the resident model, for the fused F32 and the
// per-expert block-scaled FP8 checkpoints.
func TestGLM5StreamedExperts(t *testing.T) {
	for _, name := range []string{"glm5-tiny", "glm5-fp8-tiny"} {
		t.Run(name, func(t *testing.T) {
			ref, err := LoadGLM5("testdata/"+name, device.CPU())
			if err != nil {
				t.Fatal(err)
			}
			t.Setenv("LIGHT_EXPERT_SLOTS", "2")
			m, err := LoadGLM5("testdata/"+name, device.CPU())
			if err != nil {
				t.Fatal(err)
			}
			var caches []*nn.ExpertCache[*glm5Expert]
			var sparse []*glm5Layer
			for i, ly := range m.Layers {
				if ly.RouterW == nil {
					continue
				}
				if ly.Stream == nil || ly.Experts != nil {
					t.Fatalf("layer %d: experts not streamed", i)
				}
				caches, sparse = append(caches, ly.Stream), append(sparse, ly)
			}
			if len(sparse) < 2 || sparse[0].next != sparse[1] {
				t.Fatal("streamed layers not linked for prefetch")
			}
			md := checkStreamedMoE(t, ref.ForwardWithCache, m.ForwardWithCache, ref.NewCache, m.NewCache)
			checkExpertCaches(t, caches, 2, md)
		})
	}
}

// DeepSeek-V4.1 with its FP4 routed experts streamed through 2 slots per
// layer equals the model that keeps every expert once loaded.
func TestDeepseekV41StreamedExperts(t *testing.T) {
	ref, err := LoadDeepseekV41("testdata/deepseek-v41-tiny", device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("LIGHT_EXPERT_SLOTS", "2")
	m, err := LoadDeepseekV41("testdata/deepseek-v41-tiny", device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	var caches []*nn.ExpertCache[*v41Expert]
	for i, ly := range m.Layers {
		if ly.Experts.Slots != 2 || ref.Layers[i].Experts.Slots != ref.Cfg.NRoutedExperts {
			t.Fatalf("layer %d: %d slots streamed, %d resident", i, ly.Experts.Slots, ref.Layers[i].Experts.Slots)
		}
		caches = append(caches, ly.Experts)
	}
	if m.Layers[0].next != m.Layers[1] {
		t.Fatal("layers not linked for prefetch")
	}
	md := checkStreamedMoE(t, ref.ForwardWithCache, m.ForwardWithCache, ref.NewCache, m.NewCache)
	checkExpertCaches(t, caches, 2, md)
}
