package models

import (
	"os"
	"path/filepath"
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

// granite-4.0-micro (3B, attention-only granitemoehybrid) in BF16 against
// transformers 5.21 BF16 (`ref_granite.py ibm-granite/granite-4.0-micro bf16
// --chat --cpu`): the validation-bar chat prompt, top-5 last-position logits
// (logits ÷ 10 already applied) and 16 greedy tokens ("The sky appears blue
// due to the scattering of sunlight by the Earth's atmosphere,"). Runs on
// CUDA when present, else on the CPU.
func TestGraniteMicroMatchesTransformersBF16(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (6.3 GB BF16 checkpoint)")
	}
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--ibm-granite--granite-4.0-micro/snapshots/*/model-*.safetensors")); len(m) == 0 {
		t.Skip("ibm-granite/granite-4.0-micro not in HF cache")
	}
	requireHostMemGB(t, 10)
	releaseModelMemory(t)
	ids := []int64{100264, 9125, 100265, 2675, 527, 264, 64694, 18328, 13, 100257, 198, 100264, 882, 100265, 849, 21435, 304, 832, 11914, 3249, 279, 13180, 374, 6437, 13, 100257, 198, 100264, 78191, 100265}
	ref := []struct {
		id    int
		logit float32
	}{{791, 35.5000}, {31192, 26.7500}, {47048, 26.0000}, {14235, 24.1250}, {10544, 23.6250}}
	greedyRef := []int64{791, 13180, 8111, 6437, 4245, 311, 279, 72916, 315, 40120, 555, 279, 9420, 596, 16975, 11}
	dev := device.CPU()
	if device.CUDAAvailable() {
		g, err := device.CUDA(0)
		if err != nil {
			t.Fatal(err)
		}
		dev = g
	}
	m, err := LoadCausalLMFromHF("ibm-granite/granite-4.0-micro", dev, dtype.BF16)
	if err != nil {
		t.Fatal(err)
	}
	cache := NewKVCache(m.NumLayers())
	row := lastLogits(t, m, ids, cache, dev)
	got := top5(row)
	for i, w := range ref {
		if i < 3 && got[i] != w.id {
			t.Errorf("top-%d: got %d (%.3f), want %d (%.3f)", i+1, got[i], row[got[i]], w.id, w.logit)
		}
		if d := row[w.id] - w.logit; d > 0.75 || d < -0.75 {
			t.Errorf("logit[%d] = %.3f, want %.3f (transformers BF16; tol 0.75)", w.id, row[w.id], w.logit)
		}
	}
	var greedy []int64
	for len(greedy) < len(greedyRef) {
		next := int64(top5(row)[0])
		greedy = append(greedy, next)
		row = lastLogits(t, m, []int64{next}, cache, dev)
	}
	agree := 0
	for agree < len(greedy) && greedy[agree] == greedyRef[agree] {
		agree++
	}
	t.Logf("greedy %v (first %d of %d agree with transformers BF16)", greedy, agree, len(greedy))
	if agree < 12 {
		t.Errorf("greedy %v diverges from transformers BF16 %v after %d tokens", greedy, greedyRef, agree)
	}
}
