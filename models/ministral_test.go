package models

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
)

// Tiny-random Ministral 3 from `scripts/verify/ref_ministral.py
// --make-tiny`, laid out like mistralai/Ministral-3-3B-Instruct-2512: a
// mistral3 wrapper (tiny Pixtral tower, skipped), `language_model.model.*`
// names, per-tensor FP8 linears (BF16 weight_scale_inv / activation_scale)
// and YaRN RoPE with llama_4_scaling_beta 0.1 over an
// original_max_position_embeddings of 8, so the Llama 4 query scaling is
// live from position 8 of the 18-token prompt. Reference from
// `ref_ministral.py testdata/ministral/tiny f32 --ids …` (transformers
// 5.21 dequantizes the FP8 weights to BF16 on the CPU, then .float()).
func TestMinistralTinyMatchesTransformers(t *testing.T) {
	ids := []int64{5, 17, 300, 42, 8, 499, 13, 77, 256, 3, 64, 250, 19, 6, 400, 41, 2, 90}
	ref := []struct {
		id    int
		logit float32
	}{{90, 1.0388}, {457, 0.4979}, {491, 0.4756}, {479, 0.4706}, {332, 0.3981}}
	greedyRef := []int64{90, 90, 90, 90, 90, 90, 90, 90}
	cpu := device.CPU()
	m, err := LoadCausalLMFromHF("testdata/ministral/tiny", cpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cfg
	if c.Llama4ScalingBeta != 0.1 || c.Llama4ScalingOrig != 8 || c.RopeScaling == nil || c.RopeScaling.Type != "yarn" ||
		c.RopeScaling.Factor != 16 || c.RopeBase != 1e6 || c.HeadDim != 16 || c.NumKVHeads != 2 || !c.TieEmbed {
		t.Fatalf("config %+v (rope scaling %+v)", c, c.RopeScaling)
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
	var gpu device.Device
	if device.CUDAAvailable() {
		if gpu, err = device.CUDA(0); err != nil {
			t.Fatal(err)
		}
	} else if g, err := device.WebGPU(0); err == nil {
		gpu = g
	}
	if gpu == nil {
		return
	}
	mg, err := LoadCausalLMFromHF("testdata/ministral/tiny", gpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	if d := maxAbsDiff(full, lastLogits(t, mg, ids, NewKVCache(mg.NumLayers()), gpu)); d > 1e-4 {
		t.Errorf("%v vs CPU logits differ by %.2e", gpu, d)
	}
}

// Ministral-3-3B-Instruct-2512 (FP8 checkpoint, dequantized to BF16 at
// load) in BF16 against transformers 5.21 BF16 on the CPU (which also
// dequantizes; `ref_ministral.py mistralai/Ministral-3-3B-Instruct-2512 bf16
// --chat`): top-5 logits and 16 greedy tokens ("The sky appears blue due to
// **Rayleigh scattering**, where shorter blue wavelengths of"). CUDA when
// present, else the CPU.
func TestMinistral3BMatchesTransformersBF16(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (4.4 GB FP8 checkpoint, 7.7 GB in BF16)")
	}
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--mistralai--Ministral-3-3B-Instruct-2512/snapshots/*/model.safetensors")); len(m) == 0 {
		t.Skip("mistralai/Ministral-3-3B-Instruct-2512 not in HF cache")
	}
	requireHostMemGB(t, 12)
	releaseModelMemory(t)
	ids := []int64{1, 17, 4568, 1584, 1261, 104335, 27089, 1046, 18, 3, 2763, 44122, 1294, 1925, 19286, 5782, 1278, 21283, 1395, 10991, 1046, 4}
	ref := []struct {
		id    int
		logit float32
	}{{1784, 18.8750}, {5492, 17.0000}, {99701, 16.5000}, {27236, 16.5000}, {47451, 16.0000}}
	greedyRef := []int64{1784, 21283, 10249, 10991, 3964, 1317, 1603, 51199, 70972, 35805, 6143, 2478, 24701, 10991, 63233, 1307}
	dev := device.CPU()
	if device.CUDAAvailable() {
		g, err := device.CUDA(0)
		if err != nil {
			t.Fatal(err)
		}
		dev = g
	}
	m, err := LoadCausalLMFromHF("mistralai/Ministral-3-3B-Instruct-2512", dev, dtype.BF16)
	if err != nil {
		t.Fatal(err)
	}
	if c := m.Cfg; c.NumLayers != 26 || c.HeadDim != 128 || c.Llama4ScalingBeta != 0.1 || c.Llama4ScalingOrig != 16384 || c.RopeScaling == nil || c.RopeScaling.Type != "yarn" {
		t.Fatalf("config %+v", c)
	}
	cache := NewKVCache(m.NumLayers())
	row := lastLogits(t, m, ids, cache, dev)
	got := top5(row)
	if got[0] != ref[0].id {
		t.Errorf("top-1: got %d (%.3f), want %d (%.3f)", got[0], row[got[0]], ref[0].id, ref[0].logit)
	}
	for _, w := range ref {
		if d := row[w.id] - w.logit; d > 0.5 || d < -0.5 {
			t.Errorf("logit[%d] = %.3f, want %.3f (transformers BF16; tol 0.5)", w.id, row[w.id], w.logit)
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
