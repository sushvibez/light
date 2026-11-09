package models

import (
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
