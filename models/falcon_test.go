package models

import (
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
)

// Tiny-random Falcon checkpoints, one per decoder variant, with transformers
// F32 reference logits from scripts/verify/ref_falcon.py. The ALiBi case uses
// attn_implementation="sdpa": transformers' eager Falcon adds the ALiBi bias
// twice (inside the attention and again folded into the causal mask), while
// the SDPA path, the original Falcon code and Light add it once.
//
//	testdata/falcon/mq7b   rotary + multi_query + parallel block, no bias (falcon-7b layout)
//	testdata/falcon/alibi  ALiBi + multi-head + sequential block, biases (falcon-rw-1b layout)
//	hf-internal-testing/tiny-random-FalconForCausalLM  new decoder architecture (grouped K/V, ln_attn + ln_mlp)
var falconTinyIDs = []int64{5, 17, 300, 42, 8, 999, 13, 77, 512, 3, 64, 250, 19, 6, 1000, 41, 2, 90}

type falconTinyCase struct {
	name, repo string
	ref        []struct {
		id    int
		logit float32
	}
}

var falconTinyCases = []falconTinyCase{
	{"mq7b", "testdata/falcon/mq7b", []struct {
		id    int
		logit float32
	}{{90, 0.6212}, {160, 0.5033}, {604, 0.4586}, {898, 0.4495}, {810, 0.4407}}},
	{"alibi", "testdata/falcon/alibi", []struct {
		id    int
		logit float32
	}{{918, 0.5195}, {90, 0.4580}, {503, 0.4341}, {468, 0.4253}, {701, 0.4085}}},
	{"newarch", "hf-internal-testing/tiny-random-FalconForCausalLM", []struct {
		id    int
		logit float32
	}{{261, 0.3429}, {926, 0.3381}, {1017, 0.3163}, {203, 0.3007}, {90, 0.2849}}},
}

func falconTinyLoad(t *testing.T, c falconTinyCase, dev device.Device) *Falcon {
	t.Helper()
	if c.repo[0] != 't' {
		home, _ := os.UserHomeDir()
		if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--hf-internal-testing--tiny-random-FalconForCausalLM/snapshots/*/model.safetensors")); len(m) == 0 {
			t.Skip(c.repo + " not in HF cache")
		}
		os.Setenv("HF_HUB_OFFLINE", "1")
	}
	m, err := LoadFalconFromHF(c.repo, dev, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestFalconTinyVariantsMatchTransformers(t *testing.T) {
	cpu := device.CPU()
	for _, c := range falconTinyCases {
		t.Run(c.name, func(t *testing.T) {
			m := falconTinyLoad(t, c, cpu)
			switch c.name {
			case "mq7b":
				if m.Cfg.Alibi || !m.Cfg.MultiQuery || !m.Cfg.ParallelAttn || m.Cfg.Bias || m.Cfg.kvHeads() != 1 || m.Blocks[0].LNMLP != nil || !m.Cfg.TieEmbed {
					t.Fatalf("config %+v", m.Cfg)
				}
			case "alibi":
				if !m.Cfg.Alibi || m.Cfg.MultiQuery || m.Cfg.ParallelAttn || !m.Cfg.Bias || m.Cfg.kvHeads() != 4 || m.Blocks[0].LNMLP == nil || m.Blocks[0].QKV.B == nil {
					t.Fatalf("config %+v", m.Cfg)
				}
			case "newarch":
				if !m.Cfg.NewDecoderArch || !m.Cfg.twoLN() || m.Cfg.kvHeads() != 4 || m.Blocks[0].LNMLP == nil {
					t.Fatalf("config %+v", m.Cfg)
				}
			}
			full := lastLogits(t, m, falconTinyIDs, NewKVCache(m.NumLayers()), cpu)
			got := top5(full)
			for i, w := range c.ref {
				if i < 3 && got[i] != w.id {
					t.Errorf("top-%d: got %d (%.4f), want %d (%.4f)", i+1, got[i], full[got[i]], w.id, w.logit)
				}
				if d := full[w.id] - w.logit; d > 2e-3 || d < -2e-3 {
					t.Errorf("logit[%d] = %.4f, want %.4f", w.id, full[w.id], w.logit)
				}
			}
			// Chunked decode through the cache == single shot.
			cache := NewKVCache(m.NumLayers())
			lastLogits(t, m, falconTinyIDs[:7], cache, cpu)
			var step []float32
			for i := 7; i < len(falconTinyIDs); i++ {
				step = lastLogits(t, m, falconTinyIDs[i:i+1], cache, cpu)
			}
			if d := maxAbsDiff(full, step); d > 1e-4 {
				t.Errorf("chunked vs single-shot differ by %.2e", d)
			}
			if !device.CUDAAvailable() {
				return
			}
			gpu, err := device.CUDA(0)
			if err != nil {
				t.Fatal(err)
			}
			mg := falconTinyLoad(t, c, gpu)
			g := lastLogits(t, mg, falconTinyIDs, NewKVCache(mg.NumLayers()), gpu)
			if d := maxAbsDiff(full, g); d > 1e-3 {
				t.Errorf("CPU vs CUDA differ by %.2e", d)
			}
			mb, err := LoadFalconFromHF(c.repo, gpu, dtype.BF16)
			if err != nil {
				t.Fatal(err)
			}
			b := lastLogits(t, mb, falconTinyIDs, NewKVCache(mb.NumLayers()), gpu)
			if d := maxAbsDiff(full, b); d > 5e-2 {
				t.Errorf("F32 vs BF16 differ by %.2e", d)
			}
		})
	}
}

func TestAlibiSlopes(t *testing.T) {
	// transformers build_alibi_tensor: 8 heads → 2^-1 … 2^-8; 6 heads → 4 from
	// base 2^-2 plus 2 odd powers of 2^-1.
	want8 := []float64{0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625}
	for i, s := range AlibiSlopes(8) {
		if s != want8[i] {
			t.Fatalf("8 heads: %v", AlibiSlopes(8))
		}
	}
	want6 := []float64{0.25, 0.0625, 0.015625, 0.00390625, 0.5, 0.125}
	for i, s := range AlibiSlopes(6) {
		if s != want6[i] {
			t.Fatalf("6 heads: %v", AlibiSlopes(6))
		}
	}
}
