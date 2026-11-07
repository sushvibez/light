package models

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
)

// Tiny Qwen3.5 in the fine-grained FP8 format of Qwen/Qwen3.5-27B-FP8
// (F8E4M3 projections with a per-16×16-block weight_scale_inv; embedding,
// LM head and in_proj_a / in_proj_b unquantized), written by
// `scripts/verify/ref_qwen35_fp8.py --make`. Reference: transformers 5.21
// loading the same files through its FP8 support (dequantize on load), F32,
// CPU, the 70-token ids of TestQwen35TinyMatchesTransformers.
const qwen35TinyFP8Dir = "testdata/qwen35/tiny-fp8"

var (
	qwen35TinyFP8Ref = []struct {
		id    int
		logit float32
	}{{34, 0.4421}, {309, 0.4375}, {259, 0.4161}, {254, 0.4029}, {278, 0.3864}}
	qwen35TinyFP8Greedy = []int64{34, 335, 165, 470, 231, 226, 169, 287}
)

func TestQwen35FP8TinyMatchesTransformers(t *testing.T) {
	devs := []device.Device{device.CPU()}
	if device.CUDAAvailable() {
		if gpu, err := device.CUDA(0); err == nil {
			devs = append(devs, gpu)
		}
	}
	ids := qwen35TinyIDs()
	for _, dev := range devs {
		t.Run(dev.String(), func(t *testing.T) {
			m, err := LoadQwen35FromHF(qwen35TinyFP8Dir, dev, dtype.F32)
			if err != nil {
				t.Fatal(err)
			}
			b0, b1 := m.Blocks[0], m.Blocks[1]
			for name, l := range map[string]struct {
				w     dtype.DType
				block bool
			}{
				"in_proj_qkv": {b0.InQKV.W.DType(), b0.InQKV.WBlockScale != nil},
				"out_proj":    {b0.OutProj.W.DType(), b0.OutProj.WBlockScale != nil},
				"q_proj":      {b1.QProj.W.DType(), b1.QProj.WBlockScale != nil},
				"down_proj":   {b1.Down.W.DType(), b1.Down.WBlockScale != nil},
			} {
				if l.w != dtype.F8E4M3 || !l.block {
					t.Fatalf("%s: %v, block scale %v; want F8E4M3 with block scales", name, l.w, l.block)
				}
			}
			if b0.InA.W.DType() != dtype.F32 || b0.InA.WBlockScale != nil || m.LMHead.W.DType() != dtype.F32 {
				t.Fatalf("unquantized modules: in_proj_a %v, lm_head %v", b0.InA.W.DType(), m.LMHead.W.DType())
			}
			if b0.InQKV.WBlock != [2]int{16, 16} {
				t.Fatalf("block %v, want 16 × 16", b0.InQKV.WBlock)
			}

			cache := NewKVCache(m.NumLayers())
			row := lastLogits(t, m, ids, cache, dev)
			got := top5(row)
			for i, w := range qwen35TinyFP8Ref {
				if i < 3 && got[i] != w.id {
					t.Errorf("top-%d: got %d (%.4f), want %d (%.4f)", i+1, got[i], row[got[i]], w.id, w.logit)
				}
				if d := row[w.id] - w.logit; d > 2e-3 || d < -2e-3 {
					t.Errorf("logit[%d] = %.4f, want %.4f", w.id, row[w.id], w.logit)
				}
			}
			var greedy []int64
			for len(greedy) < len(qwen35TinyFP8Greedy) {
				next := int64(top5(row)[0])
				greedy = append(greedy, next)
				row = lastLogits(t, m, []int64{next}, cache, dev)
			}
			if !slices.Equal(greedy, qwen35TinyFP8Greedy) {
				t.Errorf("greedy %v, want %v", greedy, qwen35TinyFP8Greedy)
			}
		})
	}
}

// Qwen/Qwen3.5-27B-FP8 (fine-grained FP8, 128 × 128 blocks; 30.9 GB with the
// vision tower, ~28 GB of text weights on the GPU) in BF16 activations on
// CUDA. Weights stay F8E4M3 and are expanded to BF16 one matmul at a time,
// so it fits a 40 GB A100; an L4's 22 GiB is too small.
//
// Reference: transformers 5.21 on the same checkpoint. Below sm_89 (A100)
// transformers dequantizes the FP8 weights to BF16 on load, which gives the
// very weights Light computes with, so the comparison is BF16 vs BF16:
//
//	python scripts/verify/ref_qwen35.py Qwen/Qwen3.5-27B-FP8 bf16 --chat
//
// TODO(qwen35-27b-fp8): fill in the reference values (and the prompt ids if
// they differ from the 9B's); until then the test stops after the
// reference-free checks.
const qwen3527BFP8Repo = "Qwen/Qwen3.5-27B-FP8"

var (
	qwen3527BFP8ChatIDs = qwen359BChatIDs
	qwen3527BFP8Ref     []struct {
		id    int
		logit float32
	}
	qwen3527BFP8Greedy []int64
)

func TestQwen3527BFP8MatchesTransformers(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (31 GB FP8 checkpoint)")
	}
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--Qwen--Qwen3.5-27B-FP8/snapshots/*/model*.safetensors")); len(m) == 0 {
		t.Skip(qwen3527BFP8Repo + " not in HF cache")
	}
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	requireHostMemGB(t, 36)
	requireGPUMemGB(t, 0, 32)
	releaseModelMemory(t)
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	os.Setenv("HF_HUB_OFFLINE", "1")
	m, err := LoadQwen35FromHF(qwen3527BFP8Repo, dev, dtype.BF16)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cfg
	if c.NumLayers != 64 || c.LinearKHeads != 16 || c.LinearVHeads != 48 || c.HiddenSize != 5120 || c.VHeadsTiled ||
		c.fp8Block != [2]int{128, 128} {
		t.Fatalf("config %+v", c)
	}
	b0 := m.Blocks[0]
	if b0.InQKV.W.DType() != dtype.F8E4M3 || b0.InQKV.WBlockScale == nil || b0.Down.W.DType() != dtype.F8E4M3 ||
		b0.InA.W.DType() != dtype.BF16 || m.LMHead.W.DType() != dtype.BF16 {
		t.Fatalf("weight formats: in_proj_qkv %v, down %v, in_proj_a %v, lm_head %v",
			b0.InQKV.W.DType(), b0.Down.W.DType(), b0.InA.W.DType(), m.LMHead.W.DType())
	}
	t.Logf("text params %.3f B", float64(m.ParamCount())/1e9)

	cache := NewKVCache(m.NumLayers())
	row := lastLogits(t, m, qwen3527BFP8ChatIDs, cache, dev)
	got := top5(row)
	t.Logf("top-5 %v logits %.3f %.3f %.3f %.3f %.3f", got, row[got[0]], row[got[1]], row[got[2]], row[got[3]], row[got[4]])
	first := slices.Clone(row)
	var greedy []int64
	for len(greedy) < 16 {
		next := int64(top5(row)[0])
		greedy = append(greedy, next)
		row = lastLogits(t, m, []int64{next}, cache, dev)
	}
	t.Logf("greedy %v", greedy)

	if qwen3527BFP8Ref == nil {
		t.Skip("transformers reference values not recorded yet (see TODO(qwen35-27b-fp8))")
	}
	for i, w := range qwen3527BFP8Ref {
		if i < 2 && got[i] != w.id {
			t.Errorf("top-%d: got %d (%.3f), want %d (%.3f)", i+1, got[i], first[got[i]], w.id, w.logit)
		}
		if d := first[w.id] - w.logit; d > 0.75 || d < -0.75 {
			t.Errorf("logit[%d] = %.3f, want %.3f (BF16 tol 0.75)", w.id, first[w.id], w.logit)
		}
	}
	agree := 0
	for agree < len(greedy) && agree < len(qwen3527BFP8Greedy) && greedy[agree] == qwen3527BFP8Greedy[agree] {
		agree++
	}
	t.Logf("greedy: first %d of %d agree with transformers BF16", agree, len(qwen3527BFP8Greedy))
	if agree < 8 {
		t.Errorf("greedy %v diverges from transformers %v after %d tokens", greedy, qwen3527BFP8Greedy, agree)
	}
}
