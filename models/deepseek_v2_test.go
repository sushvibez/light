package models

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
)

// Tiny-random DeepSeek-V2 checkpoints written by
// `scripts/verify/ref_deepseek_v2.py --make-tiny {lite|qlora}`, reference
// values from `ref_deepseek_v2.py testdata/deepseek_v2/tiny-<v> f32 --ids …`
// (transformers 5.21, eager, CPU). Both: 3 layers (the first dense), 4 heads,
// KV latent 32, no-RoPE key 16 + RoPE key 8, value 12, YaRN (factor 4,
// mscale = mscale_all_dim = 0.707), top-3 of 8 experts without
// renormalisation, routed_scaling_factor 1.5, one shared expert, perturbed
// norms. "lite" has a plain q_proj (V2-Lite), "qlora" q_a → norm → q_b (V2).
var deepseekTinyCases = []struct {
	dir   string
	qlora bool
	ref   []struct {
		id    int
		logit float32
	}
	greedy []int64
}{
	{"testdata/deepseek_v2/tiny-lite", false,
		[]struct {
			id    int
			logit float32
		}{{325, 0.5222}, {425, 0.3842}, {256, 0.3763}, {150, 0.3689}, {411, 0.3450}},
		[]int64{325, 198, 175, 503, 390, 107, 358, 305}},
	{"testdata/deepseek_v2/tiny-qlora", true,
		[]struct {
			id    int
			logit float32
		}{{48, 0.5615}, {434, 0.4427}, {257, 0.4138}, {395, 0.3935}, {292, 0.3745}},
		[]int64{48, 466, 355, 379, 13, 105, 13, 434}},
}

var deepseekTinyIDs = []int64{5, 17, 300, 42, 8, 499, 13, 77, 256, 3, 64, 250, 19, 6, 400, 41, 2, 90}

func TestDeepseekV2TinyMatchesTransformers(t *testing.T) {
	for _, tc := range deepseekTinyCases {
		t.Run(tc.dir, func(t *testing.T) {
			cpu := device.CPU()
			m, err := LoadDeepseekV2FromHF(tc.dir, cpu, dtype.F32)
			if err != nil {
				t.Fatal(err)
			}
			c := m.Cfg
			if (c.QLoraRank > 0) != tc.qlora || c.KVLoraRank != 32 || c.QKNopeDim != 16 || c.QKRopeDim != 8 ||
				c.VHeadDim != 12 || c.NumExperts != 8 || c.TopK != 3 || c.NumShared != 1 || c.FirstKDense != 1 ||
				c.NormTopK || c.RoutedScale != 1.5 || c.RopeScaling == nil || c.RopeScaling.Type != "yarn" ||
				c.RopeScaling.MScaleAllDim != 0.707 || m.Blocks[0].MoE != nil || m.Blocks[1].MoE == nil {
				t.Fatalf("config %+v (rope %+v)", c, c.RopeScaling)
			}
			full := lastLogits(t, m, deepseekTinyIDs, NewKVCache(m.NumLayers()), cpu)
			got := top5(full)
			for i, w := range tc.ref {
				if i < 3 && got[i] != w.id {
					t.Errorf("top-%d: got %d (%.4f), want %d (%.4f)", i+1, got[i], full[got[i]], w.id, w.logit)
				}
				if d := full[w.id] - w.logit; d > 2e-4 || d < -2e-4 {
					t.Errorf("logit[%d] = %.5f, want %.4f", w.id, full[w.id], w.logit)
				}
			}
			// Greedy decode through the compressed-latent cache.
			cache := NewKVCache(m.NumLayers())
			step := lastLogits(t, m, deepseekTinyIDs, cache, cpu)
			var greedy []int64
			for len(greedy) < len(tc.greedy) {
				next := int64(top5(step)[0])
				greedy = append(greedy, next)
				step = lastLogits(t, m, []int64{next}, cache, cpu)
			}
			if !slices.Equal(greedy, tc.greedy) {
				t.Errorf("greedy %v, want %v", greedy, tc.greedy)
			}
			if k := cache.K[0]; k.Dim(1) != 1 || k.Dim(3) != c.KVLoraRank {
				t.Errorf("cache holds %v, want (B, 1, T, %d) latents", k.Shape(), c.KVLoraRank)
			}
			// Chunked prefill + single steps == single shot.
			cache = NewKVCache(m.NumLayers())
			lastLogits(t, m, deepseekTinyIDs[:5], cache, cpu)
			var chunked []float32
			for i := 5; i < len(deepseekTinyIDs); i++ {
				chunked = lastLogits(t, m, deepseekTinyIDs[i:i+1], cache, cpu)
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
			mg, err := LoadDeepseekV2FromHF(tc.dir, gpu, dtype.F32)
			if err != nil {
				t.Fatal(err)
			}
			if d := maxAbsDiff(full, lastLogits(t, mg, deepseekTinyIDs, NewKVCache(mg.NumLayers()), gpu)); d > 1e-4 {
				t.Errorf("CUDA vs CPU logits differ by %.2e", d)
			}
		})
	}
}

// DeepSeek-V2-Lite-Chat (15.7 B params, 2.4 B active; 27 layers, the first
// dense; 64 routed experts top-6 + 2 shared; MLA with a 512-dim latent;
// YaRN factor 40) in BF16 on CUDA: 31.4 GB, fits a 40 GB A100.
//
// Reference: transformers 5.21 BF16 eager,
//
//	python scripts/verify/ref_deepseek_v2.py deepseek-ai/DeepSeek-V2-Lite-Chat bf16 --chat
//
// TODO(deepseek-v2-lite): fill in the prompt ids, reference logits and greedy
// ids from that run; until then the test stops after the reference-free
// checks.
const deepseekV2LiteRepo = "deepseek-ai/DeepSeek-V2-Lite-Chat"

var (
	deepseekV2LiteChatIDs []int64
	deepseekV2LiteRef     []struct {
		id    int
		logit float32
	}
	deepseekV2LiteGreedy []int64
)

func TestDeepseekV2LiteMatchesTransformersBF16(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (31.4 GB BF16 checkpoint)")
	}
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/*/model*.safetensors")); len(m) == 0 {
		t.Skip(deepseekV2LiteRepo + " not in HF cache")
	}
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	requireHostMemGB(t, 36)
	requireGPUMemGB(t, 0, 34)
	releaseModelMemory(t)
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	os.Setenv("HF_HUB_OFFLINE", "1")
	m, err := LoadDeepseekV2FromHF(deepseekV2LiteRepo, dev, dtype.BF16)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cfg
	if c.NumLayers != 27 || c.QLoraRank != 0 || c.KVLoraRank != 512 || c.QKNopeDim != 128 || c.QKRopeDim != 64 ||
		c.VHeadDim != 128 || c.NumExperts != 64 || c.TopK != 6 || c.NumShared != 2 || c.FirstKDense != 1 ||
		c.RopeScaling == nil || c.RopeScaling.Factor != 40 {
		t.Fatalf("config %+v (rope %+v)", c, c.RopeScaling)
	}
	t.Logf("params %.3f B, softmax scale %.5f", float64(m.ParamCount())/1e9, c.softmaxScale())
	ids := deepseekV2LiteChatIDs
	if ids == nil {
		ids = make([]int64, 32) // a fixed sequence until the chat-template ids are recorded
		for i := range ids {
			ids[i] = int64(1000 + 37*i)
		}
	}
	cache := NewKVCache(m.NumLayers())
	row := lastLogits(t, m, ids, cache, dev)
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
	if deepseekV2LiteRef == nil {
		t.Skip("transformers reference values not recorded yet (see TODO(deepseek-v2-lite))")
	}
	for i, w := range deepseekV2LiteRef {
		if i < 2 && got[i] != w.id {
			t.Errorf("top-%d: got %d (%.3f), want %d (%.3f)", i+1, got[i], first[got[i]], w.id, w.logit)
		}
		if d := first[w.id] - w.logit; d > 0.75 || d < -0.75 {
			t.Errorf("logit[%d] = %.3f, want %.3f (BF16 tol 0.75)", w.id, first[w.id], w.logit)
		}
	}
	agree := 0
	for agree < len(greedy) && agree < len(deepseekV2LiteGreedy) && greedy[agree] == deepseekV2LiteGreedy[agree] {
		agree++
	}
	t.Logf("greedy: first %d of %d agree with transformers BF16", agree, len(deepseekV2LiteGreedy))
	if agree < 8 {
		t.Errorf("greedy %v diverges from transformers BF16 %v after %d tokens", greedy, deepseekV2LiteGreedy, agree)
	}
}
