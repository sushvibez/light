package models

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
)

// Tiny-random GLM-4 checkpoints written by `scripts/verify/ref_glm.py
// --make-tiny {glm|glm4}`, reference values from `ref_glm.py
// testdata/glm/tiny-<v> f32 --ids …` (transformers 5.21, eager, CPU): GQA
// 4 / 2 heads of 24, partial interleaved RoPE over 12 dims, Q/K/V biases,
// fused gate_up_proj; "glm4" adds the post-attention / post-MLP norms. Norms
// and biases are perturbed, so the rotary-row reordering of the biases and
// every norm affect the logits.
var glmTinyCases = []struct {
	dir       string
	postNorms bool
	ref       []struct {
		id    int
		logit float32
	}
	greedy []int64
}{
	{"testdata/glm/tiny-glm", false,
		[]struct {
			id    int
			logit float32
		}{{493, 0.4627}, {400, 0.4585}, {98, 0.4524}, {135, 0.4296}, {120, 0.3930}},
		[]int64{493, 120, 98, 400, 493, 120, 98, 400}},
	{"testdata/glm/tiny-glm4", true,
		[]struct {
			id    int
			logit float32
		}{{52, 0.5730}, {56, 0.4925}, {164, 0.4795}, {302, 0.4149}, {263, 0.4036}},
		[]int64{52, 52, 52, 52, 52, 52, 52, 52}},
}

func TestGLMTinyMatchesTransformers(t *testing.T) {
	ids := []int64{5, 17, 300, 42, 8, 499, 13, 77, 256, 3, 64, 250, 19, 6, 400, 41, 2, 90}
	for _, tc := range glmTinyCases {
		t.Run(tc.dir, func(t *testing.T) {
			cpu := device.CPU()
			m, err := LoadCausalLMFromHF(tc.dir, cpu, dtype.F32)
			if err != nil {
				t.Fatal(err)
			}
			c := m.Cfg
			if c.PostNorms != tc.postNorms || !c.QKVBias || c.PartialRotary != 0.5 || m.rotaryDim() != 12 ||
				c.NumKVHeads != 2 || c.HeadDim != 24 || (m.Blocks[0].PostAttnNorm != nil) != tc.postNorms {
				t.Fatalf("config %+v", c)
			}
			full := lastLogits(t, m, ids, NewKVCache(m.NumLayers()), cpu)
			got := top5(full)
			for i, w := range tc.ref {
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
			for len(greedy) < len(tc.greedy) {
				next := int64(top5(step)[0])
				greedy = append(greedy, next)
				step = lastLogits(t, m, []int64{next}, cache, cpu)
			}
			if !slices.Equal(greedy, tc.greedy) {
				t.Errorf("greedy %v, want %v", greedy, tc.greedy)
			}
			cache = NewKVCache(m.NumLayers())
			lastLogits(t, m, ids[:5], cache, cpu)
			var chunked []float32
			for i := 5; i < len(ids); i++ {
				chunked = lastLogits(t, m, ids[i:i+1], cache, cpu)
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
			mg, err := LoadCausalLMFromHF(tc.dir, gpu, dtype.F32)
			if err != nil {
				t.Fatal(err)
			}
			if d := maxAbsDiff(full, lastLogits(t, mg, ids, NewKVCache(mg.NumLayers()), gpu)); d > 1e-4 {
				t.Errorf("CUDA vs CPU logits differ by %.2e", d)
			}
		})
	}
}

// GLM-4-9B-0414 (glm4: 40 layers, 32 / 2 heads of 128, partial RoPE 0.5,
// post-norms) in BF16 on CUDA: 18.8 GB.
//
// Reference: transformers 5.21 BF16 eager,
//
//	python scripts/verify/ref_glm.py zai-org/GLM-4-9B-0414 bf16 --chat
//
// (A100, on the GPU.) The chat template ends in <|assistant|>; the model
// opens its reply with a newline, 14 logits ahead of the runner-up.
const glm4_9BRepo = "zai-org/GLM-4-9B-0414"

var (
	glm4_9BChatIDs = []int64{151331, 151333, 151335, 198, 2610, 525, 264, 63141, 17821, 13, 151336, 198, 840, 20726, 304,
		825, 11646, 3170, 279, 12877, 374, 6303, 13, 151337}
	glm4_9BRef = []struct {
		id    int
		logit float32
	}{{198, 32.0}, {271, 17.75}, {785, 16.5}, {576, 13.25}, {319, 13.0}}
	// "\nThe sky appears blue because Earth's atmosphere scatters shorter blue wavelengths of sunlight"
	glm4_9BGreedy = []int64{198, 785, 12877, 7951, 6303, 1576, 9234, 594, 16550, 1136, 10170, 23263, 6303, 92080, 315, 38810}
)

func TestGLM4_9BMatchesTransformersBF16(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (18.8 GB BF16 checkpoint)")
	}
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--zai-org--GLM-4-9B-0414/snapshots/*/model*.safetensors")); len(m) == 0 {
		t.Skip(glm4_9BRepo + " not in HF cache")
	}
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	requireHostMemGB(t, 22)
	requireGPUMemGB(t, 0, 21)
	releaseModelMemory(t)
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	os.Setenv("HF_HUB_OFFLINE", "1")
	m, err := LoadCausalLMFromHF(glm4_9BRepo, dev, dtype.BF16)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cfg
	if c.NumLayers != 40 || c.NumHeads != 32 || c.NumKVHeads != 2 || c.HeadDim != 128 || m.rotaryDim() != 64 ||
		!c.QKVBias || !c.PostNorms || c.TieEmbed {
		t.Fatalf("config %+v", c)
	}
	t.Logf("params %.3f B", float64(m.ParamCount())/1e9)
	ids := glm4_9BChatIDs
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
	for i, w := range glm4_9BRef {
		if i < 2 && got[i] != w.id {
			t.Errorf("top-%d: got %d (%.3f), want %d (%.3f)", i+1, got[i], first[got[i]], w.id, w.logit)
		}
		if d := first[w.id] - w.logit; d > 0.75 || d < -0.75 {
			t.Errorf("logit[%d] = %.3f, want %.3f (BF16 tol 0.75)", w.id, first[w.id], w.logit)
		}
	}
	agree := 0
	for agree < len(greedy) && agree < len(glm4_9BGreedy) && greedy[agree] == glm4_9BGreedy[agree] {
		agree++
	}
	t.Logf("greedy: first %d of %d agree with transformers BF16", agree, len(glm4_9BGreedy))
	if agree < 8 {
		t.Errorf("greedy %v diverges from transformers BF16 %v after %d tokens", greedy, glm4_9BGreedy, agree)
	}
}
