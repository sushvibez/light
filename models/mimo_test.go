package models

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
)

// Tiny-random MiMo written by `scripts/verify/ref_mimo.py --make-tiny`: a
// Qwen2 relabelled model_type "mimo" with random model.mtp_layers.0.* weights,
// as XiaomiMiMo/MiMo-7B ships. Reference from `ref_mimo.py testdata/mimo/tiny
// f32 --ids …`, i.e. transformers' Qwen2ForCausalLM on the same checkpoint
// (MiMoForCausalLM's forward is Qwen2's; the MTP layers are never run).
var (
	mimoTinyRef = []struct {
		id    int
		logit float32
	}{{340, 0.5138}, {490, 0.4418}, {429, 0.4140}, {236, 0.3703}, {425, 0.3665}}
	mimoTinyGreedy = []int64{340, 340, 340, 340, 340, 340, 340, 340}
)

func TestMiMoTinyMatchesTransformers(t *testing.T) {
	ids := []int64{5, 17, 300, 42, 8, 499, 13, 77, 256, 3, 64, 250, 19, 6, 400, 41, 2, 90}
	cpu := device.CPU()
	m, err := LoadCausalLMFromHF("testdata/mimo/tiny", cpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	if !m.Cfg.QKVBias || m.Cfg.QKNorm || m.Cfg.RopeBase != 640000 {
		t.Fatalf("config %+v", m.Cfg)
	}
	full := lastLogits(t, m, ids, NewKVCache(m.NumLayers()), cpu)
	got := top5(full)
	for i, w := range mimoTinyRef {
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
	for len(greedy) < len(mimoTinyGreedy) {
		next := int64(top5(step)[0])
		greedy = append(greedy, next)
		step = lastLogits(t, m, []int64{next}, cache, cpu)
	}
	if !slices.Equal(greedy, mimoTinyGreedy) {
		t.Errorf("greedy %v, want %v", greedy, mimoTinyGreedy)
	}
}

// MiMo-7B-RL (36 layers, 32 / 8 heads, QKV bias, rope θ 640000; one MTP
// layer in the checkpoint, unused) in BF16 on CUDA: 15.6 GB.
//
// Reference: transformers' Qwen2ForCausalLM on the checkpoint (no remote code),
//
//	python scripts/verify/ref_mimo.py XiaomiMiMo/MiMo-7B-RL bf16 --chat
//
// TODO(mimo-7b): fill in the prompt ids, reference logits and greedy ids from
// that run; until then the test stops after the reference-free checks.
const mimo7BRepo = "XiaomiMiMo/MiMo-7B-RL"

var (
	mimo7BChatIDs []int64
	mimo7BRef     []struct {
		id    int
		logit float32
	}
	mimo7BGreedy []int64
)

func TestMiMo7BMatchesTransformersBF16(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (15.6 GB BF16 checkpoint)")
	}
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--XiaomiMiMo--MiMo-7B-RL/snapshots/*/model*.safetensors")); len(m) == 0 {
		t.Skip(mimo7BRepo + " not in HF cache")
	}
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	requireHostMemGB(t, 20)
	requireGPUMemGB(t, 0, 18)
	releaseModelMemory(t)
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	os.Setenv("HF_HUB_OFFLINE", "1")
	m, err := LoadCausalLMFromHF(mimo7BRepo, dev, dtype.BF16)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cfg
	if c.NumLayers != 36 || c.NumHeads != 32 || c.NumKVHeads != 8 || !c.QKVBias || c.RopeBase != 640000 || c.TieEmbed {
		t.Fatalf("config %+v", c)
	}
	t.Logf("params %.3f B", float64(m.ParamCount())/1e9)
	ids := mimo7BChatIDs
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
	if mimo7BRef == nil {
		t.Skip("transformers reference values not recorded yet (see TODO(mimo-7b))")
	}
	for i, w := range mimo7BRef {
		if i < 2 && got[i] != w.id {
			t.Errorf("top-%d: got %d (%.3f), want %d (%.3f)", i+1, got[i], first[got[i]], w.id, w.logit)
		}
		if d := first[w.id] - w.logit; d > 0.75 || d < -0.75 {
			t.Errorf("logit[%d] = %.3f, want %.3f (BF16 tol 0.75)", w.id, first[w.id], w.logit)
		}
	}
	agree := 0
	for agree < len(greedy) && agree < len(mimo7BGreedy) && greedy[agree] == mimo7BGreedy[agree] {
		agree++
	}
	t.Logf("greedy: first %d of %d agree with transformers BF16", agree, len(mimo7BGreedy))
	if agree < 8 {
		t.Errorf("greedy %v diverges from transformers BF16 %v after %d tokens", greedy, mimo7BGreedy, agree)
	}
}
