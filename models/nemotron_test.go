package models

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
)

// Tiny-random Nemotron written by `scripts/verify/ref_nemotron.py
// --make-tiny`, reference values from `ref_nemotron.py testdata/nemotron/tiny
// f32 --ids …` (transformers 5.21, eager, CPU): GQA 4 / 2 heads of 24,
// partial RoPE over 12 dims, LayerNorm1P norms with perturbed weights and
// biases, non-gated relu² MLP.
var (
	nemotronTinyRef = []struct {
		id    int
		logit float32
	}{{438, 0.7448}, {456, 0.5729}, {207, 0.5336}, {355, 0.5294}, {73, 0.4924}}
	nemotronTinyGreedy = []int64{438, 207, 438, 207, 152, 438, 207, 152}
)

func TestNemotronTinyMatchesTransformers(t *testing.T) {
	ids := []int64{5, 17, 300, 42, 8, 499, 13, 77, 256, 3, 64, 250, 19, 6, 400, 41, 2, 90}
	cpu := device.CPU()
	m, err := LoadCausalLMFromHF("testdata/nemotron/tiny", cpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cfg
	if !c.LayerNorm1P || !c.ReLU2MLP || c.PartialRotary != 0.5 || m.rotaryDim() != 12 || c.HeadDim != 24 ||
		c.NumKVHeads != 2 || m.Blocks[0].GateProj != nil || m.FinalLN == nil {
		t.Fatalf("config %+v", c)
	}
	full := lastLogits(t, m, ids, NewKVCache(m.NumLayers()), cpu)
	got := top5(full)
	for i, w := range nemotronTinyRef {
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
	for len(greedy) < len(nemotronTinyGreedy) {
		next := int64(top5(step)[0])
		greedy = append(greedy, next)
		step = lastLogits(t, m, []int64{next}, cache, cpu)
	}
	if !slices.Equal(greedy, nemotronTinyGreedy) {
		t.Errorf("greedy %v, want %v", greedy, nemotronTinyGreedy)
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
	mg, err := LoadCausalLMFromHF("testdata/nemotron/tiny", gpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	if d := maxAbsDiff(full, lastLogits(t, mg, ids, NewKVCache(mg.NumLayers()), gpu)); d > 1e-4 {
		t.Errorf("CUDA vs CPU logits differ by %.2e", d)
	}
}

// Nemotron-4-Mini-Hindi-4B-Instruct: the Nemotron-Mini-4B architecture (32
// layers, 24 / 8 heads of 128, partial RoPE 0.5, relu² MLP, LayerNorm1P,
// 256k vocabulary) and, unlike Nemotron-Mini-4B-Instruct / Minitron-4B (which
// ship pytorch_model.bin only), a safetensors checkpoint. BF16 on CUDA, 8.4 GB.
//
// Reference: transformers 5.21 BF16 eager,
//
//	python scripts/verify/ref_nemotron.py nvidia/Nemotron-4-Mini-Hindi-4B-Instruct bf16 --chat
//
// TODO(nemotron-4b): fill in the prompt ids, reference logits and greedy ids
// from that run; until then the test stops after the reference-free checks.
const nemotron4BRepo = "nvidia/Nemotron-4-Mini-Hindi-4B-Instruct"

var (
	nemotron4BChatIDs []int64
	nemotron4BRef     []struct {
		id    int
		logit float32
	}
	nemotron4BGreedy []int64
)

func TestNemotron4BMatchesTransformersBF16(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (8.4 GB BF16 checkpoint)")
	}
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--nvidia--Nemotron-4-Mini-Hindi-4B-Instruct/snapshots/*/model*.safetensors")); len(m) == 0 {
		t.Skip(nemotron4BRepo + " not in HF cache")
	}
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	requireHostMemGB(t, 12)
	releaseModelMemory(t)
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	os.Setenv("HF_HUB_OFFLINE", "1")
	m, err := LoadCausalLMFromHF(nemotron4BRepo, dev, dtype.BF16)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cfg
	if c.NumLayers != 32 || c.NumHeads != 24 || c.NumKVHeads != 8 || c.HeadDim != 128 || m.rotaryDim() != 64 ||
		!c.LayerNorm1P || !c.ReLU2MLP || c.TieEmbed {
		t.Fatalf("config %+v", c)
	}
	t.Logf("params %.3f B", float64(m.ParamCount())/1e9)
	ids := nemotron4BChatIDs
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
	if nemotron4BRef == nil {
		t.Skip("transformers reference values not recorded yet (see TODO(nemotron-4b))")
	}
	for i, w := range nemotron4BRef {
		if i < 2 && got[i] != w.id {
			t.Errorf("top-%d: got %d (%.3f), want %d (%.3f)", i+1, got[i], first[got[i]], w.id, w.logit)
		}
		if d := first[w.id] - w.logit; d > 0.75 || d < -0.75 {
			t.Errorf("logit[%d] = %.3f, want %.3f (BF16 tol 0.75)", w.id, first[w.id], w.logit)
		}
	}
	agree := 0
	for agree < len(greedy) && agree < len(nemotron4BGreedy) && greedy[agree] == nemotron4BGreedy[agree] {
		agree++
	}
	t.Logf("greedy: first %d of %d agree with transformers BF16", agree, len(nemotron4BGreedy))
	if agree < 8 {
		t.Errorf("greedy %v diverges from transformers BF16 %v after %d tokens", greedy, nemotron4BGreedy, agree)
	}
}
