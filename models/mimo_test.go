package models

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
	"light/tensor"
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
// (A100, on the GPU; transformers reports exactly the 16 mtp_layers tensors
// as unused.) The reasoning model opens with "<think>", 16 logits ahead.
const mimo7BRepo = "XiaomiMiMo/MiMo-7B-RL"

var (
	mimo7BChatIDs = []int64{151644, 8948, 198, 2610, 525, 264, 63594, 17847, 13, 151645, 198, 151644, 872, 198, 840,
		20772, 304, 825, 11652, 3170, 279, 12884, 374, 6303, 13, 151645, 198, 151644, 77091, 198}
	mimo7BRef = []struct {
		id    int
		logit float32
	}{{13708, 34.5}, {27, 18.25}, {2, 17.625}, {12, 16.0}, {4421, 15.875}}
	// "<think>" + newline + "Okay, the user wants to know why the sky is blue in"
	mimo7BGreedy = []int64{13708, 766, 397, 32313, 11, 279, 1196, 6801, 311, 1414, 3170, 279, 12884, 374, 6303, 304}
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
	// Logits against transformers F32, loosely: MiMo's k_proj biases (up
	// to 55) make BF16 attention scores very sensitive to rounding order —
	// transformers' own BF16 run puts "<think>" at 34.5 against 30.15 in F32
	// and a different set of runners-up ("<", "#", "-" instead of " Ray",
	// newline, " more"). Light BF16 lands closer (33.0, " Ray" second).
	if got[0] != mimo7BRefF32[0].id {
		t.Errorf("top-1: got %d (%.3f), want %d", got[0], first[got[0]], mimo7BRefF32[0].id)
	}
	for _, w := range mimo7BRefF32 {
		if d := first[w.id] - w.logit; d > 4 || d < -4 {
			t.Errorf("logit[%d] = %.3f, transformers F32 %.3f (BF16 tol 4.0)", w.id, first[w.id], w.logit)
		}
	}
	_ = mimo7BRef // transformers BF16 top-5, kept for the record
	agree := 0
	for agree < len(greedy) && agree < len(mimo7BGreedy) && greedy[agree] == mimo7BGreedy[agree] {
		agree++
	}
	t.Logf("greedy: first %d of %d agree with transformers BF16", agree, len(mimo7BGreedy))
	if agree < 8 {
		t.Errorf("greedy %v diverges from transformers BF16 %v after %d tokens", greedy, mimo7BGreedy, agree)
	}
}

// MiMo-7B-RL in F32 on CUDA (30.5 GB of weights: an 80 GB A100), against
// transformers F32 (`ref_mimo.py XiaomiMiMo/MiMo-7B-RL f32 --chat`). With
// k_proj biases up to 55 the attention scores carry a large common offset,
// so BF16 runs are sensitive to rounding order; the F32 comparison tells a
// real difference from BF16 noise. Reference from an A100 80 GB.
var mimo7BRefF32 = []struct {
	id    int
	logit float32
}{{13708, 30.1521}, {13255, 18.7655}, {198, 17.3808}, {803, 16.4502}, {71816, 15.8957}}

func TestMiMo7BMatchesTransformersF32(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (30.5 GB F32)")
	}
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--XiaomiMiMo--MiMo-7B-RL/snapshots/*/model*.safetensors")); len(m) == 0 {
		t.Skip(mimo7BRepo + " not in HF cache")
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
	m, err := LoadCausalLMFromHF(mimo7BRepo, dev, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	row := lastLogits(t, m, mimo7BChatIDs, NewKVCache(m.NumLayers()), dev)
	got := top5(row)
	t.Logf("F32 top-5 %v logits %.4f %.4f %.4f %.4f %.4f", got, row[got[0]], row[got[1]], row[got[2]], row[got[3]], row[got[4]])
	for _, w := range mimo7BRef {
		t.Logf("BF16-reference id %d: Light F32 logit %.4f (transformers BF16 %.3f)", w.id, row[w.id], w.logit)
	}
	for i, w := range mimo7BRefF32 {
		if i < 2 && got[i] != w.id {
			t.Errorf("top-%d: got %d (%.4f), want %d (%.4f)", i+1, got[i], row[got[i]], w.id, w.logit)
		}
		if d := row[w.id] - w.logit; d > 0.02 || d < -0.02 {
			t.Errorf("logit[%d] = %.4f, want %.4f (F32 tol 0.02)", w.id, row[w.id], w.logit)
		}
	}
}

// The MTP draft head of the tiny MiMo (random mtp_layers.0) against
// `scripts/verify/ref_mimo_mtp.py testdata/mimo/tiny --ids …`: MiMo's MTP
// layer re-expressed in plain PyTorch on transformers' final normed hidden
// states, the pairs (h_i, ids[i+1]) as one causal sequence; last row top-5.
func TestMiMoMTPDraftMatchesReference(t *testing.T) {
	ids := []int64{5, 17, 300, 42, 8, 499, 13, 77, 256, 3, 64, 250, 19, 6, 400, 41, 2, 90}
	ref := []struct {
		id    int
		logit float32
	}{{148, 0.4520}, {7, 0.4276}, {245, 0.4224}, {472, 0.4010}, {191, 0.4002}}
	cpu := device.CPU()
	m, err := LoadCausalLMFromHF("testdata/mimo/tiny", cpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	if m.MTP == nil {
		t.Fatal("MTP head not loaded")
	}
	lastLogits(t, m, ids, NewKVCache(m.NumLayers()), cpu)
	T := len(ids) - 1
	h, err := m.LastHidden.Narrow(1, 0, T)
	if err != nil {
		t.Fatal(err)
	}
	if h, err = h.Contiguous(); err != nil {
		t.Fatal(err)
	}
	logits, err := m.MTPDraft(h, ids[1:], NewKVCache(1))
	if err != nil {
		t.Fatal(err)
	}
	rs, err := rows(logits)
	if err != nil {
		t.Fatal(err)
	}
	row := rs[len(rs)-1]
	got := top5(row)
	for i, w := range ref {
		if i < 3 && got[i] != w.id {
			t.Errorf("draft top-%d: got %d (%.4f), want %d (%.4f)", i+1, got[i], row[got[i]], w.id, w.logit)
		}
		if d := row[w.id] - w.logit; d > 2e-4 || d < -2e-4 {
			t.Errorf("draft logit[%d] = %.5f, want %.4f", w.id, row[w.id], w.logit)
		}
	}
}

// Speculative greedy decoding with the MTP head emits exactly the plain
// greedy tokens (the tiny model's random draft head is mostly wrong, so
// both the accept and the reject / rollback paths run).
func TestSpecGreedyEqualsGreedy(t *testing.T) {
	ids := []int64{5, 17, 300, 42, 8, 499, 13, 77, 256, 3, 64, 250, 19, 6, 400, 41, 2, 90}
	const n = 24
	cpu := device.CPU()
	m, err := LoadCausalLMFromHF("testdata/mimo/tiny", cpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	cache := NewKVCache(m.NumLayers())
	row := lastLogits(t, m, ids, cache, cpu)
	var want []int64
	for len(want) < n {
		next := int64(top5(row)[0])
		want = append(want, next)
		row = lastLogits(t, m, []int64{next}, cache, cpu)
	}
	var got []int64
	st, err := SpecGreedy(m, ids, n, func(tok int64) bool { got = append(got, tok); return true })
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("drafted %d, accepted %d", st.Drafted, st.Accepted)
	if !slices.Equal(got, want) {
		t.Fatalf("speculative %v\nwant greedy %v", got, want)
	}
	if st.Drafted == 0 || st.Accepted == st.Drafted {
		t.Logf("note: the run did not exercise both accept and reject (%d / %d)", st.Accepted, st.Drafted)
	}
}

// The accept and reject paths of speculative decoding with a scripted
// drafter (right except at every third draft): the output is still plain
// greedy, and the drafter sees every (hidden, next) pair of the stream
// exactly once and in order: prompt[1:] followed by the emitted tokens.
func TestSpecGreedyAcceptReject(t *testing.T) {
	ids := []int64{5, 17, 300, 42, 8, 499, 13, 77, 256, 3, 64, 250, 19, 6, 400, 41, 2, 90}
	const n = 24
	cpu := device.CPU()
	m, err := LoadCausalLMFromHF("testdata/mimo/tiny", cpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	cache := NewKVCache(m.NumLayers())
	row := lastLogits(t, m, ids, cache, cpu)
	var want []int64
	for len(want) < n+2 {
		next := int64(top5(row)[0])
		want = append(want, next)
		row = lastLogits(t, m, []int64{next}, cache, cpu)
	}
	var got, fed []int64
	hiddenRows, calls := 0, 0
	st, err := specGreedy(m, ids, n, func(tok int64) bool { got = append(got, tok); return true },
		func(h *tensor.Tensor, next []int64) (int64, error) {
			if h.Dim(1) != len(next) {
				t.Fatalf("draft call %d: %d hidden rows for %d tokens", calls, h.Dim(1), len(next))
			}
			hiddenRows += h.Dim(1)
			fed = append(fed, next...)
			calls++
			k := len(fed) - (len(ids) - 1) // emitted tokens known so far
			if calls%3 == 0 {
				return (want[k] + 1) % 512, nil
			}
			return want[k], nil
		})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("drafted %d, accepted %d", st.Drafted, st.Accepted)
	if !slices.Equal(got, want[:n]) {
		t.Fatalf("speculative %v\nwant greedy %v", got, want[:n])
	}
	if st.Accepted == 0 || st.Accepted == st.Drafted {
		t.Fatalf("want both accepts and rejects, got %d / %d", st.Accepted, st.Drafted)
	}
	stream := append(slices.Clone(ids[1:]), want...)
	if !slices.Equal(fed, stream[:len(fed)]) {
		t.Fatalf("drafter fed %v\nwant prefix of %v", fed, stream)
	}
}
