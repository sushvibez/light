package models

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
)

// Qwen3-MoE on the tiny random hf-internal-testing checkpoint (2 layers,
// 4 experts, top-2, fused transformers-5 expert layout, 39 MB). Random
// weights are fine for an architecture check: every routed expert, the
// renormalised top-k weights and the QK-norm attention have to line up for
// the logits to match. Reference: scripts/verify/ref_qwen3_moe.py
// (transformers 5.21, router copied from `mlp.router.weight`).
const qwen3MoERepo = "hf-internal-testing/tiny-random-Qwen3MoeForCausalLM"

var qwen3MoEPromptIDs = []int64{785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 1576, 432, 4829, 311, 5545, 279, 1008, 3108, 315, 279, 14796}

var qwen3MoERefF32 = []struct {
	id    int
	logit float32
}{{148776, 0.68822}, {122188, 0.66155}, {113771, 0.65596}, {13854, 0.65313}, {19119, 0.64669}}

func qwen3MoECached(t *testing.T) {
	t.Helper()
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--hf-internal-testing--tiny-random-Qwen3MoeForCausalLM/snapshots/*/model.safetensors")); len(m) == 0 {
		t.Skip(qwen3MoERepo + " not in HF cache")
	}
	os.Setenv("HF_HUB_OFFLINE", "1")
}

func checkQwen3MoE(t *testing.T, dev device.Device, wdt dtype.DType, logitTol float64, idsMustMatch int) {
	t.Helper()
	m, err := LoadCausalLMFromHF(qwen3MoERepo, dev, wdt)
	if err != nil {
		t.Fatal(err)
	}
	if m.Cfg.NumExperts != 4 || m.Cfg.NumExpertsPerTok != 2 || !m.Cfg.NormTopKProb || m.Blocks[0].MoE == nil || m.Blocks[0].GateProj != nil {
		t.Fatalf("MoE config not applied: %+v", m.Cfg)
	}
	row := lastLogits(t, m, qwen3MoEPromptIDs, NewKVCache(m.NumLayers()), dev)
	got := top5(row)
	for i, w := range qwen3MoERefF32 {
		if i < idsMustMatch && got[i] != w.id {
			t.Errorf("top-%d: got id %d (%.5f), want %d (%.5f)", i+1, got[i], row[got[i]], w.id, w.logit)
		}
		if d := math.Abs(float64(row[w.id] - w.logit)); d > logitTol {
			t.Errorf("logit[%d] = %.5f, want %.5f (tol %g)", w.id, row[w.id], w.logit, logitTol)
		}
	}
	t.Logf("%v %v top-5 %v logits %.5f %.5f %.5f %.5f %.5f", dev, wdt, got, row[got[0]], row[got[1]], row[got[2]], row[got[3]], row[got[4]])
}

func TestQwen3MoEMatchesTransformersCPU(t *testing.T) {
	qwen3MoECached(t)
	checkQwen3MoE(t, device.CPU(), dtype.F32, 2e-3, 5)
}

func TestQwen3MoECUDA(t *testing.T) {
	qwen3MoECached(t)
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	checkQwen3MoE(t, dev, dtype.F32, 2e-3, 5)
	// BF16: the random-weight logits sit within 0.05 of each other, so only
	// the top-1 id is pinned; values must still be within BF16 noise.
	checkQwen3MoE(t, dev, dtype.BF16, 2e-2, 1)
}
