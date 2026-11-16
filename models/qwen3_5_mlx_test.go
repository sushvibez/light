package models

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"light/device"
	"light/dtype"
)

// Tiny Edge0-35B-A3B-shaped checkpoint from scripts/verify/make_edge0_tiny.py:
// qwen3_5_moe in MLX's layout (4-bit affine, 8-bit router and shared-expert
// gate, stacked switch_mlp experts, norms saved with the 1 added) and an
// edge0 LoRA adapter (lora_tiny.safetensors, scale alpha / r = 2, K = 3).
// The reference is transformers on the dequantized weights, top-2 without
// the adapter and top-3 with it merged.
const edge0TinyDir = "testdata/edge0-tiny"

type edge0TinyRef struct {
	IDs  []int64   `json:"ids"`
	Base []float32 `json:"base"`
	LoRA []float32 `json:"lora"`
}

func loadEdge0TinyRef(t *testing.T) edge0TinyRef {
	t.Helper()
	raw, err := os.ReadFile("testdata/edge0-tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	var r edge0TinyRef
	if err := json.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

// checkEdge0Logits compares a last-position row with the reference: the
// same argmax and every logit within tol.
func checkEdge0Logits(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	maxDiff := 0.0
	for i := range want {
		maxDiff = math.Max(maxDiff, math.Abs(float64(got[i]-want[i])))
	}
	g, w := top5(got), top5(want)
	t.Logf("top-5 %v (want %v), max |Δlogit| %.2g", g, w, maxDiff)
	if g[0] != w[0] || maxDiff > tol {
		t.Errorf("argmax %d (want %d), max |Δlogit| %.3g > %.g", g[0], w[0], maxDiff, tol)
	}
}

func TestEdge0TinyMLXMatchesTransformers(t *testing.T) {
	t.Setenv("LIGHT_LORA", "off")
	ref := loadEdge0TinyRef(t)
	cpu := device.CPU()
	m, err := LoadQwen35FromHF(edge0TinyDir, cpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	b := m.Blocks[0]
	if m.Cfg.NumExperts != 8 || b.MoE.TopK != 2 || b.MoE.GateUpQ4 == nil || b.InQKV.W4 == nil || m.LMHead.W4 == nil ||
		b.MoE.Gate.W == nil || b.InQKV.LoRA != nil {
		t.Fatalf("layout: experts %d top-%d, 4-bit experts %v, qkv %v, head %v, router dense %v",
			m.Cfg.NumExperts, b.MoE.TopK, b.MoE.GateUpQ4 != nil, b.InQKV.W4 != nil, m.LMHead.W4 != nil, b.MoE.Gate.W != nil)
	}
	checkEdge0Logits(t, lastLogits(t, m, ref.IDs, NewKVCache(m.NumLayers()), cpu), ref.Base, 1e-4)

	// BF16 activations (the 4-bit kernels and expanded weights at BF16):
	// rounding only, 0.13 at most here.
	if m, err = LoadQwen35FromHF(edge0TinyDir, cpu, dtype.BF16); err != nil {
		t.Fatal(err)
	}
	checkEdge0Logits(t, lastLogits(t, m, ref.IDs, NewKVCache(m.NumLayers()), cpu), ref.Base, 0.2)

	// The CLI's CPU default (w16): F32 compute, the dense weights narrowed
	// to BF16 storage (the expanded 8-bit router and embedding round), the
	// 4-bit ones left packed: 0.061 at most here.
	if m, err = LoadQwen35FromHF(edge0TinyDir, cpu, dtype.F32); err != nil {
		t.Fatal(err)
	}
	if _, err := StoreWeights16(m, dtype.BF16); err != nil {
		t.Fatal(err)
	}
	checkEdge0Logits(t, lastLogits(t, m, ref.IDs, NewKVCache(m.NumLayers()), cpu), ref.Base, 0.1)

	// The engine's entry point dispatches qwen3_5_moe here.
	if lm, err := LoadChatModelFromHF(edge0TinyDir, cpu, dtype.F32); err != nil {
		t.Fatal(err)
	} else if _, ok := lm.(*Qwen35); !ok {
		t.Fatalf("LoadChatModelFromHF: %T", lm)
	}
}

func TestEdge0TinyLoRAMatchesTransformers(t *testing.T) {
	ref := loadEdge0TinyRef(t)
	cpu := device.CPU()
	for _, slots := range []string{"", "3"} {
		t.Run("slots="+slots, func(t *testing.T) {
			t.Setenv("LIGHT_EXPERT_SLOTS", slots)
			m, err := LoadQwen35FromHF(edge0TinyDir, cpu, dtype.F32)
			if err != nil {
				t.Fatal(err)
			}
			b := m.Blocks[1]
			if b.MoE.TopK != 3 || b.QProj.LoRA == nil || b.Gate.LoRA == nil || b.MoE.Gate.LoRA != nil || (slots != "") != (b.MoE.Stream != nil) {
				t.Fatalf("top-%d, q_proj LoRA %v, shared gate_proj LoRA %v, stream %v", b.MoE.TopK, b.QProj.LoRA != nil, b.Gate.LoRA != nil, b.MoE.Stream != nil)
			}
			checkEdge0Logits(t, lastLogits(t, m, ref.IDs, NewKVCache(m.NumLayers()), cpu), ref.LoRA, 1e-4)
			// Prefill all but the last token, then decode it.
			cache := NewKVCache(m.NumLayers())
			n := len(ref.IDs)
			lastLogits(t, m, ref.IDs[:n-1], cache, cpu)
			checkEdge0Logits(t, lastLogits(t, m, ref.IDs[n-1:], cache, cpu), ref.LoRA, 1e-4)
		})
	}
}
