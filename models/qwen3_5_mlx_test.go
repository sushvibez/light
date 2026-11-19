package models

import (
	"encoding/json"
	"math"
	"os"
	"strconv"
	"testing"

	"light/device"
	"light/dtype"
	"light/tensor"
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

// The 4-bit experts streamed through slots run the kernels the resident
// 4-bit experts run: with a slot per expert (no token chunks) the same
// logits bit for bit on the prompt and on single-token decode steps; with
// 3 for the adapter's top-3 (the prompt runs in token chunks, so the
// fused kernel's row tiling differs; evictions, decode-step prefetch)
// within 1e-5. The slots stay packed (15 KB an expert here, 6.4x less
// than F32) and no mispredicted prefetch lingers.
func TestEdge0TinyStreamedQ4MatchesResident(t *testing.T) {
	ref := loadEdge0TinyRef(t)
	cpu := device.CPU()
	run := func(m *Qwen35) [][]float32 {
		cache := NewKVCache(m.NumLayers())
		n := len(ref.IDs) - 3
		rows := [][]float32{append([]float32(nil), lastLogits(t, m, ref.IDs[:n], cache, cpu)...)}
		for _, id := range ref.IDs[n:] {
			rows = append(rows, append([]float32(nil), lastLogits(t, m, []int64{id}, cache, cpu)...))
		}
		return rows
	}
	t.Setenv("LIGHT_EXPERT_SLOTS", "")
	res, err := LoadQwen35FromHF(edge0TinyDir, cpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	want := run(res)
	checkEdge0Logits(t, want[len(want)-1], ref.LoRA, 1e-4)
	for _, slots := range []int{2, 3, 8} {
		// Top-3 routing needs 3 slots; 2 must be refused.
		t.Setenv("LIGHT_EXPERT_SLOTS", strconv.Itoa(slots))
		m, err := LoadQwen35FromHF(edge0TinyDir, cpu, dtype.F32)
		if err != nil {
			t.Fatal(err)
		}
		if slots < m.Blocks[0].MoE.TopK {
			x, _ := tensor.New(ref.IDs[:1], []int{1, 1}, cpu)
			if _, err := m.ForwardWithCache(x, NewKVCache(m.NumLayers())); err == nil {
				t.Errorf("%d slots for top-%d routing should fail", slots, m.Blocks[0].MoE.TopK)
			}
			continue
		}
		got := run(m)
		tol, md := 1e-5, 0.0
		if slots == m.Cfg.NumExperts {
			tol = 0
		}
		for i := range want {
			for j := range want[i] {
				md = math.Max(md, math.Abs(float64(got[i][j]-want[i][j])))
			}
		}
		if md > tol {
			t.Errorf("%d slots: max |Δlogit| %.3g against resident (tolerance %g)", slots, md, tol)
		}
		var loads, hits, pf, pfUsed, bytes int64
		for _, b := range m.Blocks {
			s := b.MoE.Stream
			if s == nil || s.LoadQ4 == nil {
				t.Fatalf("layer experts not streamed 4-bit")
			}
			loads, hits, pf, pfUsed, bytes = loads+s.Loads, hits+s.Hits, pf+s.Prefetched, pfUsed+s.PrefetchUsed, bytes+s.SlotBytes()
			if n := len(s.Resident()); n > slots {
				t.Fatalf("%d experts resident in %d slots", n, slots)
			}
			if p := s.Pending(); p > slots {
				t.Fatalf("%d loads pending for %d slots", p, slots)
			}
		}
		// Per expert: gate-up codes (128, 16) U32 + 2 x (128, 2) F32, down
		// codes (128, 8) + 2 x (128, 1).
		const perExpert = 128*16*4 + 2*128*2*4 + 128*8*4 + 2*128*1*4
		t.Logf("%d slots: max |Δlogit| %.3g vs resident; %d loads, %d hits, %d prefetched (%d used); slots %d B (%d B an expert, F32 %d)",
			slots, md, loads, hits, pf, pfUsed, bytes, perExpert, (128*128+128*64)*4)
		if bytes > int64(len(m.Blocks)*slots*perExpert) || loads == 0 || hits == 0 || pfUsed > pf ||
			(pf == 0) != (slots == m.Cfg.NumExperts) { // all resident after the prompt: nothing to prefetch
			t.Errorf("slots %d B (budget %d), loads %d, hits %d, prefetched %d / used %d",
				bytes, len(m.Blocks)*slots*perExpert, loads, hits, pf, pfUsed)
		}
	}
}
