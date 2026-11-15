package models

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// A tiny 4-bit Gemma 4 in gemma-4-31B's layout (scripts/verify/ref_gemma4_int4.py:
// GPTQ projections from layer 1, layer 0 in BF16) against transformers on
// the dequantized model: every position's logits.
func TestGemma4Int4TinyMatchesDequantized(t *testing.T) {
	raw, err := os.ReadFile("testdata/gemma4-int4-tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		IDs    []int64
		Logits []float32
		Vocab  int
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	m, err := LoadGemma4FromHF("testdata/gemma4-int4-tiny", device.CPU(), dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	if m.Blocks[1].QProj.W4 == nil || m.Blocks[0].QProj.W4 != nil {
		t.Fatalf("4-bit layout: layer 1 q_proj 4-bit %v, layer 0 %v", m.Blocks[1].QProj.W4 != nil, m.Blocks[0].QProj.W4 != nil)
	}
	ids, err := tensor.New(ref.IDs, []int{1, len(ref.IDs)}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	out, err := m.Forward(ids)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](out)
	if len(got) != len(ref.Logits) {
		t.Fatalf("%d logits, want %d", len(got), len(ref.Logits))
	}
	md := 0.0
	for i := range got {
		md = math.Max(md, math.Abs(float64(got[i]-ref.Logits[i])))
	}
	t.Logf("logits: max diff %.3g over %d", md, len(got))
	if md > 1e-3 {
		t.Errorf("logits differ by %.3g", md)
	}
}
