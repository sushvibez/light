package models

import (
	"math"
	"testing"
)

// Reference values from transformers 5.7 for microsoft/Phi-3.5-mini-instruct
// (rope_type longrope, head_dim 96, base 1e4, 4096 → 131072 positions):
// model.model.rotary_emb.inv_freq[:4], inv_freq[-2:], attention_scaling.
func TestLongRopePhi35(t *testing.T) {
	short := []float64{1.0, 1.0199999809265137, 1.0299999713897705, 1.0299999713897705, 1.0499999523162842, 1.0499999523162842, 1.0499999523162842, 1.0499999523162842, 1.0499999523162842, 1.0699999332427979, 1.0999999046325684, 1.1099998950958252, 1.1599998474121094, 1.1599998474121094, 1.1699998378753662, 1.2899998426437378, 1.339999794960022, 1.679999828338623, 1.7899998426437378, 1.8199998140335083, 1.8499997854232788, 1.8799997568130493, 1.9099997282028198, 1.9399996995925903, 1.9899996519088745, 2.0199997425079346, 2.0199997425079346, 2.0199997425079346, 2.0199997425079346, 2.0199997425079346, 2.0199997425079346, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0799996852874756, 2.0899996757507324, 2.189999580383301, 2.2199995517730713, 2.5899994373321533, 2.729999542236328, 2.749999523162842, 2.8399994373321533}
	s := RopeScaling{Type: "longrope", ShortFactor: short, LongFactor: short, OriginalMaxPositions: 4096, MaxPositions: 131072}
	inv, err := s.InvFreq(96, 10000)
	if err != nil {
		t.Fatal(err)
	}
	want := map[int]float64{0: 1.0, 1: 0.8092197775840759, 2: 0.6614486575126648, 3: 0.5459624528884888, 46: 5.337453694664873e-05, 47: 4.2659426981117576e-05}
	for i, w := range want {
		if d := math.Abs(float64(inv[i]) - w); d > 1e-6*math.Max(1, w) {
			t.Errorf("inv[%d] = %g, want %g", i, inv[i], w)
		}
	}
	if af := s.AttentionFactor(); math.Abs(af-1.1902380714238083) > 1e-9 {
		t.Errorf("AttentionFactor = %v, want 1.1902380714238083", af)
	}
	if af := (RopeScaling{Type: "llama3"}).AttentionFactor(); af != 1 {
		t.Errorf("llama3 AttentionFactor = %v, want 1", af)
	}
	if _, err := s.InvFreq(64, 10000); err == nil {
		t.Error("expected error for factor count mismatch")
	}
}

// Reference values from transformers' ROPE_INIT_FUNCTIONS["llama3"] for
// meta-llama/Llama-3.2-1B-Instruct (head_dim 64, theta 500000, factor 32,
// low 1, high 4, original 8192).
func TestLlama3InvFreqMatchesTransformers(t *testing.T) {
	s := RopeScaling{Type: "llama3", Factor: 32, LowFreqFactor: 1, HighFreqFactor: 4, OriginalMaxPositions: 8192}
	inv, err := s.InvFreq(64, 500000)
	if err != nil {
		t.Fatal(err)
	}
	if len(inv) != 32 {
		t.Fatalf("len = %d, want 32", len(inv))
	}
	want := map[int]float64{
		0: 1.0, 1: 0.663601279258728, 2: 0.44036662578582764, 3: 0.2922278344631195,
		12: 0.00729266507551074, 13: 0.00483942124992609, 14: 0.0032114461064338684, 15: 0.0012905480107292533,
		28: 3.2229328894572973e-07, 29: 2.1387423032592778e-07, 30: 1.4192720243499934e-07, 31: 9.418306490260875e-08,
	}
	for i, w := range want {
		got := float64(inv[i])
		if math.Abs(got-w) > 1e-6*math.Max(1, math.Abs(w))+1e-12 {
			t.Errorf("invFreq[%d] = %v, want %v", i, got, w)
		}
	}
	// High-frequency pairs are untouched; the lowest are divided by factor.
	base := BaseInvFreq(64, 500000)
	if inv[0] != base[0] || inv[3] != base[3] {
		t.Errorf("high-frequency entries changed: %v vs %v", inv[:4], base[:4])
	}
	if r := float64(base[31]) / float64(inv[31]); math.Abs(r-32) > 1e-3 {
		t.Errorf("lowest frequency ratio = %v, want 32", r)
	}
}

func TestRopeScalingUnsupported(t *testing.T) {
	if _, err := (RopeScaling{Type: "yarn"}).InvFreq(64, 10000); err == nil {
		t.Error("expected error for unsupported type")
	}
}
