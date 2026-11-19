package models

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
)

// LLaDA-8B-Instruct / Dream-v0-Instruct-7B on real weights
// (scripts/colab/masked_diffusion_check.py downloads a release and records
// its own F32 forward and a deterministic generation): the first masked
// positions' logits, the argmax everywhere and the generated tokens. Runs
// when LIGHT_MASKED_DIR names the folder (LIGHT_MASKED_REF overrides
// ref.json's path).
func TestMaskedDiffusionRealMatchesReference(t *testing.T) {
	dir := os.Getenv("LIGHT_MASKED_DIR")
	if dir == "" {
		t.Skip("LIGHT_MASKED_DIR not set")
	}
	refPath := os.Getenv("LIGHT_MASKED_REF")
	if refPath == "" {
		refPath = filepath.Join(dir, "ref.json")
	}
	raw, err := os.ReadFile(refPath)
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Kind        string
		IDs         []int64
		MaskID      int64       `json:"mask_id"`
		GenLength   int         `json:"gen_length"`
		FirstLogits [][]float32 `json:"first_logits"`
		Argmax      []int64
		Gen         []int64
		Text        string
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	m, err := LoadCausalLMFromHF(dir, device.CPU(), dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	canvas := append(append([]int64(nil), ref.IDs...), make([]int64, ref.GenLength)...)
	for i := len(ref.IDs); i < len(canvas); i++ {
		canvas[i] = ref.MaskID
	}
	lg, err := m.bidiLogits([][]int64{canvas})
	if err != nil {
		t.Fatal(err)
	}
	V := len(lg[0]) / len(canvas)
	first := len(ref.IDs)
	if ref.Kind == "dream" {
		first-- // Dream's shifted logits: the first mask reads the prompt's last position
	}
	md, mr := 0.0, 0.0
	for r, row := range ref.FirstLogits {
		got := lg[0][(first+r)*V : (first+r+1)*V]
		for i, w := range row {
			md = math.Max(md, math.Abs(float64(got[i]-w)))
			mr = math.Max(mr, math.Abs(float64(w)))
		}
	}
	mism := 0
	for p := range canvas {
		row := lg[0][p*V : (p+1)*V]
		best := 0
		for i, v := range row {
			if v > row[best] {
				best = i
			}
		}
		if int64(best) != ref.Argmax[p] {
			mism++
		}
	}
	o := MaskedDiffusionOpts{MaskID: ref.MaskID, GenLength: ref.GenLength, Steps: ref.GenLength}
	var gen []int64
	if ref.Kind == "llada" {
		gen, err = LLaDAGenerate(m, ref.IDs, o)
	} else {
		o.Alg = "entropy"
		gen, err = DreamGenerate(m, ref.IDs, o)
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%s: first masked logits max diff %.3g (max |ref| %.3g); argmax mismatches %d / %d; gen %v, reference %v (%q)",
		ref.Kind, md, mr, mism, len(canvas), gen, ref.Gen, ref.Text)
	if !(md <= 1e-3*math.Max(mr, 1)) {
		t.Errorf("logits differ by %.3g", md)
	}
	if mism > 0 {
		t.Errorf("%d argmax mismatches", mism)
	}
	if fmt.Sprint(gen) != fmt.Sprint(ref.Gen) {
		t.Errorf("generation differs")
	}
}
