package models

import (
	"encoding/json"
	"image/png"
	"math"
	"os"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/tensor"
)

// Gemma-4-E2B with an image (vision tower + embedder, pad-token lookups
// under the placeholders, image-block bidirectional sliding layers) against
// transformers F32 over the BF16 weights (scripts/verify/ref_gemma4_vl.py):
// features, placeholder layout, prefill top-5 and 16 greedy tokens. The LM
// keeps BF16 weight storage (exact for this checkpoint).
func TestGemma4ImageChatMatchesTransformers(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (gemma-4-E2B, ~10 GB)")
	}
	const dir = "testdata/gemma4-vision/"
	raw, err := os.ReadFile(dir + "vl.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Repo      string    `json:"repo"`
		IDs       []int64   `json:"ids"`
		FShape    []int     `json:"features_shape"`
		FHead     []float32 `json:"features_head"`
		FSum      float64   `json:"features_sum"`
		FAbsMean  float64   `json:"features_absmean"`
		Top5      []int     `json:"top5_ids"`
		Top5Lgt   []float64 `json:"top5_logits"`
		Greedy    []int64   `json:"greedy"`
		GreedyTxt string    `json:"greedy_text"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	cpu := device.CPU()
	vis, err := LoadGemma4VisionFromHF(ref.Repo, cpu)
	if err != nil {
		t.Skip(err)
	}
	f, err := os.Open(dir + "image.png")
	if err != nil {
		t.Fatal(err)
	}
	img, err := png.Decode(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	im, err := vis.Gemma4Preprocess(img)
	if err != nil {
		t.Fatal(err)
	}
	feats, err := vis.Forward([]Gemma4Image{im}, cpu)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(feats.Shape().Dims(), ref.FShape) {
		t.Fatalf("features %v, want %v", feats.Shape(), ref.FShape)
	}
	fv := tensor.MustSlice[float32](feats)
	md, ma, sum, abs := 0.0, 0.0, 0.0, 0.0
	for i, v := range fv {
		if i < len(ref.FHead) {
			md = math.Max(md, math.Abs(float64(v-ref.FHead[i])))
			ma = math.Max(ma, math.Abs(float64(ref.FHead[i])))
		}
		sum += float64(v)
		abs += math.Abs(float64(v))
	}
	abs /= float64(len(fv))
	t.Logf("features: head max diff %.3g (max |ref| %.3g); sum %.4f (ref %.4f), mean |x| %.6f (ref %.6f)", md, ma, sum, ref.FSum, abs, ref.FAbsMean)
	if md > 1e-2*ma || math.Abs(abs-ref.FAbsMean) > 1e-3*ref.FAbsMean {
		t.Fatalf("image features differ")
	}
	lay := vis.Gemma4ImageTokens(im, 255999, 258880, 258882)
	at := slices.Index(ref.IDs, 255999)
	if at < 0 || !slices.Equal(ref.IDs[at:at+len(lay)], lay) {
		t.Fatalf("image token layout differs from the processor's")
	}

	var lm *Gemma4
	if err := nn.WithWeightStorage16(dtype.BF16, func() error {
		var err error
		lm, err = LoadGemma4FromHF(ref.Repo, cpu, dtype.F32)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	ids, _ := tensor.New(ref.IDs, []int{1, len(ref.IDs)}, cpu)
	cache := NewKVCache(lm.NumLayers())
	logits, err := lm.ForwardMultimodal(ids, feats, 258880, 0, cache)
	if err != nil {
		t.Fatal(err)
	}
	last := lastLogitsRow(t, logits)
	got := top5(last)
	ld := 0.0
	for i, id := range ref.Top5 {
		ld = math.Max(ld, math.Abs(float64(last[id])-ref.Top5Lgt[i]))
	}
	t.Logf("top-5 %v (ref %v), logits within %.3g", got, ref.Top5, ld)
	if got[0] != ref.Top5[0] {
		t.Fatalf("prefill top-1 differs")
	}
	var gen []int64
	next := int64(got[0])
	for range len(ref.Greedy) {
		gen = append(gen, next)
		tok, _ := tensor.New([]int64{next}, []int{1, 1}, cpu)
		lg, err := lm.ForwardWithCache(tok, cache)
		if err != nil {
			t.Fatal(err)
		}
		next = int64(top5(lastLogitsRow(t, lg))[0])
	}
	t.Logf("greedy %v\n   ref %v (%q)", gen, ref.Greedy, ref.GreedyTxt)
	if !slices.Equal(gen, ref.Greedy) {
		t.Errorf("greedy reply differs")
	}
}
