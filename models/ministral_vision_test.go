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
	"light/tensor"
)

// Ministral-3-3B with an image (Pixtral tower + Mistral3 projector,
// placeholder rows, Llama decoder) against transformers F32 over the BF16
// weights (scripts/verify/ref_ministral_vl.py): projector output, the
// [IMG] / [IMG_BREAK] / [IMG_END] layout, prefill top-5 and 16 greedy
// tokens.
func TestMinistralImageChatMatchesTransformers(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (Ministral-3B at F32, ~15 GB)")
	}
	const dir = "testdata/ministral-vision/"
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
	vis, err := LoadPixtralFromHF(ref.Repo, cpu)
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
	im, err := vis.PixtralPreprocess(img)
	if err != nil {
		t.Fatal(err)
	}
	feats, err := vis.Forward([]PixtralImage{im}, cpu)
	if err != nil {
		t.Fatal(err)
	}
	fv := tensor.MustSlice[float32](feats)
	if !slices.Equal(feats.Shape().Dims(), ref.FShape) {
		t.Fatalf("features %v, want %v", feats.Shape(), ref.FShape)
	}
	md, sum, abs := 0.0, 0.0, 0.0
	for i, v := range fv {
		if i < len(ref.FHead) {
			md = math.Max(md, math.Abs(float64(v-ref.FHead[i])))
		}
		sum += float64(v)
		abs += math.Abs(float64(v))
	}
	abs /= float64(len(fv))
	t.Logf("features: head max diff %.3g; sum %.4f (ref %.4f), mean |x| %.6f (ref %.6f)", md, sum, ref.FSum, abs, ref.FAbsMean)
	if md > 2e-3 || math.Abs(abs-ref.FAbsMean) > 1e-4*ref.FAbsMean+1e-6 {
		t.Fatalf("projector output differs")
	}
	// The placeholder run in the reference prompt equals PixtralTokens.
	lay := vis.PixtralTokens(im, 10, 12, 13)
	at := slices.Index(ref.IDs, 10)
	if at < 0 || !slices.Equal(ref.IDs[at:at+len(lay)], lay) {
		t.Fatalf("image token layout differs from the processor's")
	}

	lm, err := LoadMinistralFromHF(ref.Repo, cpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	ids, _ := tensor.New(ref.IDs, []int{1, len(ref.IDs)}, cpu)
	h, err := lm.TokenEmb.Forward(ids)
	if err != nil {
		t.Fatal(err)
	}
	if h, err = scatterImage(h, ref.IDs, 10, feats); err != nil {
		t.Fatal(err)
	}
	cache := NewKVCache(lm.NumLayers())
	logits, err := lm.ForwardEmbeds(h, cache)
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
