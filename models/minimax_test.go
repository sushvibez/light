package models

import (
	"math"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
	"light/safetensors"
	"light/tensor"
)

// Tiny-random MiniMax written by `scripts/verify/ref_minimax.py --make-tiny`,
// reference values from `ref_minimax.py testdata/minimax/tiny f32 --ids …`
// (transformers 5.21, eager, CPU): lightning / softmax attention alternating
// over 4 layers, lightning block_size 4 (the 18-token prefill crosses five
// blocks of transformers' block algorithm; Light's own chunk length is
// independent, see TestMiniMaxLightningChunks), residual factors ≠ 1, GQA 4 / 2 heads of 16, 4 experts top-2,
// perturbed norms. The real MiniMax-Text-01 (456 B) is out of reach here.
var (
	minimaxTinyRef = []struct {
		id    int
		logit float32
	}{{105, 0.5880}, {466, 0.4734}, {239, 0.3914}, {66, 0.3672}, {7, 0.3668}}
	minimaxTinyGreedy = []int64{105, 54, 347, 227, 269, 349, 321, 29}
)

const minimaxTinyDir = "testdata/minimax/tiny"

func TestMiniMaxTinyMatchesTransformers(t *testing.T) {
	ids := []int64{5, 17, 300, 42, 8, 499, 13, 77, 256, 3, 64, 250, 19, 6, 400, 41, 2, 90}
	cpu := device.CPU()
	m, err := LoadMiniMaxFromHF(minimaxTinyDir, cpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cfg
	if !slices.Equal(c.LinearLayers, []bool{true, false, true, false}) || c.LinearAlpha != 1.2 || c.MLPBeta != 0.7 ||
		c.NumExperts != 4 || c.TopK != 2 || c.HeadDim != 16 || c.NumKVHeads != 2 {
		t.Fatalf("config %+v", c)
	}
	// The derived decay rates equal the checkpoint's slope_rate buffers.
	st, _, err := safetensors.LoadFiles([]string{minimaxTinyDir + "/model.safetensors"}, cpu)
	if err != nil {
		t.Fatal(err)
	}
	for _, l := range []int{0, 2} {
		buf := tensor.MustSlice[float32](st["model.layers."+string(rune('0'+l))+".self_attn.slope_rate"])
		for h, s := range m.Blocks[l].slopes {
			if math.Abs(s-float64(buf[h])) > 1e-7 {
				t.Errorf("layer %d head %d slope %.8g, checkpoint %.8g", l, h, s, buf[h])
			}
		}
	}

	full := lastLogits(t, m, ids, NewKVCache(m.NumLayers()), cpu)
	got := top5(full)
	for i, w := range minimaxTinyRef {
		if i < 3 && got[i] != w.id {
			t.Errorf("top-%d: got %d (%.4f), want %d (%.4f)", i+1, got[i], full[got[i]], w.id, w.logit)
		}
		if d := full[w.id] - w.logit; d > 2e-4 || d < -2e-4 {
			t.Errorf("logit[%d] = %.5f, want %.4f", w.id, full[w.id], w.logit)
		}
	}
	// Greedy decode: one-token steps carry the lightning state and the KV
	// cache.
	cache := NewKVCache(m.NumLayers())
	step := lastLogits(t, m, ids, cache, cpu)
	var greedy []int64
	for len(greedy) < len(minimaxTinyGreedy) {
		next := int64(top5(step)[0])
		greedy = append(greedy, next)
		step = lastLogits(t, m, []int64{next}, cache, cpu)
	}
	if !slices.Equal(greedy, minimaxTinyGreedy) {
		t.Errorf("greedy %v, want %v", greedy, minimaxTinyGreedy)
	}
	// Prefill in uneven pieces, then single tokens == single shot.
	cache = NewKVCache(m.NumLayers())
	lastLogits(t, m, ids[:7], cache, cpu)
	lastLogits(t, m, ids[7:12], cache, cpu)
	var chunked []float32
	for i := 12; i < len(ids); i++ {
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
	mg, err := LoadMiniMaxFromHF(minimaxTinyDir, gpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	if d := maxAbsDiff(full, lastLogits(t, mg, ids, NewKVCache(mg.NumLayers()), gpu)); d > 1e-4 {
		t.Errorf("CUDA vs CPU logits differ by %.2e", d)
	}
}

// Lightning attention in chunks (the state carried across chunk
// boundaries) equals one chunk over the whole prompt, for chunk lengths
// that do and do not divide the 18-token prompt, with and without a cache.
func TestMiniMaxLightningChunks(t *testing.T) {
	ids := []int64{5, 17, 300, 42, 8, 499, 13, 77, 256, 3, 64, 250, 19, 6, 400, 41, 2, 90}
	cpu := device.CPU()
	m, err := LoadMiniMaxFromHF(minimaxTinyDir, cpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	defer func(c int) { lightningChunk = c }(lightningChunk)
	lightningChunk = 256
	whole := lastLogits(t, m, ids, NewKVCache(m.NumLayers()), cpu)
	for _, c := range []int{1, 4, 5, 7} {
		lightningChunk = c
		if d := maxAbsDiff(whole, lastLogits(t, m, ids, NewKVCache(m.NumLayers()), cpu)); d > 1e-5 {
			t.Errorf("chunk %d (cache) differs from one chunk by %.2e", c, d)
		}
		if d := maxAbsDiff(whole, lastLogits(t, m, ids, nil, cpu)); d > 1e-5 {
			t.Errorf("chunk %d (no cache) differs from one chunk by %.2e", c, d)
		}
	}
	if !device.CUDAAvailable() {
		return
	}
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	mg, err := LoadMiniMaxFromHF(minimaxTinyDir, gpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	lightningChunk = 5
	if d := maxAbsDiff(whole, lastLogits(t, mg, ids, NewKVCache(mg.NumLayers()), gpu)); d > 1e-4 {
		t.Errorf("CUDA chunk 5 vs CPU differ by %.2e", d)
	}
}
