package models

import (
	"math/rand"
	"testing"

	"light/device"
	"light/tensor"
)

// Tiny random EmbeddingGemma 2 on CPU F32: a single sliding layer with
// radius w must make position j independent of tokens farther than w away
// in either direction, and a full-attention layer must not.
func TestEmbeddingGemma2SlidingMask(t *testing.T) {
	for _, sliding := range []bool{true, false} {
		cfg := EmbeddingGemma2Config{
			VocabSize: 50, HiddenSize: 16, IntermediateSize: 32, NumLayers: 1, NumHeads: 2, NumKVHeads: 1,
			NumGlobalKVHeads: 1, HeadDim: 8, GlobalHeadDim: 16, MaxPosition: 64, RopeBase: 1e6, RopeLocalBase: 1e4,
			RmsEps: 1e-6, SlidingWindow: 3, SlidingLayers: []bool{sliding}, PLEDim: 8, EmbeddingDim: 12,
		}
		m, err := NewEmbeddingGemma2(cfg, device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		rng := rand.New(rand.NewSource(7))
		for _, p := range m.Parameters() {
			data := make([]float32, p.NumElements())
			for i := range data {
				data[i] = float32(rng.NormFloat64()) * 0.3
			}
			if p.NumElements() == 1 {
				data[0] = 1
			}
			src, _ := tensor.New(data, p.Shape().Dims(), device.CPU())
			if err := p.CopyFrom(src); err != nil {
				t.Fatal(err)
			}
		}
		const T = 12
		ids := make([]int64, T)
		for i := range ids {
			ids[i] = int64(rng.Intn(50))
		}
		run := func(ids []int64) []float32 {
			tk, _ := tensor.New(ids, []int{1, T}, device.CPU())
			h, err := m.Forward(tk, nil)
			if err != nil {
				t.Fatal(err)
			}
			return tensor.MustSlice[float32](h)
		}
		base := run(ids)
		ids2 := append([]int64(nil), ids...)
		ids2[0] = (ids2[0] + 1) % 50
		changed := run(ids2)
		D := cfg.EmbeddingDim
		for j := 1; j < T; j++ {
			d := maxAbsDiff(base[j*D:(j+1)*D], changed[j*D:(j+1)*D])
			far := j > cfg.SlidingWindow
			switch {
			case sliding && far && d != 0:
				t.Errorf("sliding: position %d changed by %.2e when token 0 changed (radius %d)", j, d, cfg.SlidingWindow)
			case (!sliding || !far) && d == 0:
				t.Errorf("sliding=%v: position %d unaffected by token 0", sliding, j)
			}
		}
	}
}

// BF16 on CUDA agrees with the F32 CPU forward of the same tiny model.
func TestEmbeddingGemma2TinyCUDAParity(t *testing.T) {
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	cfg := EmbeddingGemma2Config{
		VocabSize: 50, HiddenSize: 16, IntermediateSize: 32, NumLayers: 3, NumHeads: 2, NumKVHeads: 1,
		NumGlobalKVHeads: 1, HeadDim: 8, GlobalHeadDim: 16, MaxPosition: 64, RopeBase: 1e6, RopeLocalBase: 1e4,
		RmsEps: 1e-6, SlidingWindow: 4, SlidingLayers: []bool{true, false, true}, PLEDim: 8, EmbeddingDim: 12,
	}
	rng := rand.New(rand.NewSource(3))
	mc, err := NewEmbeddingGemma2(cfg, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	mg, err := NewEmbeddingGemma2(cfg, gpu)
	if err != nil {
		t.Fatal(err)
	}
	pc, pg := mc.Parameters(), mg.Parameters()
	for i, p := range pc {
		data := make([]float32, p.NumElements())
		for k := range data {
			data[k] = float32(rng.NormFloat64()) * 0.3
		}
		if len(data) == 1 {
			data[0] = 1
		}
		src, _ := tensor.New(data, p.Shape().Dims(), device.CPU())
		if err := p.CopyFrom(src); err != nil {
			t.Fatal(err)
		}
		g, err := src.To(gpu)
		if err != nil {
			t.Fatal(err)
		}
		if err := pg[i].CopyFrom(g); err != nil {
			t.Fatal(err)
		}
	}
	const B, T = 2, 10
	ids := make([]int64, B*T)
	keep := make([]int64, B*T)
	for i := range ids {
		ids[i] = int64(rng.Intn(50))
		keep[i] = 1
	}
	for j := 7; j < T; j++ { // pad the second row
		keep[T+j] = 0
	}
	tc, _ := tensor.New(ids, []int{B, T}, device.CPU())
	kc, _ := tensor.New(keep, []int{B, T}, device.CPU())
	ec, err := mc.Embed(tc, kc)
	if err != nil {
		t.Fatal(err)
	}
	tg, _ := tc.To(gpu)
	kg, _ := kc.To(gpu)
	eg, err := mg.Embed(tg, kg)
	if err != nil {
		t.Fatal(err)
	}
	if d := maxAbsDiff(tensor.MustSlice[float32](ec), toCPUSlice(t, eg)); d > 1e-4 {
		t.Fatalf("CPU/CUDA tiny parity differ by %.2e", d)
	}
}
