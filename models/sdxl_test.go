package models

import (
	"math"
	"os"
	"testing"

	"light/device"
	"light/dtype"
	"light/npy"
	"light/ops"
	"light/tensor"
)

// TestSDXLMatchesDiffusers checks the SDXL port against
// scripts/verify/ref_sdxl.py (LIGHT_SDXL_REF=<npz>): both tokenizers' ids,
// the text encoders' penultimate hidden states and bigG's projected pooled
// output (F32, tight), and one UNet noise prediction (reference fp16 on the
// GPU, Light BF16 on CUDA: compared on mean absolute error relative to the
// prediction's scale). LIGHT_HEAVY_TESTS=1: ~9 GB on the card.
func TestSDXLMatchesDiffusers(t *testing.T) {
	path := os.Getenv("LIGHT_SDXL_REF")
	if path == "" || os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_SDXL_REF (scripts/verify/ref_sdxl.py) and LIGHT_HEAVY_TESTS=1")
	}
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	cpu := device.CPU()
	ref, err := npy.ReadNpz(path, cpu)
	if err != nil {
		t.Fatal(err)
	}
	releaseModelMemory(t)
	sd, err := LoadSDXLFromHF(SDXLRepo, dtype.BF16, dev)
	if err != nil {
		t.Fatal(err)
	}
	prompt := "a photograph of an astronaut riding a horse on the moon"
	ids1 := sd.Tokenizer1.Encode(prompt, 77)
	ids2 := sd.Tokenizer2.EncodeWithPad(prompt, 77, 0)
	for i, want := range tensor.MustSlice[int64](ref["ids1"]) {
		if int64(ids1[i]) != want {
			t.Fatalf("tokenizer 1 id %d: %d, want %d", i, ids1[i], want)
		}
	}
	for i, want := range tensor.MustSlice[int64](ref["ids2"]) {
		if int64(ids2[i]) != want {
			t.Fatalf("tokenizer 2 id %d: %d, want %d", i, ids2[i], want)
		}
	}
	t1, _ := tokensToI64Tensor(ids1, dev)
	t2, _ := tokensToI64Tensor(ids2, dev)
	h1, _, err := sd.TextEncoder1.ForwardPenultimate(t1)
	if err != nil {
		t.Fatal(err)
	}
	h2, pooled, err := sd.TextEncoder2.ForwardPenultimate(t2)
	if err != nil {
		t.Fatal(err)
	}
	check := func(name string, got, want *tensor.Tensor, tol float64) {
		gc, _ := got.To(cpu)
		g, w := tensor.MustSlice[float32](gc), tensor.MustSlice[float32](want)
		if len(g) != len(w) {
			t.Fatalf("%s: %d values, reference %d", name, len(g), len(w))
		}
		var maxAbs, sumAbs, sumRef float64
		for i := range g {
			d := math.Abs(float64(g[i] - w[i]))
			sumAbs += d
			sumRef += math.Abs(float64(w[i]))
			if d > maxAbs {
				maxAbs = d
			}
		}
		rel := sumAbs / sumRef
		t.Logf("%s: max abs %.4g, mean abs %.4g (%.3f%% of the reference's mean magnitude)", name, maxAbs, sumAbs/float64(len(g)), 100*rel)
		if rel > tol {
			t.Errorf("%s: relative error %.4f > %.4f", name, rel, tol)
		}
	}
	check("text encoder 1 penultimate", h1, ref["h1"], 2e-3)
	check("text encoder 2 penultimate", h2, ref["h2"], 2e-3)
	check("text encoder 2 pooled (projected)", pooled, ref["pooled"], 2e-3)

	// UNet step at the reference timestep, BF16, F32 output.
	S := ref["latent"].Dim(2) * 8
	addEmb, err := SDXLAddEmbedding(pooled, S, S, 0, 0, S, S, dev)
	if err != nil {
		t.Fatal(err)
	}
	ctx, err := ops.Concat([]*tensor.Tensor{h1, h2}, 2)
	if err != nil {
		t.Fatal(err)
	}
	ts := tensor.MustSlice[int64](ref["timestep"])[0]
	temb, err := TimestepEmbedding([]float32{float32(ts)}, 320, 10000, dev)
	if err != nil {
		t.Fatal(err)
	}
	latent, err := ref["latent"].To(dev)
	if err != nil {
		t.Fatal(err)
	}
	bf := func(x *tensor.Tensor) *tensor.Tensor {
		y, err := ops.Cast(x, dtype.BF16)
		if err != nil {
			t.Fatal(err)
		}
		return y
	}
	eps, err := sd.UNet.ForwardAug(bf(latent), bf(temb), bf(ctx), bf(addEmb))
	if err != nil {
		t.Fatal(err)
	}
	eps32, err := ops.Cast(eps, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	// BF16 weights and activations against an fp16 reference: a few
	// percent of the mean magnitude is the expected rounding floor.
	check("unet eps", eps32, ref["eps"], 0.05)
}
