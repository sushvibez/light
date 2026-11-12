package diffusion

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
	"testing"

	"light/device"
	"light/hub"
	"light/tensor"
)

// QwenImage21Pipeline text-to-image on a tiny pipeline
// (scripts/verify/ref_qwenimage21_pipe.py: tiny Qwen3-VL encoder with
// Qwen-Image-2.1's tokenizer, tiny transformer, tiny 16x RGBA VAE, the real
// scheduler config) against diffusers: the templated ids, the pre-norm
// prompt embeddings without the system tokens, the schedule and the image.
func TestQwenImage21PipelineTinyMatchesDiffusers(t *testing.T) {
	raw, err := os.ReadFile("testdata/qwenimage21-tiny-pipe.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Prompt       string    `json:"prompt"`
		Height       int       `json:"height"`
		Width        int       `json:"width"`
		Steps        int       `json:"steps"`
		IDs          []int64   `json:"ids"`
		Embeds       []float32 `json:"embeds"`
		EmbedsShape  []int     `json:"embeds_shape"`
		Sigmas       []float32 `json:"sigmas"`
		Latents      []float32 `json:"latents"`
		LatentsShape []int     `json:"latents_shape"`
		Image        []float32 `json:"image"`
		ImageShape   []int     `json:"image_shape"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	tj, err := hub.Download("Qwen/Qwen-Image-2.1", "main", "processor/tokenizer.json")
	if err != nil {
		t.Skip(err)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	cmp := func(t *testing.T, name string, x *tensor.Tensor, want []float32, tol float64) {
		t.Helper()
		x, err := x.To(device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		got := tensor.MustSlice[float32](x)
		if len(got) != len(want) {
			t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
		}
		md := 0.0
		for i := range got {
			md = math.Max(md, math.Abs(float64(got[i]-want[i])))
		}
		t.Logf("%s: max diff %.3g over %d values", name, md, len(got))
		if md > tol {
			t.Errorf("%s differs by %.3g", name, md)
		}
	}
	type run struct {
		dev, enc device.Device
		seq      bool
	}
	var runs []run
	for _, d := range devs {
		runs = append(runs, run{d, d, false})
	}
	if len(devs) > 1 {
		runs = append(runs, run{devs[1], device.CPU(), true})
	}
	for _, r := range runs {
		dev := r.dev
		t.Run(fmt.Sprintf("%v/encoder-%v/seq=%v", dev, r.enc, r.seq), func(t *testing.T) {
			p, err := LoadQwenImage21("testdata/qwenimage21-tiny-pipe", dev, QwenImage21LoadOptions{TokenizerJSON: tj, EncoderDevice: r.enc, Sequential: r.seq})
			if err != nil {
				t.Fatal(err)
			}
			lat, err := tensor.New(ref.Latents, ref.LatentsShape, dev)
			if err != nil {
				t.Fatal(err)
			}
			res, err := p.Generate(QwenImage21Options{Prompt: ref.Prompt, Width: ref.Width, Height: ref.Height, Steps: ref.Steps, Latents: lat})
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(res.IDs, ref.IDs) {
				t.Errorf("ids\n got %v\nwant %v", res.IDs, ref.IDs)
			}
			for i, s := range ref.Sigmas {
				if math.Abs(float64(res.Sigmas[i]-s)) > 2e-6 {
					t.Errorf("sigmas %v, want %v", res.Sigmas, ref.Sigmas)
					break
				}
			}
			if !slices.Equal(res.PromptEmbeds.Shape().Dims(), ref.EmbedsShape) {
				t.Fatalf("embeds %v, want %v", res.PromptEmbeds.Shape(), ref.EmbedsShape)
			}
			cmp(t, "prompt embeds", res.PromptEmbeds, ref.Embeds, 1e-5)
			if !slices.Equal(res.Image.Shape().Dims(), ref.ImageShape) {
				t.Fatalf("image %v, want %v", res.Image.Shape(), ref.ImageShape)
			}
			cmp(t, "image", res.Image, ref.Image, 1e-4)
		})
	}
}
