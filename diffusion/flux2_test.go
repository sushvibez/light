package diffusion

import (
	"encoding/json"
	"math"
	"os"
	"slices"
	"testing"

	"light/device"
	"light/hub"
	"light/tensor"
)

// A tiny Flux2KleinPipeline (FLUX.2-klein-4B's tokenizer and scheduler, tiny
// random Qwen3 / FLUX.2 transformer / FLUX.2 VAE;
// scripts/verify/ref_flux2_pipe.py) against diffusers, non-distilled
// (classifier-free guidance 3) for 2 steps at 64 x 64 with 40 padded text
// tokens: prompt ids, text conditioning, sigmas and the final image, on the
// CPU and WebGPU.
func TestFlux2PipelineTinyMatchesDiffusers(t *testing.T) {
	raw, err := os.ReadFile("testdata/flux2-tiny-pipe.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Prompt       string    `json:"prompt"`
		Height       int       `json:"height"`
		Width        int       `json:"width"`
		Steps        int       `json:"steps"`
		MaxSeqLen    int       `json:"max_seq_len"`
		Layers       []int     `json:"layers"`
		Guidance     float32   `json:"guidance"`
		IDs          []int64   `json:"ids"`
		Latents      []float32 `json:"latents"`
		LatentsShape []int     `json:"latents_shape"`
		PromptEmbeds []float32 `json:"prompt_embeds_f64"` // see ref_flux2_pipe.py
		Sigmas       []float32 `json:"sigmas"`
		Image        []float32 `json:"image"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	tj, err := hub.Download("black-forest-labs/FLUX.2-klein-4B", "main", "tokenizer/tokenizer.json")
	if err != nil {
		t.Skip(err)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	// tol is relative to max(1, max |ref|).
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
		md, ma := 0.0, 0.0
		for i := range got {
			md = math.Max(md, math.Abs(float64(got[i]-want[i])))
			ma = math.Max(ma, math.Abs(float64(want[i])))
		}
		t.Logf("%s: max diff %.3g (max |ref| %.3g) over %d values", name, md, ma, len(got))
		if md > tol*math.Max(1, ma) {
			t.Errorf("%s differs by %.3g", name, md)
		}
	}
	for _, dev := range devs {
		t.Run(dev.String(), func(t *testing.T) {
			p, err := LoadFlux2("testdata/flux2-tiny-pipe", dev, Flux2LoadOptions{TokenizerJSON: tj})
			if err != nil {
				t.Fatal(err)
			}
			lat, err := tensor.New(ref.Latents, ref.LatentsShape, dev)
			if err != nil {
				t.Fatal(err)
			}
			res, err := p.Generate(Flux2Options{Prompt: ref.Prompt, Width: ref.Width, Height: ref.Height, Steps: ref.Steps,
				Guidance: ref.Guidance, MaxSeqLen: ref.MaxSeqLen, Layers: ref.Layers, Latents: lat})
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(res.IDs, ref.IDs) {
				t.Errorf("ids\n got %v\nwant %v", res.IDs, ref.IDs)
			}
			for i, s := range ref.Sigmas {
				if math.Abs(float64(res.Sigmas[i]-s)) > 1e-6 {
					t.Errorf("sigmas %v, want %v", res.Sigmas, ref.Sigmas)
					break
				}
			}
			cmp(t, "text conditioning", res.PromptEmbeds, ref.PromptEmbeds, 1e-5)
			cmp(t, "image", res.Image, ref.Image, 1e-4)
		})
	}
}
