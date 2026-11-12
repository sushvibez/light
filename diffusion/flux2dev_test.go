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

// Flux2Pipeline (FLUX.2 [dev]) on a tiny pipeline (scripts/verify/
// ref_flux2dev_pipe.py: tiny Mistral3 encoder with Mistral-Small-3.1's
// tokenizer, tiny guidance-embedding transformer, tiny FLUX.2 VAE) against
// diffusers: the system-message prompt ids with right padding, the stacked
// hidden states, the schedule and the final image with guidance 4.
func TestFlux2DevPipelineTinyMatchesDiffusers(t *testing.T) {
	raw, err := os.ReadFile("testdata/flux2dev-tiny-pipe.json")
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
		Embeds       []float32 `json:"prompt_embeds"`
		Sigmas       []float32 `json:"sigmas"`
		Image        []float32 `json:"image"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	tj, err := hub.Download("mistralai/Mistral-Small-3.1-24B-Instruct-2503", "main", "tokenizer.json")
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
			p, err := LoadFlux2("testdata/flux2dev-tiny-pipe", dev, Flux2LoadOptions{TokenizerJSON: tj, EncoderDevice: r.enc, Sequential: r.seq})
			if err != nil {
				t.Fatal(err)
			}
			if !p.Mistral {
				t.Fatal("not detected as Flux2Pipeline")
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
			cmp(t, "prompt embeds", res.PromptEmbeds, ref.Embeds, 1e-5)
			cmp(t, "image", res.Image, ref.Image, 1e-4)
		})
	}
}
