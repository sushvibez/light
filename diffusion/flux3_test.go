package diffusion

import (
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"math"
	"os"
	"slices"
	"testing"

	"light/device"
	"light/hub"
	"light/tensor"
)

// The distilled FLUX 3 text-to-image recipe on a tiny pipeline
// (scripts/verify/ref_flux3_pipe.py: tiny Qwen3-VL text encoder, tiny
// flow-map Flux3 transformer, tiny AutoencoderKLFlux2) against diffusers'
// own building blocks: the prompt ids, the padded stacked text embeddings,
// the schedule and the final image, on the CPU and WebGPU.
func TestFlux3PipelineTinyMatchesDiffusers(t *testing.T) {
	raw, err := os.ReadFile("testdata/flux3-tiny-pipe.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Prompt       string    `json:"prompt"`
		Height       int       `json:"height"`
		Width        int       `json:"width"`
		Steps        int       `json:"steps"`
		Layers       []int     `json:"layers"`
		IDs          []int64   `json:"ids"`
		Embeds       []float32 `json:"embeds"`
		EmbedsShape  []int     `json:"embeds_shape"`
		Sigmas       []float32 `json:"sigmas"`
		Latents      []float32 `json:"latents"`
		LatentsShape []int     `json:"latents_shape"`
		Image        []float32 `json:"image"`
		Edit1Lat     []float32 `json:"edit1_latents"`
		Edit1LatSh   []int     `json:"edit1_latents_shape"`
		Edit1        []float32 `json:"edit1"`
		Edit1Sh      []int     `json:"edit1_shape"`
		Edit2Lat     []float32 `json:"edit2_latents"`
		Edit2LatSh   []int     `json:"edit2_latents_shape"`
		Edit2        []float32 `json:"edit2"`
		Edit2Sh      []int     `json:"edit2_shape"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	tj, err := hub.Download("Qwen/Qwen3-VL-4B-Instruct", "main", "tokenizer.json")
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
			p, err := LoadFlux3("testdata/flux3-tiny-pipe", dev, Flux3LoadOptions{TokenizerJSON: tj, EncoderDevice: r.enc, Sequential: r.seq})
			if err != nil {
				t.Fatal(err)
			}
			lat, err := tensor.New(ref.Latents, ref.LatentsShape, dev)
			if err != nil {
				t.Fatal(err)
			}
			res, err := p.Generate(Flux3Options{Prompt: ref.Prompt, Width: ref.Width, Height: ref.Height,
				Steps: ref.Steps, Layers: ref.Layers, Latents: lat})
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
			if !slices.Equal(res.PromptEmbeds.Shape().Dims(), ref.EmbedsShape) {
				t.Fatalf("embeds %v, want %v", res.PromptEmbeds.Shape(), ref.EmbedsShape)
			}
			cmp(t, "prompt embeds", res.PromptEmbeds, ref.Embeds, 1e-5)
			cmp(t, "image", res.Image, ref.Image, 1e-4)

			refs := make([]image.Image, 3)
			for i := range refs {
				f, err := os.Open(fmt.Sprintf("testdata/flux3-tiny-pipe/ref%d.png", i))
				if err != nil {
					t.Fatal(err)
				}
				refs[i], err = png.Decode(f)
				f.Close()
				if err != nil {
					t.Fatal(err)
				}
			}
			for _, c := range []struct {
				name        string
				imgs        []image.Image
				lat, want   []float32
				latSh, shap []int
			}{{"edit, one reference (fill)", refs[:1], ref.Edit1Lat, ref.Edit1, ref.Edit1LatSh, ref.Edit1Sh},
				{"edit, two references (crop)", refs[1:], ref.Edit2Lat, ref.Edit2, ref.Edit2LatSh, ref.Edit2Sh}} {
				lat, err := tensor.New(c.lat, c.latSh, dev)
				if err != nil {
					t.Fatal(err)
				}
				res, err := p.Generate(Flux3Options{Prompt: ref.Prompt, Steps: ref.Steps, Layers: ref.Layers, Latents: lat, Images: c.imgs})
				if err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(res.Image.Shape().Dims(), c.shap) {
					t.Fatalf("%s: image %v, want %v", c.name, res.Image.Shape(), c.shap)
				}
				cmp(t, c.name, res.Image, c.want, 1e-4)
			}
		})
	}
}
