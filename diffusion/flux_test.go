package diffusion

import (
	"encoding/json"
	"image"
	"image/png"
	"math"
	"os"
	"slices"
	"testing"

	"light/device"
	"light/tensor"
)

// The whole FluxPipeline on hf-internal-testing/tiny-flux-pipe (VAE swapped
// for a 32-group one; scripts/verify/ref_flux_pipe.py) against diffusers:
// CLIP / T5 token ids, T5 states, CLIP pooled, the sigma schedule and the
// final image after 2 Euler steps, on the CPU and WebGPU.
func TestFluxPipelineTinyMatchesDiffusers(t *testing.T) {
	raw, err := os.ReadFile("testdata/flux-tiny-pipe.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Prompt       string    `json:"prompt"`
		Height       int       `json:"height"`
		Width        int       `json:"width"`
		Steps        int       `json:"steps"`
		MaxSeqLen    int       `json:"max_seq_len"`
		CLIPIDs      []int32   `json:"clip_ids"`
		T5IDs        []int64   `json:"t5_ids"`
		Latents      []float32 `json:"latents"`
		LatentsShape []int     `json:"latents_shape"`
		PromptEmbeds []float32 `json:"prompt_embeds"`
		Pooled       []float32 `json:"pooled"`
		Sigmas       []float32 `json:"sigmas"`
		Image        []float32 `json:"image"`
		I2IHeight    int       `json:"i2i_height"`
		I2IWidth     int       `json:"i2i_width"`
		I2ISteps     int       `json:"i2i_steps"`
		I2IStrength  float32   `json:"i2i_strength"`
		I2INoise     []float32 `json:"i2i_noise"`
		I2INoiseSh   []int     `json:"i2i_noise_shape"`
		I2IImage     []float32 `json:"i2i_image"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	maxDiff := func(t *testing.T, name string, x *tensor.Tensor, want []float32, tol float64) {
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
	for _, dev := range devs {
		t.Run(dev.String(), func(t *testing.T) {
			p, err := LoadFlux("testdata/flux-tiny-pipe", dev)
			if err != nil {
				t.Fatal(err)
			}
			if got := p.CLIPTok.Encode(ref.Prompt, p.CLIP.Cfg.MaxPosition); !slices.Equal(got, ref.CLIPIDs) {
				t.Errorf("CLIP ids\n got %v\nwant %v", got, ref.CLIPIDs)
			}
			if got, _ := p.t5IDs(ref.Prompt, ref.MaxSeqLen); !slices.Equal(got, ref.T5IDs) {
				t.Errorf("T5 ids\n got %v\nwant %v", got, ref.T5IDs)
			}
			lat, err := tensor.New(ref.Latents, ref.LatentsShape, dev)
			if err != nil {
				t.Fatal(err)
			}
			res, err := p.Generate(FluxOptions{Prompt: ref.Prompt, Width: ref.Width, Height: ref.Height,
				Steps: ref.Steps, MaxSeqLen: ref.MaxSeqLen, Latents: lat})
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(res.Sigmas, ref.Sigmas) {
				t.Errorf("sigmas %v, want %v", res.Sigmas, ref.Sigmas)
			}
			maxDiff(t, "T5 states", res.PromptEmbeds, ref.PromptEmbeds, 1e-4)
			maxDiff(t, "CLIP pooled", res.Pooled, ref.Pooled, 1e-4)
			maxDiff(t, "image", res.Image, ref.Image, 1e-4)

			noise, err := tensor.New(ref.I2INoise, ref.I2INoiseSh, dev)
			if err != nil {
				t.Fatal(err)
			}
			res, err = p.Generate(FluxOptions{Prompt: ref.Prompt, Width: ref.I2IWidth, Height: ref.I2IHeight,
				Steps: ref.I2ISteps, MaxSeqLen: ref.MaxSeqLen, Noise: noise,
				Img2Img: Img2Img{Image: refPNG(t), Strength: ref.I2IStrength}})
			if err != nil {
				t.Fatal(err)
			}
			maxDiff(t, "img2img image", res.Image, ref.I2IImage, 1e-4)
		})
	}
}

// refPNG is flux2-ref.png, the init / reference image of the pipeline tests.
func refPNG(t *testing.T) image.Image {
	t.Helper()
	f, err := os.Open("testdata/flux2-ref.png")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	return img
}
