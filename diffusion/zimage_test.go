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

// A tiny ZImagePipeline (Z-Image-Turbo's scheduler, a Qwen3 tokenizer, tiny
// random Qwen3Model / Z-Image transformer / VAE; scripts/verify/ref_zimage_pipe.py)
// against diffusers with guidance 2 for 3 steps at 64 x 64: prompt ids,
// caption features (vs float64), sigmas and the final image, on the CPU and
// WebGPU.
func TestZImagePipelineTinyMatchesDiffusers(t *testing.T) {
	raw, err := os.ReadFile("testdata/zimage-tiny-pipe.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Prompt       string    `json:"prompt"`
		Height       int       `json:"height"`
		Width        int       `json:"width"`
		Steps        int       `json:"steps"`
		MaxSeqLen    int       `json:"max_seq_len"`
		Guidance     float32   `json:"guidance"`
		IDs          []int64   `json:"ids"`
		Latents      []float32 `json:"latents"`
		LatentsShape []int     `json:"latents_shape"`
		Caption      []float32 `json:"caption_f64"`
		Sigmas       []float32 `json:"sigmas"`
		Image        []float32 `json:"image"`
		I2ISteps     int       `json:"i2i_steps"`
		I2IStrength  float32   `json:"i2i_strength"`
		I2INoise     []float32 `json:"i2i_noise"`
		I2INoiseSh   []int     `json:"i2i_noise_shape"`
		I2IImage     []float32 `json:"i2i_image"`
		I2IImageSh   []int     `json:"i2i_image_shape"`
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
	type run struct {
		dev, enc device.Device
		seq      bool
	}
	var runs []run
	for _, d := range devs {
		runs = append(runs, run{d, d, false})
	}
	runs = append(runs, run{devs[0], devs[0], true})
	if len(devs) > 1 {
		runs = append(runs, run{devs[1], device.CPU(), true}) // GPU transformer, CPU text encoder, loaded in turn
	}
	for _, r := range runs {
		dev := r.dev
		t.Run(fmt.Sprintf("%v/encoder-%v/seq=%v", dev, r.enc, r.seq), func(t *testing.T) {
			p, err := LoadZImage("testdata/zimage-tiny-pipe", dev, ZImageLoadOptions{TokenizerJSON: tj, EncoderDevice: r.enc, Sequential: r.seq})
			if err != nil {
				t.Fatal(err)
			}
			lat, err := tensor.New(ref.Latents, ref.LatentsShape, dev)
			if err != nil {
				t.Fatal(err)
			}
			res, err := p.Generate(ZImageOptions{Prompt: ref.Prompt, Width: ref.Width, Height: ref.Height,
				Steps: ref.Steps, Guidance: ref.Guidance, MaxSeqLen: ref.MaxSeqLen, Latents: lat})
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
			cmp(t, "caption", res.Caption, ref.Caption, 1e-5)
			cmp(t, "image", res.Image, ref.Image, 1e-4)

			noise, err := tensor.New(ref.I2INoise, ref.I2INoiseSh, dev)
			if err != nil {
				t.Fatal(err)
			}
			res, err = p.Generate(ZImageOptions{Prompt: ref.Prompt, Steps: ref.I2ISteps, Guidance: ref.Guidance,
				MaxSeqLen: ref.MaxSeqLen, Noise: noise, Img2Img: Img2Img{Image: refPNG(t), Strength: ref.I2IStrength}})
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(res.Image.Shape().Dims(), ref.I2IImageSh) {
				t.Errorf("img2img image %v, want %v", res.Image.Shape(), ref.I2IImageSh)
			}
			cmp(t, "img2img image", res.Image, ref.I2IImage, 1e-4)
		})
	}
}
