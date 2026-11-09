package models

import (
	"encoding/json"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/tensor"
)

// Stable Diffusion 1.5 end to end against diffusers
// (scripts/verify/ref_sd15_pipe.py): text-to-image from explicit latents and
// image-to-image (StableDiffusionImg2ImgPipeline, strength 0.75) at
// 128 x 128 for 4 DDIM steps with CFG 7.5, at F32 on the CPU.
func TestSD15MatchesDiffusers(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (4.3 GB F32 checkpoint)")
	}
	const repo = "stable-diffusion-v1-5/stable-diffusion-v1-5"
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/*/unet/diffusion_pytorch_model.safetensors")); len(m) == 0 {
		t.Skip(repo + " not in HF cache")
	}
	requireHostMemGB(t, 8)
	releaseModelMemory(t)
	raw, err := os.ReadFile("testdata/sd15/ref.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Prompt      string    `json:"prompt"`
		Height      int       `json:"height"`
		Width       int       `json:"width"`
		Steps       int       `json:"steps"`
		Guidance    float32   `json:"guidance"`
		Latents     []float32 `json:"latents"`
		Image       []float32 `json:"image"`
		I2IStrength float32   `json:"i2i_strength"`
		I2INoise    []float32 `json:"i2i_noise"`
		I2IImage    []float32 `json:"i2i_image"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open("testdata/sd15/init.png")
	if err != nil {
		t.Fatal(err)
	}
	init, err := png.Decode(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	dev := device.CPU()
	sd, err := LoadSD15FromHF(repo, dev)
	if err != nil {
		t.Fatal(err)
	}
	// cmp compares an (H, W, 3) image with diffusers' (1, 3, H, W).
	cmp := func(name string, img *tensor.Tensor, want []float32) {
		got := tensor.MustSlice[float32](img)
		H, W := img.Dim(0), img.Dim(1)
		if len(got) != len(want) {
			t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
		}
		md := 0.0
		for c := range 3 {
			for i := range H * W {
				md = math.Max(md, math.Abs(float64(got[i*3+c]-want[c*H*W+i])))
			}
		}
		t.Logf("%s: max diff %.3g over %d values", name, md, len(got))
		if md > 2e-3 {
			t.Errorf("%s differs by %.3g", name, md)
		}
	}
	shape := []int{1, 4, ref.Height / 8, ref.Width / 8}
	lat, err := tensor.New(ref.Latents, shape, dev)
	if err != nil {
		t.Fatal(err)
	}
	o := DefaultSDGenerateOpts(ref.Prompt)
	o.Width, o.Height, o.NumInference, o.GuidanceScale, o.Noise = ref.Width, ref.Height, ref.Steps, ref.Guidance, lat
	img, err := sd.Generate(o, dev)
	if err != nil {
		t.Fatal(err)
	}
	cmp("txt2img", img, ref.Image)

	noise, err := tensor.New(ref.I2INoise, shape, dev)
	if err != nil {
		t.Fatal(err)
	}
	o.Width, o.Height, o.Noise, o.InitImage, o.Strength = 0, 0, noise, init, ref.I2IStrength
	if img, err = sd.Generate(o, dev); err != nil {
		t.Fatal(err)
	}
	cmp("img2img", img, ref.I2IImage)
}
