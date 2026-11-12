//go:build !nocuda && cgo

package diffusion

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"light/device"
	"light/tensor"
)

// Real-weight checks against diffusers on the GPU (BF16 both sides). The
// reference comes from scripts/verify/ref_real_pipe.py; point the env var
// at its JSON:
//
//	LIGHT_HEAVY_TESTS=1 LIGHT_ZIMAGE_REAL_REF=/path/zimage.json go test ./diffusion -run TestZImageTurboRealCUDA -v
//	LIGHT_HEAVY_TESTS=1 LIGHT_KLEIN_REAL_REF=/path/klein.json go test ./diffusion -run TestFlux2KleinRealCUDA -v
//
// Four BF16 denoising steps on both sides accumulate rounding, so the
// images are compared by mean absolute pixel difference (tolerance 0.03
// of the [0, 1] range) and the max is only reported.
type realRef struct {
	Repo         string    `json:"repo"`
	Prompt       string    `json:"prompt"`
	Height       int       `json:"height"`
	Width        int       `json:"width"`
	Steps        int       `json:"steps"`
	Guidance     float32   `json:"guidance"`
	Latents      []float32 `json:"latents"`
	LatentsShape []int     `json:"latents_shape"`
	Image        []float32 `json:"image"`
	ImageShape   []int     `json:"image_shape"`
}

func loadRealRef(t *testing.T, env string) (realRef, device.Device) {
	t.Helper()
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1")
	}
	p := os.Getenv(env)
	if p == "" {
		t.Skipf("set %s to the reference JSON (scripts/verify/ref_real_pipe.py)", env)
	}
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	var r realRef
	if err := json.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}
	return r, dev
}

func compareReal(t *testing.T, got *tensor.Tensor, ref realRef) {
	t.Helper()
	c, err := got.To(device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	g := tensor.MustSlice[float32](c)
	if len(g) != len(ref.Image) {
		t.Fatalf("image %v, want %v", got.Shape(), ref.ImageShape)
	}
	var sum, mx float64
	for i := range g {
		d := math.Abs(float64(g[i] - ref.Image[i]))
		sum += d
		mx = math.Max(mx, d)
	}
	mean := sum / float64(len(g))
	t.Logf("%s: %dx%d, %d steps: mean |Δ| = %.4f, max %.3f (of 1.0)", ref.Repo, ref.Width, ref.Height, ref.Steps, mean, mx)
	if dir := os.Getenv("LIGHT_REAL_SAVE_DIR"); dir != "" { // both images, for a look
		name := strings.ReplaceAll(ref.Repo, "/", "--")
		if err := (Image{Pixels: got}).SavePNG(filepath.Join(dir, name+"-light.png")); err != nil {
			t.Error(err)
		}
		r, _ := tensor.New(ref.Image, append([]int{1}, ref.ImageShape...), device.CPU())
		if err := (Image{Pixels: r}).SavePNG(filepath.Join(dir, name+"-diffusers.png")); err != nil {
			t.Error(err)
		}
	}
	if mean > 0.03 {
		t.Errorf("mean pixel difference %.4f > 0.03", mean)
	}
}

func TestZImageTurboRealCUDA(t *testing.T) {
	ref, dev := loadRealRef(t, "LIGHT_ZIMAGE_REAL_REF")
	p, err := LoadZImage(ref.Repo, dev, ZImageLoadOptions{Storage16: true, Sequential: true})
	if err != nil {
		t.Fatal(err)
	}
	lat, err := tensor.New(ref.Latents, ref.LatentsShape[1:], dev) // (C, h, w): the batch axis dropped
	if err != nil {
		t.Fatal(err)
	}
	res, err := p.Generate(ZImageOptions{Prompt: ref.Prompt, Width: ref.Width, Height: ref.Height, Steps: ref.Steps, Guidance: ref.Guidance, Latents: lat})
	if err != nil {
		t.Fatal(err)
	}
	compareReal(t, res.Image, ref)
}

func TestFlux2KleinRealCUDA(t *testing.T) {
	ref, dev := loadRealRef(t, "LIGHT_KLEIN_REAL_REF")
	p, err := LoadFlux2(ref.Repo, dev, Flux2LoadOptions{Storage16: true, Sequential: true})
	if err != nil {
		t.Fatal(err)
	}
	lat, err := tensor.New(ref.Latents, ref.LatentsShape, dev)
	if err != nil {
		t.Fatal(err)
	}
	res, err := p.Generate(Flux2Options{Prompt: ref.Prompt, Width: ref.Width, Height: ref.Height, Steps: ref.Steps, Guidance: ref.Guidance, Latents: lat})
	if err != nil {
		t.Fatal(err)
	}
	compareReal(t, res.Image, ref)
}
