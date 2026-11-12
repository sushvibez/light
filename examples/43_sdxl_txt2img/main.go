// Example 43 — SDXL base text-to-image.
//
// First run downloads ~7 GB (fp16 variant) into ~/.cache/huggingface/hub/.
// Defaults: stabilityai/stable-diffusion-xl-base-1.0, 1024×1024, 30 DDIM
// steps, CFG 5, UNet at BF16 (5.2 GB), text encoders and VAE at F32 — about
// 9 GB on the card in flight.
//
//	go run ./examples/43_sdxl_txt2img -prompt "a photograph of an astronaut riding a horse on the moon" -seed 7
package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"os"
	"time"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/models"
	"light/tensor"
)

func main() {
	repo := flag.String("repo", models.SDXLRepo, "HF repo (SDXL base layout)")
	prompt := flag.String("prompt", "a photograph of an astronaut riding a horse on the moon", "prompt")
	negative := flag.String("negative", "", "negative prompt (CFG uncond branch)")
	steps := flag.Int("steps", 30, "DDIM sampler steps")
	cfg := flag.Float64("cfg", 5, "classifier-free guidance scale")
	width := flag.Int("width", 1024, "output width (multiple of 8)")
	height := flag.Int("height", 1024, "output height (multiple of 8)")
	seed := flag.Uint64("seed", 0, "RNG seed (0 = nondeterministic)")
	out := flag.String("out", "sdxl.png", "output PNG path")
	f32 := flag.Bool("f32", false, "load the UNet at F32 (10.3 GB) instead of BF16")
	flag.Parse()

	if !device.CUDAAvailable() {
		log.Fatal("SDXL needs CUDA (the UNet alone is 5.2 GB at BF16)")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		log.Fatal(err)
	}
	wdt := dtype.BF16
	if *f32 {
		wdt = dtype.F32
	}
	t0 := time.Now()
	sd, err := models.LoadSDXLFromHF(*repo, wdt, dev)
	if err != nil {
		log.Fatalf("load: %v", err)
	}
	fmt.Printf("loaded in %v — clip-L %.0fM, bigG %.0fM, unet %.0fM (%v), vae %.0fM\n",
		time.Since(t0).Round(time.Millisecond),
		float64(sd.TextEncoder1.ParamCount())/1e6, float64(sd.TextEncoder2.ParamCount())/1e6,
		float64(sd.UNet.ParamCount())/1e6, wdt, float64(sd.VAE.ParamCount())/1e6)
	if free, total, err := cudart.MemInfo(); err == nil {
		fmt.Printf("gpu:     %d MB used / %d MB total\n", (total-free)>>20, total>>20)
	}

	opts := models.DefaultSDXLGenerateOpts(*prompt)
	opts.Negative = *negative
	opts.Width, opts.Height = *width, *height
	opts.NumInference = *steps
	opts.GuidanceScale = float32(*cfg)
	opts.Seed = *seed
	opts.OnPhase = func(name string) { fmt.Printf("→ %s\n", name) }
	stepStart := time.Now()
	opts.OnStep = func(step, total, t int, elapsed time.Duration) {
		eta := time.Duration(0)
		if step > 0 {
			avg := time.Since(stepStart) / time.Duration(step+1)
			eta = avg * time.Duration(total-step-1)
		}
		fmt.Printf("   step %2d/%d  t=%-3d  %v  (eta %v)\n", step+1, total, t, elapsed.Round(time.Millisecond), eta.Round(time.Second))
	}
	fmt.Printf("\nprompt:  %q\nsteps:   %d  cfg: %.1f  size: %dx%d  seed: %d\n", opts.Prompt, opts.NumInference, opts.GuidanceScale, opts.Width, opts.Height, opts.Seed)
	t0 = time.Now()
	img, err := sd.Generate(opts, dev)
	if err != nil {
		log.Fatalf("generate: %v", err)
	}
	fmt.Printf("generated in %v\n", time.Since(t0).Round(time.Millisecond))
	if err := saveAsPNG(img, *out); err != nil {
		log.Fatalf("save PNG: %v", err)
	}
	fmt.Printf("wrote %s\n", *out)
}

// saveAsPNG writes an (H, W, 3) F32 image in [0, 1] as a PNG file.
func saveAsPNG(t *tensor.Tensor, path string) error {
	if t.Rank() != 3 || t.Dim(2) != 3 {
		return fmt.Errorf("saveAsPNG: expected (H, W, 3), got %v", t.Shape())
	}
	H, W := t.Dim(0), t.Dim(1)
	flat := tensor.MustSlice[float32](t)
	img := image.NewRGBA(image.Rect(0, 0, W, H))
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			off := (y*W + x) * 3
			img.Set(x, y, color.RGBA{R: uint8(clamp01(flat[off]) * 255), G: uint8(clamp01(flat[off+1]) * 255), B: uint8(clamp01(flat[off+2]) * 255), A: 255})
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

func clamp01(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
