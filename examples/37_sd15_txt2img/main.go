// Example 37 — Stable Diffusion 1.5 text-to-image.
//
// First run downloads ~4 GB into ~/.cache/huggingface/hub/. Defaults to
// `runwayml/stable-diffusion-v1-5`, 512×512, 50 DDIM steps, CFG 7.5.
//
//	go run            ./examples/37_sd15_txt2img -prompt "a photograph of an astronaut riding a horse"
//	go run            ./examples/37_sd15_txt2img -prompt "..." -steps 25 -cfg 6.0 -seed 42
package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"os"
	"runtime"
	"time"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/models"
	"light/tensor"
)

func main() {
	deviceFlag := flag.String("device", "gpu", "gpu or cpu")
	repo := flag.String("repo", "runwayml/stable-diffusion-v1-5", "HF repo")
	prompt := flag.String("prompt", "a photograph of an astronaut riding a horse", "prompt")
	negative := flag.String("negative", "", "negative prompt (CFG uncond branch)")
	steps := flag.Int("steps", 50, "DDIM sampler steps")
	cfg := flag.Float64("cfg", 7.5, "classifier-free guidance scale")
	width := flag.Int("width", 512, "output width (multiple of 8)")
	height := flag.Int("height", 512, "output height (multiple of 8)")
	seed := flag.Uint64("seed", 0, "RNG seed (0 = nondeterministic)")
	out := flag.String("out", "out.png", "output PNG path")
	bf16 := flag.Bool("bf16", false, "load UNet + VAE at BF16 (half the GPU memory; CLIP stays F32)")
	flag.Parse()

	dev := pickDevice(*deviceFlag)
	fmt.Printf("device:  %v\nrepo:    %s\n", dev, *repo)

	t0 := time.Now()
	weightDT := dtype.F32
	if *bf16 {
		weightDT = dtype.BF16
	}
	sd, err := models.LoadSD15FromHFWithDType(*repo, weightDT, dev)
	if err != nil {
		log.Fatalf("load: %v", err)
	}
	teDT := sd.TextEncoder.TokenEmb.Weight.DType()
	unetDT := sd.UNet.ConvIn.Weight.DType()
	vaeDT := sd.VAE.ConvIn.Weight.DType()
	fmt.Printf("loaded in %v — text=%.1fM (%v) unet=%.1fM (%v) vae=%.1fM (%v)\n",
		time.Since(t0).Round(time.Millisecond),
		float64(sd.TextEncoder.ParamCount())/1e6, teDT,
		float64(sd.UNet.ParamCount())/1e6, unetDT,
		float64(sd.VAE.ParamCount())/1e6, vaeDT,
	)
	if free, total, err := cudart.MemInfo(); err == nil {
		fmt.Printf("gpu:     %d MB used / %d MB total\n", (total-free)>>20, total>>20)
	}
	// Override the loader's default 50-step scheduler if requested.
	if *steps != 50 {
		sd.Scheduler = models.NewDDIMScheduler(*steps)
	}

	opts := models.DefaultSDGenerateOpts(*prompt)
	opts.Negative = *negative
	opts.Width = *width
	opts.Height = *height
	opts.NumInference = *steps
	opts.GuidanceScale = float32(*cfg)
	opts.Seed = *seed
	opts.OnPhase = func(name string) {
		fmt.Printf("→ %s\n", name)
	}
	stepStart := time.Now()
	opts.OnStep = func(step, total, t int, elapsed time.Duration) {
		eta := time.Duration(0)
		if step > 0 {
			avg := time.Since(stepStart) / time.Duration(step+1)
			eta = avg * time.Duration(total-step-1)
		}
		fmt.Printf("   step %2d/%d  t=%-3d  %v  (eta %v)\n",
			step+1, total, t, elapsed.Round(time.Millisecond), eta.Round(time.Second))
	}

	fmt.Printf("\nprompt:  %q\nsteps:   %d  cfg:     %.1f  size: %dx%d  seed: %d\n",
		opts.Prompt, opts.NumInference, opts.GuidanceScale, opts.Width, opts.Height, opts.Seed)

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
			r := clamp01(flat[off+0])
			g := clamp01(flat[off+1])
			b := clamp01(flat[off+2])
			img.Set(x, y, color.RGBA{
				R: uint8(r * 255),
				G: uint8(g * 255),
				B: uint8(b * 255),
				A: 255,
			})
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

func pickDevice(name string) device.Device {
	if name == "gpu" && device.CUDAAvailable() {
		runtime.LockOSThread()
		g, err := device.CUDA(0)
		if err == nil {
			return g
		}
		fmt.Printf("CUDA unavailable (%v); falling back to CPU\n", err)
	}
	return device.CPU()
}
