package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"time"

	"light/diffusion"
	"light/engine"
	"light/hub"
)

// cmdImage generates an image with a diffusers pipeline (FLUX.2 klein,
// FLUX.1) and writes it as a PNG.
func cmdImage(args []string) error {
	fs := flag.NewFlagSet("image", flag.ContinueOnError)
	dev := fs.String("device", "", "cpu, webgpu or gpu (default: CUDA when present, else cpu)")
	out := fs.String("o", "image.png", "output PNG")
	width := fs.Int("width", 0, "image width in pixels (default 1024, or the first reference image's)")
	height := fs.Int("height", 0, "image height in pixels (default 1024, or the first reference image's)")
	steps := fs.Int("steps", 0, "denoising steps (default: the pipeline's)")
	guidance := fs.Float64("guidance", 0, "classifier-free / distilled guidance (default: the pipeline's)")
	seed := fs.Uint64("seed", 0, "noise seed")
	f32 := fs.Bool("f32", false, "keep F32 weights (default: 16-bit weight storage, F32 compute)")
	var refs stringList
	fs.Var(&refs, "image", "reference image for editing (FLUX.2 klein; repeatable; the output size defaults to the first's)")
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: light image [flags] <org/repo> <prompt>\n\nPipelines: Flux2KleinPipeline (FLUX.2 klein, text-to-image and editing), ZImagePipeline (Z-Image / Turbo), FluxPipeline (FLUX.1).\n\nflags:")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 {
		fs.Usage()
		return errors.New("need a pipeline repo and a prompt")
	}
	repo, prompt := fs.Arg(0), fs.Arg(1)
	d, err := engine.PickDevice(*dev)
	if err != nil {
		return err
	}
	idx, err := hub.Download(repo, "main", "model_index.json")
	if err != nil {
		return err
	}
	raw, err := os.ReadFile(idx)
	if err != nil {
		return err
	}
	var mi struct {
		Class string `json:"_class_name"`
	}
	if err := json.Unmarshal(raw, &mi); err != nil {
		return err
	}
	t0 := time.Now()
	var img diffusion.Image
	switch mi.Class {
	case "Flux2KleinPipeline":
		p, err := diffusion.LoadFlux2(repo, d, diffusion.Flux2LoadOptions{Storage16: !*f32})
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "loaded %s in %.1f s\n", repo, time.Since(t0).Seconds())
		t1 := time.Now()
		var imgs []image.Image
		for _, r := range refs {
			im, err := engine.LoadImage(r)
			if err != nil {
				return err
			}
			imgs = append(imgs, im)
		}
		res, err := p.Generate(diffusion.Flux2Options{Prompt: prompt, Width: *width, Height: *height,
			Steps: *steps, Guidance: float32(*guidance), Seed: *seed, Images: imgs})
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "generated in %.1f s\n", time.Since(t1).Seconds())
		img = diffusion.Image{Pixels: res.Image}
	case "ZImagePipeline":
		if len(refs) > 0 {
			return fmt.Errorf("%s: reference images need FLUX.2 klein", repo)
		}
		p, err := diffusion.LoadZImage(repo, d, diffusion.ZImageLoadOptions{Storage16: !*f32})
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "loaded %s in %.1f s\n", repo, time.Since(t0).Seconds())
		t1 := time.Now()
		res, err := p.Generate(diffusion.ZImageOptions{Prompt: prompt, Width: *width, Height: *height,
			Steps: *steps, Guidance: float32(*guidance), Seed: *seed})
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "generated in %.1f s\n", time.Since(t1).Seconds())
		img = diffusion.Image{Pixels: res.Image}
	case "FluxPipeline":
		if len(refs) > 0 {
			return fmt.Errorf("%s: reference images need FLUX.2 klein", repo)
		}
		p, err := diffusion.LoadFlux(repo, d)
		if err != nil {
			return err
		}
		res, err := p.Generate(diffusion.FluxOptions{Prompt: prompt, Width: *width, Height: *height,
			Steps: *steps, Guidance: float32(*guidance), Seed: *seed})
		if err != nil {
			return err
		}
		img = diffusion.Image{Pixels: res.Image}
	default:
		return fmt.Errorf("%s: pipeline %q not supported (Flux2KleinPipeline, ZImagePipeline, FluxPipeline)", repo, mi.Class)
	}
	if err := img.SavePNG(*out); err != nil {
		return err
	}
	abs, _ := filepath.Abs(*out)
	fmt.Fprintln(os.Stderr, "wrote", abs)
	return nil
}
