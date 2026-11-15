package main

import (
	"flag"
	"fmt"
	"image/color"
	"image/png"
	"os"
	"strings"
	"time"

	"light/diffusion"
	"light/engine"
	"light/splat"
)

// cmdIdol turns one image of an object into 3D Gaussians (speridlabs/idol).
func cmdIdol(args []string) error {
	fs := flag.NewFlagSet("idol", flag.ContinueOnError)
	dev := fs.String("device", "", "cpu, webgpu or gpu (default: CUDA when present, else cpu)")
	out := fs.String("o", "", "output PLY (default: <image name>.ply); the cropped input is written next to it as <name>_input.png")
	repo := fs.String("repo", "speridlabs/idol", "Idol weights: a Hub repo or a folder")
	dino := fs.String("dinov3", "", "DINOv3 encoder repo or folder (default: config.json's, facebook/dinov3-vith16plus-pretrain-lvd1689m, gated)")
	vae := fs.String("flux-vae", "black-forest-labs/FLUX.2-klein-4B", "a FLUX.2 repo or folder whose vae/ holds the autoencoder (all FLUX.2 models share it)")
	mode := fs.String("mode", "synthetic", "synthetic (renders, generated images) or real (photos): how much room the crop leaves")
	seed := fs.Uint64("seed", 42, "noise seed")
	steps := fs.Int("steps", 50, "sampling steps")
	cfg := fs.Float64("cfg", 3, "classifier-free guidance (0 disables)")
	shift := fs.Float64("shift", 3, "time shift: higher spends more steps at high noise")
	n := fs.Int("num-gaussians", 262144, "32768 to 1048576, a multiple of 32")
	f32 := fs.Bool("f32", false, "keep F32 weights (default: 16-bit weight storage, F32 compute)")
	views := fs.Int("views", 4, "also render this many turntable views from the predicted camera into <name>_views.png (0: none)")
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: light idol [flags] <image>\n\nOne image of an object to a 3D Gaussian splat (.ply). An image with a transparent background is used as is; any other (a photo, a render on a plain background) has its background removed by BiRefNet (the repo's birefnet.safetensors) first.\n\nflags:")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		fs.Usage()
		return fmt.Errorf("want one image")
	}
	in := fs.Arg(0)
	if *out == "" {
		*out = strings.TrimSuffix(in, ext(in)) + ".ply"
	}
	d, err := engine.PickDevice(*dev)
	if err != nil {
		return err
	}
	img, err := engine.LoadImage(in)
	if err != nil {
		return err
	}
	t0 := time.Now()
	p, err := diffusion.LoadIdolPipeline(*repo, d, diffusion.IdolLoadOptions{DINOv3: *dino, FluxVAE: *vae, Storage16: !*f32})
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "loaded idol in %.1f s\n", time.Since(t0).Seconds())
	g := *cfg
	if g == 0 {
		g = -1 // IdolOptions: negative disables guidance, 0 means the default
	}
	t1 := time.Now()
	res, err := p.Generate(diffusion.IdolOptions{Image: img, Mode: *mode, Seed: *seed, Steps: *steps, CFG: g, Shift: *shift,
		NumGaussians: *n, Step: func(i, n int) { fmt.Fprintf(os.Stderr, "\rstep %d/%d", i+1, n) }})
	fmt.Fprintln(os.Stderr)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "generated %d Gaussians in %.1f s\n", res.Gaussians.N(), time.Since(t1).Seconds())
	if err := res.Gaussians.WritePLY(*out); err != nil {
		return err
	}
	prev := strings.TrimSuffix(*out, ".ply") + "_input.png"
	f, err := os.Create(prev)
	if err != nil {
		return err
	}
	if err := png.Encode(f, res.Input); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	c := res.Camera
	if *views > 0 {
		g := res.Gaussians
		sg := &splat.Gaussians{XYZ: g.XYZ, FeaturesDC: g.FeaturesDC, Scaling: g.Scaling, Rotation: g.Rotation, Opacity: g.Opacity}
		im := splat.Turntable(sg, splat.Camera{Yaw: c.Yaw, Pitch: c.Pitch, Radius: c.Radius, FOV: c.FOV}, *views, 384, 384,
			color.NRGBA{255, 255, 255, 255})
		vp := strings.TrimSuffix(*out, ".ply") + "_views.png"
		vf, err := os.Create(vp)
		if err != nil {
			return err
		}
		if err := png.Encode(vf, im); err != nil {
			vf.Close()
			return err
		}
		if err := vf.Close(); err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "wrote %s\n", vp)
	}
	fmt.Fprintf(os.Stderr, "wrote %s and %s\ncamera: yaw %.3f rad, pitch %.3f rad, radius %.3f, fov %.1f deg\n", *out, prev, c.Yaw, c.Pitch, c.Radius, c.FOV)
	return nil
}

func ext(p string) string {
	if i := strings.LastIndexByte(p, '.'); i > strings.LastIndexAny(p, `/\`) {
		return p[i:]
	}
	return ""
}
