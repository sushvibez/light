package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"strings"
	"time"

	"light/diffusion"
	"light/engine"
	"light/mesh"
	"light/splat"
)

// cmdTrellis turns one image of an object into 3D Gaussians and, with
// -mesh, a triangle mesh (microsoft/TRELLIS-image-large: sparse structure,
// structured latent, Gaussian and mesh decoders).
func cmdTrellis(args []string) error {
	fs := flag.NewFlagSet("trellis", flag.ContinueOnError)
	dev := fs.String("device", "", "cpu, webgpu or gpu (default: CUDA when present, else cpu)")
	out := fs.String("o", "", "output PLY (default: <image name>.ply); the encoder's input is written next to it as <name>_input.png")
	repo := fs.String("repo", "microsoft/TRELLIS-image-large", "TRELLIS weights: a Hub repo or a folder (pipeline.json, ckpts/)")
	dino := fs.String("dinov2", "", "DINOv2 encoder repo or folder (default: the transformers port of pipeline.json's, facebook/dinov2-with-registers-large)")
	seed := fs.Uint64("seed", 42, "noise seed (Light's generator: the same seed does not give TRELLIS's torch noise)")
	ssSteps := fs.Int("ss-steps", 0, "sparse-structure sampling steps (default: pipeline.json's, 25)")
	slatSteps := fs.Int("slat-steps", 0, "structured-latent sampling steps (default: pipeline.json's, 25)")
	ssCFG := fs.Float64("ss-cfg", 0, "sparse-structure guidance (default: pipeline.json's, 5; negative disables)")
	slatCFG := fs.Float64("slat-cfg", 0, "structured-latent guidance (default: pipeline.json's, 5; negative disables)")
	pre := fs.Bool("preprocessed", false, "the image is already TRELLIS's 518 x 518 composited input (skip the alpha crop)")
	f32 := fs.Bool("f32", false, "keep F32 weights (default: 16-bit weight storage, F32 compute)")
	matte := fs.String("matte", "speridlabs/idol", "BiRefNet weights for images without a transparent background: a repo or folder with birefnet.safetensors or model.safetensors, or a .safetensors file (empty refuses such images; TRELLIS itself uses rembg's u2net)")
	meshOut := fs.String("mesh", "", "also decode a triangle mesh (TRELLIS's mesh decoder, FlexiCubes) and write it here: .obj (vertex colours) or .ply; y up as TRELLIS's GLB")
	views := fs.Int("views", 4, "also render this many turntable views into <name>_views.png (0: none)")
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: light trellis [flags] <image>\n\nOne image of an object to a 3D Gaussian splat (.ply, the 3DGS layout TRELLIS's save_ply writes, y up) and, with -mesh, a vertex-coloured triangle mesh. An image with a transparent background is cropped by its alpha; any other has its background removed by BiRefNet first (-matte; TRELLIS uses rembg's u2net there, so the cut-out can differ).\n\nflags:")
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
	if e := strings.ToLower(ext(*meshOut)); *meshOut != "" && e != ".obj" && e != ".ply" {
		return fmt.Errorf("-mesh %s: want a .obj or .ply file", *meshOut)
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
	p, err := diffusion.LoadTrellisPipeline(*repo, d, diffusion.TrellisLoadOptions{DINOv2: *dino, Storage16: !*f32,
		Matte: matteFor(img, *matte, *pre), Mesh: *meshOut != ""})
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "loaded TRELLIS in %.1f s\n", time.Since(t0).Seconds())
	t1 := time.Now()
	progress := func(stage string) func(i, n int) {
		return func(i, n int) { fmt.Fprintf(os.Stderr, "\r%s step %d/%d", stage, i+1, n) }
	}
	res, err := p.Generate(diffusion.TrellisOptions{Image: img, Preprocessed: *pre, Seed: *seed,
		Steps: *ssSteps, CFG: *ssCFG, SLatSteps: *slatSteps, SLatCFG: *slatCFG,
		Step: progress("sparse structure"), SLatStep: progress("structured latent")})
	fmt.Fprintln(os.Stderr)
	if err != nil {
		return err
	}
	g := res.Gaussians
	fmt.Fprintf(os.Stderr, "%d occupied voxels, %d Gaussians in %.1f s\n", len(res.Coords), len(g.Opacity), time.Since(t1).Seconds())
	if err := splat.WritePLY(*out, g); err != nil {
		return err
	}
	base := strings.TrimSuffix(*out, ".ply")
	if err := savePNG(base+"_input.png", res.Input); err != nil {
		return err
	}
	if *views > 0 {
		// TRELLIS's render_video camera: radius 2, FOV 40, pitch 0.25, z up.
		im := splat.Turntable(g, splat.Camera{Yaw: 0, Pitch: 0.25, Radius: 2, FOV: 40}, *views, 384, 384,
			color.NRGBA{255, 255, 255, 255})
		if err := savePNG(base+"_views.png", im); err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "wrote %s_views.png\n", base)
	}
	if res.Mesh != nil {
		if err := mesh.Write(*meshOut, res.Mesh.YUp()); err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "wrote %s (%d vertices, %d faces)\n", *meshOut, res.Mesh.NumVertices(), res.Mesh.NumFaces())
	}
	fmt.Fprintf(os.Stderr, "wrote %s and %s_input.png\n", *out, base)
	return nil
}

// matteFor returns the BiRefNet weights to load: only when the image needs
// a cut-out (no alpha, not preprocessed), so cut-out inputs neither
// download nor load it.
func matteFor(img image.Image, matte string, preprocessed bool) string {
	if preprocessed || diffusion.HasAlpha(img) {
		return ""
	}
	return matte
}

func savePNG(path string, im image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := png.Encode(f, im); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
