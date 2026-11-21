package engine

import (
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"time"

	"light/device"
	"light/diffusion"
	"light/dtype"
	"light/hub"
	"light/models"
	"light/tensor"
)

// ImageOptions describes one text-to-image (or image-to-image / editing)
// run of a cached diffusers pipeline repo; zero fields take the
// pipeline's defaults, as `light image` does.
type ImageOptions struct {
	Repo     string
	Prompt   string
	Negative string // SD 1.5 / SDXL only
	Width    int    // pixels (default 1024, SD 1.5: 512; or the first reference image's)
	Height   int
	Steps    int
	Guidance float32
	Seed     uint64
	// Refs are reference images: FLUX.2 / FLUX 3 editing inputs, or the
	// init image of the single-image pipelines (with Strength).
	Refs     []image.Image
	Strength float32
	Device   string // "", "cpu", "gpu", "webgpu": PickDevice
	F32      bool   // keep F32 weights (default: 16-bit storage, F32 compute)
	// OnPhase reports coarse progress ("load", "generate", "decode"); OnStep
	// the denoising steps where the pipeline exposes them (SD 1.5, SDXL).
	OnPhase func(phase string)
	OnStep  func(step, total int)
	// OnDownload reports component files fetched from the Hub before
	// loading (done of total bytes), so a first run can show the download
	// instead of a silent "load".
	OnDownload func(file string, done, total int64)
	// Pipeline, when set, composes the pipeline from separately chosen
	// parts (PipelineSpec); Repo then only fills its empty slots. FLUX.1
	// for now.
	Pipeline *PipelineSpec
}

// ImageResult is the generated image with its timings.
type ImageResult struct {
	Image    *image.NRGBA
	Pipeline string
	LoadS    float64
	GenS     float64
}

// GenerateImage runs the pipeline of a cached repo (its model_index.json
// names the class) and returns the image. Weights load for the call and
// are released with the result; the Studio and the CLI both go through
// here.
func GenerateImage(o ImageOptions) (*ImageResult, error) {
	if o.Repo == "" || o.Prompt == "" {
		return nil, errors.New("engine: image generation needs a repo and a prompt")
	}
	d, err := PickDevice(o.Device)
	if err != nil {
		return nil, err
	}
	phase := func(p string) {
		if o.OnPhase != nil {
			o.OnPhase(p)
		}
	}
	var class string

	if o.Pipeline != nil {
		if err := o.Pipeline.Validate(); err != nil {
			return nil, err
		}
		switch o.Pipeline.Family {
		case "flux", "":
			class = "FluxPipeline"
		case "flux2":
			class = "Flux2KleinPipeline"
		default:
			return nil, fmt.Errorf("pipeline: family %q is not composable yet (flux, flux2)", o.Pipeline.Family)
		}
		if t := o.Pipeline.stageTarget("denoise"); t == "local/cpu" {
			o.Device = "cpu"
		} else if t == "local/gpu" {
			o.Device = "gpu"
		}
	} else if class, err = pipelineClass(o.Repo); err != nil {
		return nil, err
	}
	s16 := !o.F32
	ed := d
	if d.Kind() == device.KindWebGPU {
		ed = device.CPU()
	}
	initImage := func() (diffusion.Img2Img, error) {
		switch len(o.Refs) {
		case 0:
			return diffusion.Img2Img{}, nil
		case 1:
			st := o.Strength
			if st == 0 {
				st = 0.6
			}
			return diffusion.Img2Img{Image: o.Refs[0], Strength: st}, nil
		}
		return diffusion.Img2Img{}, fmt.Errorf("%s: one reference image (the init image) at most", o.Repo)
	}
	if o.OnDownload != nil {
		if _, err := diffusion.SnapshotWithProgress(o.Repo, o.OnDownload); err != nil {
			return nil, err
		}
	}
	phase("load")
	t0 := time.Now()
	var img diffusion.Image
	var loadS float64
	switch class {
	case "Flux2KleinPipeline", "Flux2Pipeline":
		lo := diffusion.Flux2LoadOptions{Storage16: s16, EncoderDevice: ed, Sequential: true}
		repo := o.Repo
		if o.Pipeline != nil {
			if lo.Components, err = flux2Components(o); err != nil {
				return nil, err
			}
			if o.Pipeline.stageTarget("text_encode") == "local/cpu" {
				lo.EncoderDevice = device.CPU()
			}
			if repo == "" {
				repo = lo.Components["transformer"]
			}
		}
		tdir := lo.Components["transformer"]
		if tdir == "" {
			tdir = filepath.Join(snapshotDir(repo), "transformer")
		}
		lo.KeepFP8 = s16 && device.IsCUDA(d)
		lo.Mixed16 = s16 && device.IsCUDA(d)
		lo.StreamBlocks = s16 && device.IsCUDA(d) && dirBytes(tdir, "*.safetensors") > 13<<30
		p, err := diffusion.LoadFlux2(repo, d, lo)
		if err != nil {
			return nil, err
		}
		loadS = time.Since(t0).Seconds()
		phase("generate")
		res, err := p.Generate(diffusion.Flux2Options{Prompt: o.Prompt, Width: o.Width, Height: o.Height, Steps: o.Steps, Guidance: o.Guidance, Seed: o.Seed, Images: o.Refs})
		if err != nil {
			return nil, err
		}
		img = diffusion.Image{Pixels: res.Image}
	case "ZImagePipeline":
		i2i, err := initImage()
		if err != nil {
			return nil, err
		}
		p, err := diffusion.LoadZImage(o.Repo, d, diffusion.ZImageLoadOptions{Storage16: s16, EncoderDevice: ed, Sequential: true})
		if err != nil {
			return nil, err
		}
		loadS = time.Since(t0).Seconds()
		phase("generate")
		res, err := p.Generate(diffusion.ZImageOptions{Prompt: o.Prompt, Width: o.Width, Height: o.Height, Steps: o.Steps, Guidance: o.Guidance, Seed: o.Seed, Img2Img: i2i})
		if err != nil {
			return nil, err
		}
		img = diffusion.Image{Pixels: res.Image}
	case "FluxPipeline":
		i2i, err := initImage()
		if err != nil {
			return nil, err
		}
		w, h := o.Width, o.Height
		if i2i.Image != nil && w == 0 && h == 0 {
			b := i2i.Image.Bounds()
			w, h = b.Dx()/16*16, b.Dy()/16*16
		}
		// FLUX.1's 12B transformer streams its blocks on CUDA (it does not
		// fit a 16 GB card resident); the tiny / small ones load whole.
		lo := diffusion.FluxLoadOptions{Storage16: s16, Sequential: true}
		repo := o.Repo
		if o.Pipeline != nil {
			if lo.Components, err = fluxComponents(o); err != nil {
				return nil, err
			}
			if lo.Loras, err = fluxLoras(o.Pipeline); err != nil {
				return nil, err
			}
			if o.Pipeline.stageTarget("text_encode") == "local/cpu" {
				lo.EncoderDevice = device.CPU()
			}
			if repo == "" {
				repo = lo.Components["transformer"] // a folder: Snapshot returns it as is
			}
		}
		tdir := lo.Components["transformer"]
		if tdir == "" {
			tdir = filepath.Join(snapshotDir(repo), "transformer")
		}
		// FP8 checkpoints stay FP8 on CUDA (12 GB resident); BF16 ones over
		// 13 GB stream their blocks.
		tbytes := dirBytes(tdir, "*.safetensors")
		if st, err := os.Stat(tdir); err == nil && !st.IsDir() {
			tbytes = st.Size()
		}
		lo.KeepFP8 = s16 && device.IsCUDA(d)
		lo.Mixed16 = s16 && device.IsCUDA(d) // BF16 tensor-core compute, as diffusers
		lo.StreamBlocks = s16 && device.IsCUDA(d) && tbytes > 13<<30
		p, err := diffusion.LoadFluxOpts(repo, d, lo)
		if err != nil {
			return nil, err
		}
		loadS = time.Since(t0).Seconds()
		phase("generate")
		res, err := p.Generate(diffusion.FluxOptions{Prompt: o.Prompt, Width: w, Height: h, Steps: o.Steps, Guidance: o.Guidance, Seed: o.Seed, Img2Img: i2i})
		if err != nil {
			return nil, err
		}
		img = diffusion.Image{Pixels: res.Image}
	case "Flux3ModularPipeline":
		p, err := diffusion.LoadFlux3(o.Repo, d, diffusion.Flux3LoadOptions{Storage16: s16, EncoderDevice: ed, Sequential: true})
		if err != nil {
			return nil, err
		}
		loadS = time.Since(t0).Seconds()
		phase("generate")
		res, err := p.Generate(diffusion.Flux3Options{Prompt: o.Prompt, Width: o.Width, Height: o.Height, Steps: o.Steps, Seed: o.Seed, Images: o.Refs})
		if err != nil {
			return nil, err
		}
		img = diffusion.Image{Pixels: res.Image}
	case "QwenImage21Pipeline":
		if len(o.Refs) > 0 {
			return nil, fmt.Errorf("%s: reference images are not supported for Qwen-Image 2.1 yet", o.Repo)
		}
		p, err := diffusion.LoadQwenImage21(o.Repo, d, diffusion.QwenImage21LoadOptions{Storage16: s16, EncoderDevice: ed, Sequential: true})
		if err != nil {
			return nil, err
		}
		loadS = time.Since(t0).Seconds()
		phase("generate")
		res, err := p.Generate(diffusion.QwenImage21Options{Prompt: o.Prompt, Width: o.Width, Height: o.Height, Steps: o.Steps, Seed: o.Seed})
		if err != nil {
			return nil, err
		}
		img = diffusion.Image{Pixels: res.Image}
	case "StableDiffusionPipeline", "StableDiffusionXLPipeline":
		i2i, err := initImage()
		if err != nil {
			return nil, err
		}
		var opts models.SDGenerateOpts
		var run func(models.SDGenerateOpts) (*tensor.Tensor, error)
		if class == "StableDiffusionXLPipeline" {
			wdt := dtype.BF16
			if o.F32 {
				wdt = dtype.F32
			}
			sd, err := models.LoadSDXLFromHF(o.Repo, wdt, d)
			if err != nil {
				return nil, err
			}
			opts = models.DefaultSDXLGenerateOpts(o.Prompt)
			run = func(op models.SDGenerateOpts) (*tensor.Tensor, error) { return sd.Generate(op, d) }
		} else {
			sd, err := models.LoadSD15FromHF(o.Repo, d) // F32: the CPU convolution has no 16-bit kernel
			if err != nil {
				return nil, err
			}
			opts = models.DefaultSDGenerateOpts(o.Prompt)
			run = func(op models.SDGenerateOpts) (*tensor.Tensor, error) { return sd.Generate(op, d) }
		}
		loadS = time.Since(t0).Seconds()
		opts.Negative = o.Negative
		if o.Width != 0 || o.Height != 0 || i2i.Image != nil {
			opts.Width, opts.Height = o.Width, o.Height
		}
		if o.Steps != 0 {
			opts.NumInference = o.Steps
		}
		if o.Guidance != 0 {
			opts.GuidanceScale = o.Guidance
		}
		opts.Seed, opts.InitImage, opts.Strength = o.Seed, i2i.Image, i2i.Strength
		opts.OnPhase = func(p string) {
			switch p {
			case "diffuse":
				phase("generate")
			case "vae decode":
				phase("decode")
			}
		}
		if o.OnStep != nil {
			opts.OnStep = func(step, total, _ int, _ time.Duration) { o.OnStep(step+1, total) }
		}
		hwc, err := run(opts) // reports "generate" and "decode" itself

		if err != nil {
			return nil, err
		}
		chw, err := sdToCHW(hwc)
		if err != nil {
			return nil, err
		}
		img = diffusion.Image{Pixels: chw}
	default:
		return nil, fmt.Errorf("%s: pipeline %q not supported", o.Repo, class)
	}
	t1 := time.Now()
	if class != "StableDiffusionPipeline" && class != "StableDiffusionXLPipeline" {
		phase("decode")
	}
	rgba, err := img.RGBA()
	if err != nil {
		return nil, err
	}
	return &ImageResult{Image: rgba, Pipeline: class, LoadS: loadS, GenS: t1.Sub(t0).Seconds() - loadS}, nil
}

// sdToCHW turns the SD pipelines' (H, W, 3) image into diffusion.Image's
// (1, 3, H, W).
func sdToCHW(hwc *tensor.Tensor) (*tensor.Tensor, error) {
	chw, err := hwc.Permute(2, 0, 1)
	if err != nil {
		return nil, err
	}
	if chw, err = chw.Contiguous(); err != nil {
		return nil, err
	}
	return chw.Reshape(1, 3, hwc.Dim(0), hwc.Dim(1))
}

// pipelineClass reads a cached repo's pipeline class (model_index.json,
// or modular_model_index.json for FLUX 3).
func pipelineClass(repo string) (string, error) {
	idx, err := hub.Download(repo, "main", "model_index.json")
	if err != nil {
		if _, merr := hub.Download(repo, "main", "modular_model_index.json"); merr != nil {
			return "", err
		}
		return "Flux3ModularPipeline", nil
	}
	raw, err := os.ReadFile(idx)
	if err != nil {
		return "", err
	}
	var mi struct {
		Class string `json:"_class_name"`
	}
	if err := json.Unmarshal(raw, &mi); err != nil {
		return "", err
	}
	return mi.Class, nil
}

// snapshotDir is the local folder of repo ("" when it is not available).
func snapshotDir(repo string) string {
	dir, err := diffusion.Snapshot(repo)
	if err != nil {
		return ""
	}
	return dir
}

// dirBytes is the size on disk of dir's files matching glob (0 when none),
// to decide whether a transformer's blocks must be streamed.
func dirBytes(dir, glob string) int64 {
	files, _ := filepath.Glob(filepath.Join(dir, glob))
	var n int64
	for _, f := range files {
		if st, err := os.Stat(f); err == nil {
			n += st.Size()
		}
	}
	return n
}
