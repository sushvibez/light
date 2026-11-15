package diffusion

import (
	"fmt"
	"image"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"

	"light/device"
	"light/dtype"
	"light/hub"
	"light/models"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// IdolPipeline is speridlabs/idol: one image of an object to 3D Gaussians.
// The image (with its background removed) is encoded by DINOv3 ViT-H+ and
// the FLUX.2 VAE, the denoiser samples the object latent and the input
// camera (50 Euler steps, CFG 3), the octree decoder places anchor points
// and the Gaussian decoder turns each into 32 Gaussians.
type IdolPipeline struct {
	Model *models.Idol
	DINO  *models.DINOv3
	VAE   *models.FluxVAE
	Dev   device.Device
}

// IdolLoadOptions adjust LoadIdolPipeline.
type IdolLoadOptions struct {
	// DINOv3 and FluxVAE override the image encoders named in config.json:
	// a Hub repo or a local folder (for the VAE, one with the autoencoder in
	// vae/; any FLUX.2 repo has the same one, e.g. FLUX.2-klein-4B).
	DINOv3, FluxVAE string
	Storage16       bool // 16-bit linear weights (F16, as the checkpoint's blocks)
}

// fetchFiles returns a local folder holding files (relative paths) of repo:
// repo itself when it is a folder, else the Hub cache after downloading
// just those files.
func fetchFiles(repo string, files ...string) (string, error) {
	if st, err := os.Stat(repo); err == nil && st.IsDir() {
		return repo, nil
	}
	var dir string
	for _, f := range files {
		p, err := hub.Download(repo, "main", f)
		if err != nil {
			return "", fmt.Errorf("%s/%s: %w", repo, f, err)
		}
		// The snapshot root: the path minus the file's own relative path.
		dir = filepath.Clean(strings.TrimSuffix(filepath.ToSlash(p), "/"+f))
	}
	return dir, nil
}

// LoadIdolPipeline loads Idol (a speridlabs/idol repo or folder) and its
// image encoders onto dev.
func LoadIdolPipeline(repo string, dev device.Device, o IdolLoadOptions) (*IdolPipeline, error) {
	dir, err := fetchFiles(repo, "config.json", "denoiser.safetensors", "decoder.safetensors", "decoder_gs.safetensors")
	if err != nil {
		return nil, err
	}
	cfg, err := models.LoadIdolConfig(dir)
	if err != nil {
		return nil, err
	}
	load := func(f func() error) error {
		if o.Storage16 {
			return nn.WithWeightStorage16(dtype.F16, f)
		}
		return f()
	}
	p := &IdolPipeline{Dev: dev}
	if err := load(func() (err error) { p.Model, err = models.LoadIdol(dir, dev); return }); err != nil {
		return nil, err
	}
	dino := o.DINOv3
	if dino == "" {
		dino = cfg.ImageEncoders["dinov3"]
	}
	ddir, err := fetchFiles(dino, "config.json", "model.safetensors")
	if err != nil {
		return nil, fmt.Errorf("idol: DINOv3 (gated: accept its licence on the Hub and log in): %w", err)
	}
	if err := load(func() (err error) { p.DINO, err = models.LoadDINOv3(ddir, dev); return }); err != nil {
		return nil, err
	}
	vae := o.FluxVAE
	if vae == "" {
		vae = cfg.ImageEncoders["flux2_vae"]
	}
	vdir := filepath.Join(vae, "vae")
	if st, err := os.Stat(vdir); err != nil || !st.IsDir() {
		if vdir, err = fetchFiles(vae, "vae/config.json", "vae/diffusion_pytorch_model.safetensors"); err != nil {
			return nil, err
		}
		vdir = filepath.Join(vdir, "vae")
	}
	if p.VAE, err = models.LoadFluxVAE(vdir, dev); err != nil {
		return nil, fmt.Errorf("idol: FLUX.2 VAE: %w", err)
	}
	if p.VAE.Encoder == nil || p.VAE.BNMean == nil {
		return nil, fmt.Errorf("idol: %s has no FLUX.2 VAE encoder with batch-norm statistics", vae)
	}
	return p, nil
}

var (
	imagenetMean = [3]float32{0.485, 0.456, 0.406}
	imagenetStd  = [3]float32{0.229, 0.224, 0.225}
)

// Condition returns idol's image tokens for pixels (3, S, S) in [0, 1]:
// feature1, the DINOv3 tokens (ImageNet-normalised input, a LayerNorm
// without affine on top), and feature2, the FLUX.2 VAE latent sample
// (mean + std·eps, eps (C, S/8, S/8) normal) patchified 2 x 2, normalised by
// the VAE's batch-norm statistics and led by 5 zero tokens (CLS +
// registers' places).
func (p *IdolPipeline) Condition(pixels []float32, S int, eps []float32) (f1, f2 *tensor.Tensor, err error) {
	norm := make([]float32, len(pixels))
	for c := range 3 {
		for i := range S * S {
			norm[c*S*S+i] = (pixels[c*S*S+i] - imagenetMean[c]) / imagenetStd[c]
		}
	}
	x, err := tensor.New(norm, []int{1, 3, S, S}, p.Dev)
	if err != nil {
		return nil, nil, err
	}
	h, err := p.DINO.Forward(x)
	if err != nil {
		return nil, nil, err
	}
	if f1, err = models.IdolImageTokens(h); err != nil {
		return nil, nil, err
	}
	sv := make([]float32, len(pixels))
	for i, v := range pixels {
		sv[i] = v*2 - 1
	}
	if x, err = tensor.New(sv, []int{1, 3, S, S}, p.Dev); err != nil {
		return nil, nil, err
	}
	mean, logvar, err := p.VAE.Encoder.Moments(x)
	if err != nil {
		return nil, nil, err
	}
	z := mean
	if logvar != nil && eps != nil {
		e, err := tensor.New(eps, mean.Shape().Dims(), p.Dev)
		if err != nil {
			return nil, nil, err
		}
		std, err := ops.Affine(logvar, 0.5, 0)
		if err != nil {
			return nil, nil, err
		}
		if std, err = ops.Exp(std); err != nil {
			return nil, nil, err
		}
		if e, err = ops.Mul(std, e); err != nil {
			return nil, nil, err
		}
		if z, err = ops.Add(mean, e); err != nil {
			return nil, nil, err
		}
	}
	o := &idolOps{}
	C, zh, zw := z.Dim(1), z.Dim(2), z.Dim(3)
	t := o.reshape(z, 1, C, zh/2, 2, zw/2, 2)
	t = o.reshape(o.perm(t, 0, 1, 3, 5, 2, 4), 1, 4*C, zh/2, zw/2)
	bm := o.new(p.VAE.BNMean, []int{1, 4 * C, 1, 1}, p.Dev)
	bs := o.new(p.VAE.BNStd, []int{1, 4 * C, 1, 1}, p.Dev)
	t = o.div(o.sub(t, bm), bs)
	t = o.reshape(o.perm(o.reshape(t, 1, 4*C, zh/2*zw/2), 0, 2, 1), 1, zh/2*zw/2, 4*C)
	pad := o.zeros([]int{1, 5, 4 * C}, p.Dev)
	f2 = o.cat(1, pad, t)
	return f1, f2, o.err
}

// IdolOptions control one generation.
type IdolOptions struct {
	Image        image.Image // an object with its background removed (alpha), or any photo once BiRefNet is ported
	Mode         string      // "synthetic" (renders, default) or "real" (photos): the crop margin
	CropMargin   float64     // overrides Mode's margin
	Seed         uint64
	Steps        int     // default 50
	CFG          float64 // default 3; negative disables guidance
	Shift        float64 // default 3
	NumGaussians int     // 32768 .. 1048576, default 262144, rounded down to a multiple of 32
	Step         func(i, n int)
}

// IdolCamera is the predicted input camera.
type IdolCamera struct {
	Yaw, Pitch  float64 // radians
	Radius, FOV float64 // FOV in degrees (vertical)
}

// IdolResult is the generated object.
type IdolResult struct {
	Gaussians *models.IdolGaussians
	Input     *image.NRGBA // the image the model saw (cropped, background black)
	Camera    IdolCamera
}

// Generate runs idol on one image.
func (p *IdolPipeline) Generate(o IdolOptions) (*IdolResult, error) {
	cfg := p.Model.Cfg
	if o.Mode == "" {
		o.Mode = "synthetic"
	}
	margin := o.CropMargin
	if margin == 0 {
		var ok bool
		if margin, ok = IdolCropMargin[o.Mode]; !ok {
			return nil, fmt.Errorf("idol: mode %q (want synthetic or real)", o.Mode)
		}
	}
	if o.NumGaussians == 0 {
		o.NumGaussians = 262144
	}
	if o.CFG == 0 {
		o.CFG = 3
	} else if o.CFG < 0 {
		o.CFG = 0
	}
	S := cfg.ImageSize
	pixels, prepared, err := idolPreprocess(o.Image, S, margin)
	if err != nil {
		return nil, err
	}
	rng := rand.New(rand.NewPCG(o.Seed, o.Seed^0x9e3779b97f4a7c15))
	normal := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	vs := p.VAE.Encoder
	scale := 1
	for range len(vs.Down) - 1 {
		scale *= 2
	}
	f1, f2, err := p.Condition(pixels, S, normal(vs.Cfg.LatentChannels*(S/scale)*(S/scale)))
	if err != nil {
		return nil, err
	}
	a := cfg.Denoiser.Args
	lat, err := tensor.New(normal(a.QTokenLength*a.InChannels), []int{1, a.QTokenLength, a.InChannels}, p.Dev)
	if err != nil {
		return nil, err
	}
	cam, err := tensor.New(normal(a.CamChannels), []int{1, 1, a.CamChannels}, p.Dev)
	if err != nil {
		return nil, err
	}
	lat, cam, err = models.IdolSample(p.Model.Denoiser, lat, cam, f1, f2,
		models.IdolSampleOptions{Steps: o.Steps, CFG: o.CFG, Shift: o.Shift, Step: o.Step})
	if err != nil {
		return nil, err
	}
	G := p.Model.Gaussian.Args.Representation.NumGaussians
	nAnchors := max(1, o.NumGaussians/G)
	anchors, err := models.IdolSampleAnchors(p.Model.Octree, lat, nAnchors, cfg.MaxVoxelLevel, rng.Float32)
	if err != nil {
		return nil, err
	}
	pts, err := tensor.New(anchors, []int{1, nAnchors, 3}, p.Dev)
	if err != nil {
		return nil, err
	}
	feats, err := p.Model.Gaussian.Forward(pts, lat)
	if err != nil {
		return nil, err
	}
	g, err := p.Model.Gaussian.ToGaussians(anchors, feats)
	if err != nil {
		return nil, err
	}
	cc, err := cam.To(device.CPU())
	if err != nil {
		return nil, err
	}
	return &IdolResult{Gaussians: g, Input: prepared, Camera: idolDecodeCamera(tensor.MustSlice[float32](cc))}, nil
}

// idolDecodeCamera is idol's decode_camera: [direction (3), inverse
// distance, half width].
func idolDecodeCamera(v []float32) IdolCamera {
	dx, dy, dz := float64(v[0]), float64(v[1]), float64(v[2])
	inv, hw := float64(v[3]), float64(v[4])
	if n := math.Sqrt(dx*dx + dy*dy + dz*dz); n > 1e-6 {
		dx, dy, dz = dx/n, dy/n, dz/n
	} else {
		dx, dy, dz = 0, 0, 1
	}
	var c IdolCamera
	if inv < 1e-3 { // orthographic
		c.Radius = 1000
		c.FOV = 2 * math.Atan(hw/c.Radius) * 180 / math.Pi
	} else {
		c.Radius = 1 / inv
		c.FOV = 2 * math.Atan(hw*inv) * 180 / math.Pi
	}
	c.Yaw = math.Atan2(dx, dy)
	c.Pitch = math.Asin(math.Min(math.Max(dz, -1), 1))
	return c
}

// idolOps chains tensor ops for the pipeline, keeping the first error.
type idolOps struct{ err error }

func (o *idolOps) do(f func() (*tensor.Tensor, error)) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	t, err := f()
	o.err = err
	return t
}
func (o *idolOps) reshape(a *tensor.Tensor, dims ...int) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) {
		c, err := a.Contiguous()
		if err != nil {
			return nil, err
		}
		return c.Reshape(dims...)
	})
}
func (o *idolOps) perm(a *tensor.Tensor, p ...int) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) { return a.Permute(p...) })
}
func (o *idolOps) sub(a, b *tensor.Tensor) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) { return ops.Sub(a, b) })
}
func (o *idolOps) div(a, b *tensor.Tensor) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) { return ops.Div(a, b) })
}
func (o *idolOps) cat(axis int, ts ...*tensor.Tensor) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) { return ops.Concat(ts, axis) })
}
func (o *idolOps) new(v []float32, dims []int, dev device.Device) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) { return tensor.New(v, dims, dev) })
}
func (o *idolOps) zeros(dims []int, dev device.Device) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) { return tensor.Zeros(dtype.F32, dims, dev) })
}
