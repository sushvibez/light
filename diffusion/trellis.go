package diffusion

import (
	"encoding/json"
	"fmt"
	"image"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"

	"light/device"
	"light/dtype"
	"light/mesh"
	"light/models"
	"light/nn"
	"light/splat"
	"light/tensor"
)

// TrellisPipeline is microsoft/TRELLIS-image-large's TrellisImageTo3DPipeline
// with the Gaussian decoder: an image of an object (alpha-cut) to 3D
// Gaussians. Stage 1: DINOv2 ViT-L/14 with registers encodes the image, the
// sparse-structure flow model samples the 16³ x 8 occupancy latent (25
// Euler steps, CFG 5 for t in [0.5, 1]) and its decoder turns it into 64³
// occupancy logits; coords = logits > 0. Stage 2: the SLat flow model
// samples 8 channels on each occupied voxel (same sampler settings),
// un-normalised by slat_normalization, and the Gaussian decoder turns each
// voxel into 32 Gaussians; optionally (TrellisLoadOptions.Mesh) the mesh
// decoder turns it into a triangle mesh by FlexiCubes. The radiance-field
// decoder is not ported.
type TrellisPipeline struct {
	Flow    *models.TrellisFlow
	Decoder *models.TrellisSSDecoder
	DINO    *models.DINOv2
	Sampler models.TrellisSampleOptions // pipeline.json's sparse_structure_sampler.params
	// Stage 2; nil with TrellisLoadOptions.SparseStructureOnly.
	SLatFlow    *models.TrellisSLatFlow
	SLatDecoder *models.TrellisSLatDecoder
	MeshDecoder *models.TrellisSLatMeshDecoder // nil unless TrellisLoadOptions.Mesh
	SLatSampler models.TrellisSampleOptions    // pipeline.json's slat_sampler.params
	SLatMean    []float64                      // slat_normalization
	SLatStd     []float64
	// BiRefNet cuts out images without alpha (TRELLIS uses rembg's u2net
	// there); nil unless TrellisLoadOptions.Matte names it, and such images
	// are then refused.
	BiRefNet *models.BiRefNet
	Dev      device.Device
}

// TrellisLoadOptions adjust LoadTrellisPipeline.
type TrellisLoadOptions struct {
	// DINOv2 overrides the image encoder: a Hub repo or local folder with
	// transformers' Dinov2WithRegistersModel (config.json,
	// model.safetensors). Default: the HF port of pipeline.json's torch.hub
	// image_cond_model (dinov2_vitl14_reg -> facebook/dinov2-with-registers-large).
	DINOv2    string
	Storage16 bool // 16-bit linear weights (F16, as the checkpoints' torsos)
	// SparseStructureOnly loads stage 1 only (no SLat flow model or
	// Gaussian decoder).
	SparseStructureOnly bool
	// Mesh also loads pipeline.json's slat_decoder_mesh: Generate then
	// returns a mesh too.
	Mesh bool
	// Matte names BiRefNet weights for images without alpha: a .safetensors
	// file, or a Hub repo / folder with birefnet.safetensors (speridlabs/idol)
	// or model.safetensors (ZhengPeng7/BiRefNet). Empty: no background
	// removal.
	Matte string
}

// trellisDINOv2Repos maps TRELLIS's torch.hub image_cond_model names to
// the transformers checkpoints holding the same weights.
var trellisDINOv2Repos = map[string]string{
	"dinov2_vits14_reg": "facebook/dinov2-with-registers-small",
	"dinov2_vitb14_reg": "facebook/dinov2-with-registers-base",
	"dinov2_vitl14_reg": "facebook/dinov2-with-registers-large",
	"dinov2_vitg14_reg": "facebook/dinov2-with-registers-giant",
}

// trellisSamplerConfig is a pipeline.json sampler entry.
type trellisSamplerConfig struct {
	Name   string `json:"name"`
	Params struct {
		Steps       int        `json:"steps"`
		CFGStrength float64    `json:"cfg_strength"`
		CFGInterval [2]float64 `json:"cfg_interval"`
		RescaleT    float64    `json:"rescale_t"`
	} `json:"params"`
}

func (c *trellisSamplerConfig) options(which string) (models.TrellisSampleOptions, error) {
	if c.Name != "FlowEulerGuidanceIntervalSampler" {
		return models.TrellisSampleOptions{}, fmt.Errorf("trellis: %s %q (want FlowEulerGuidanceIntervalSampler)", which, c.Name)
	}
	sp := c.Params
	o := models.TrellisSampleOptions{Steps: sp.Steps, CFG: sp.CFGStrength, Interval: sp.CFGInterval, RescaleT: sp.RescaleT}
	if sp.CFGStrength == 0 {
		o.CFG = -1 // explicit 0: no guidance
	}
	return o, nil
}

// trellisPipelineConfig is the part of pipeline.json the Gaussian path uses.
type trellisPipelineConfig struct {
	Args struct {
		Models            map[string]string    `json:"models"`
		Sampler           trellisSamplerConfig `json:"sparse_structure_sampler"`
		SLatSampler       trellisSamplerConfig `json:"slat_sampler"`
		SLatNormalization struct {
			Mean, Std []float64
		} `json:"slat_normalization"`
		ImageCondModel string `json:"image_cond_model"`
	} `json:"args"`
}

// LoadTrellisPipeline loads TRELLIS (a microsoft/TRELLIS-image-* repo or
// folder: pipeline.json and, under ckpts/, the sparse-structure flow model
// and decoder, the SLat flow model, the Gaussian decoder and, with
// TrellisLoadOptions.Mesh, the mesh decoder) and its DINOv2
// image encoder onto dev.
func LoadTrellisPipeline(repo string, dev device.Device, opts ...TrellisLoadOptions) (*TrellisPipeline, error) {
	var o TrellisLoadOptions
	if len(opts) > 0 {
		o = opts[0]
	}
	dir, err := fetchFiles(repo, "pipeline.json")
	if err != nil {
		return nil, err
	}
	raw, err := os.ReadFile(filepath.Join(dir, "pipeline.json"))
	if err != nil {
		return nil, err
	}
	var pc trellisPipelineConfig
	if err := json.Unmarshal(raw, &pc); err != nil {
		return nil, fmt.Errorf("trellis: pipeline.json: %w", err)
	}
	a := pc.Args
	ssOpts, err := a.Sampler.options("sparse-structure sampler")
	if err != nil {
		return nil, err
	}
	flowName, decName := a.Models["sparse_structure_flow_model"], a.Models["sparse_structure_decoder"]
	if flowName == "" || decName == "" {
		return nil, fmt.Errorf("trellis: pipeline.json names no sparse-structure flow model / decoder")
	}
	files := []string{flowName + ".json", flowName + ".safetensors", decName + ".json", decName + ".safetensors"}
	slatName, gsName := a.Models["slat_flow_model"], a.Models["slat_decoder_gs"]
	var slatOpts models.TrellisSampleOptions
	if !o.SparseStructureOnly {
		if slatName == "" || gsName == "" {
			return nil, fmt.Errorf("trellis: pipeline.json names no slat_flow_model / slat_decoder_gs (set SparseStructureOnly for stage 1)")
		}
		if slatOpts, err = a.SLatSampler.options("slat sampler"); err != nil {
			return nil, err
		}
		files = append(files, slatName+".json", slatName+".safetensors", gsName+".json", gsName+".safetensors")
	}
	meshName := a.Models["slat_decoder_mesh"]
	if o.Mesh {
		if o.SparseStructureOnly {
			return nil, fmt.Errorf("trellis: the mesh decoder needs stage 2 (SparseStructureOnly is set)")
		}
		if meshName == "" {
			return nil, fmt.Errorf("trellis: pipeline.json names no slat_decoder_mesh")
		}
		files = append(files, meshName+".json", meshName+".safetensors")
	}
	if dir, err = fetchFiles(repo, files...); err != nil {
		return nil, err
	}
	load := func(f func() error) error {
		if o.Storage16 {
			return nn.WithWeightStorage16(dtype.F16, f)
		}
		return f()
	}
	p := &TrellisPipeline{Dev: dev, Sampler: ssOpts, SLatSampler: slatOpts,
		SLatMean: a.SLatNormalization.Mean, SLatStd: a.SLatNormalization.Std}
	if err := load(func() (err error) {
		p.Flow, err = models.LoadTrellisFlow(filepath.Join(dir, filepath.FromSlash(flowName)), dev)
		return
	}); err != nil {
		return nil, err
	}
	if err := load(func() (err error) {
		p.Decoder, err = models.LoadTrellisSSDecoder(filepath.Join(dir, filepath.FromSlash(decName)), dev)
		return
	}); err != nil {
		return nil, err
	}
	if !o.SparseStructureOnly {
		if err := load(func() (err error) {
			p.SLatFlow, err = models.LoadTrellisSLatFlow(filepath.Join(dir, filepath.FromSlash(slatName)), dev)
			return
		}); err != nil {
			return nil, err
		}
		if err := load(func() (err error) {
			p.SLatDecoder, err = models.LoadTrellisSLatDecoder(filepath.Join(dir, filepath.FromSlash(gsName)), dev)
			return
		}); err != nil {
			return nil, err
		}
		if o.Mesh {
			if err := load(func() (err error) {
				p.MeshDecoder, err = models.LoadTrellisSLatMeshDecoder(filepath.Join(dir, filepath.FromSlash(meshName)), dev)
				return
			}); err != nil {
				return nil, err
			}
			if p.MeshDecoder.Args.LatentChannels != p.SLatFlow.Args.OutChannels {
				return nil, fmt.Errorf("trellis: the mesh decoder takes %d latent channels, the SLat has %d", p.MeshDecoder.Args.LatentChannels, p.SLatFlow.Args.OutChannels)
			}
		}
		if c := p.SLatFlow.Args.OutChannels; len(p.SLatMean) != c || len(p.SLatStd) != c {
			return nil, fmt.Errorf("trellis: slat_normalization has %d / %d values for %d channels", len(p.SLatMean), len(p.SLatStd), c)
		}
		if p.SLatFlow.Args.CondChannels != p.Flow.Args.CondChannels {
			return nil, fmt.Errorf("trellis: the SLat flow model expects %d-wide tokens, stage 1 %d", p.SLatFlow.Args.CondChannels, p.Flow.Args.CondChannels)
		}
	}
	dino := o.DINOv2
	if dino == "" {
		if dino = trellisDINOv2Repos[a.ImageCondModel]; dino == "" {
			return nil, fmt.Errorf("trellis: image_cond_model %q has no known transformers checkpoint; set TrellisLoadOptions.DINOv2", a.ImageCondModel)
		}
	}
	ddir, err := fetchFiles(dino, "config.json", "model.safetensors")
	if err != nil {
		return nil, fmt.Errorf("trellis: DINOv2: %w", err)
	}
	if err := load(func() (err error) { p.DINO, err = models.LoadDINOv2(ddir, dev); return }); err != nil {
		return nil, err
	}
	if o.Matte != "" {
		path, err := trellisMattePath(o.Matte)
		if err != nil {
			return nil, err
		}
		if err := load(func() (err error) { p.BiRefNet, err = models.LoadBiRefNet(path, dev); return }); err != nil {
			return nil, err
		}
	}
	if p.DINO.Cfg.HiddenSize != p.Flow.Args.CondChannels {
		return nil, fmt.Errorf("trellis: DINOv2 width %d, the flow model expects %d", p.DINO.Cfg.HiddenSize, p.Flow.Args.CondChannels)
	}
	return p, nil
}

// ImageSize is the side of the square image the encoder takes (518).
func (p *TrellisPipeline) ImageSize() int { return p.DINO.Cfg.ImageSize }

// Condition is encode_image for pixels (3, S, S) in [0, 1], S =
// ImageSize(): ImageNet-normalised, DINOv2's x_prenorm tokens (CLS,
// registers, patches) through a LayerNorm without affine; (1, L, D).
func (p *TrellisPipeline) Condition(pixels []float32, S int) (*tensor.Tensor, error) {
	if len(pixels) != 3*S*S {
		return nil, fmt.Errorf("trellis: %d pixel values, want 3 x %d x %d", len(pixels), S, S)
	}
	norm := make([]float32, len(pixels))
	for c := range 3 {
		for i := range S * S {
			norm[c*S*S+i] = (pixels[c*S*S+i] - imagenetMean[c]) / imagenetStd[c]
		}
	}
	x, err := tensor.New(norm, []int{1, 3, S, S}, p.Dev)
	if err != nil {
		return nil, err
	}
	h, err := p.DINO.Hidden(x)
	if err != nil {
		return nil, err
	}
	return models.TrellisImageTokens(h)
}

// SampleSparseStructure is sample_sparse_structure for one image's tokens
// cond (1, L, D) from noise (in x R³ values, R = 16): the sampled latent
// (1, 8, R, R, R), the decoder's occupancy logits (1, 1, 4R, 4R, 4R) and
// the occupied voxels (b, x, y, z).
func (p *TrellisPipeline) SampleSparseStructure(cond *tensor.Tensor, noise []float32, o models.TrellisSampleOptions) (latent, logits *tensor.Tensor, coords [][4]int32, err error) {
	a := p.Flow.Args
	R := a.Resolution
	n, err := tensor.New(noise, []int{1, a.InChannels, R, R, R}, p.Dev)
	if err != nil {
		return nil, nil, nil, err
	}
	if latent, err = models.TrellisSample(p.Flow, n, cond, nil, o); err != nil {
		return nil, nil, nil, err
	}
	if logits, err = p.Decoder.Forward(latent); err != nil {
		return nil, nil, nil, err
	}
	coords, err = models.TrellisCoords(logits)
	return latent, logits, coords, err
}

// SampleSLat is sample_slat for one image's tokens cond (1, L, D) on the
// occupied voxels coords (stage 1's) from noise (N x 8 values, voxel-major):
// the voxel grid, the sampler's output and the structured latent
// un-normalised by slat_normalization, both (N, 8).
func (p *TrellisPipeline) SampleSLat(cond *tensor.Tensor, coords [][4]int32, noise []float32, o models.TrellisSampleOptions) (g *models.TrellisGrid, raw, slat *tensor.Tensor, err error) {
	if p.SLatFlow == nil {
		return nil, nil, nil, fmt.Errorf("trellis: stage 2 is not loaded")
	}
	C := p.SLatFlow.Args.InChannels
	if len(noise) != len(coords)*C {
		return nil, nil, nil, fmt.Errorf("trellis: %d SLat noise values for %d voxels x %d", len(noise), len(coords), C)
	}
	if g, err = models.NewTrellisGrid(coords, p.Dev); err != nil {
		return nil, nil, nil, err
	}
	n, err := tensor.New(noise, []int{len(coords), C}, p.Dev)
	if err != nil {
		return nil, nil, nil, err
	}
	if raw, err = models.TrellisSLatSample(p.SLatFlow, g, n, cond, nil, o); err != nil {
		return nil, nil, nil, err
	}
	slat, err = models.TrellisSLatUnnormalize(raw, p.SLatMean, p.SLatStd)
	return g, raw, slat, err
}

// DecodeGaussians is decode_slat's Gaussian decoder on the structured
// latent slat (N, 8) of the voxels of g: the per-voxel parameters (N, 14 x
// 32) and the activated Gaussians (to_representation).
func (p *TrellisPipeline) DecodeGaussians(g *models.TrellisGrid, slat *tensor.Tensor) (*tensor.Tensor, *splat.Gaussians, error) {
	if p.SLatDecoder == nil {
		return nil, nil, fmt.Errorf("trellis: stage 2 is not loaded")
	}
	feats, err := p.SLatDecoder.Forward(g, slat)
	if err != nil {
		return nil, nil, err
	}
	gs, err := p.SLatDecoder.Gaussians(g, feats)
	return feats, gs, err
}

// DecodeMesh is decode_slat's mesh decoder on the structured latent slat
// (N, 8) of the voxels of g: the per-cube outputs (64N, 101) and the mesh
// SparseFeatures2Mesh extracts (z up, inside [-0.5, 0.5]³, six colour
// channels per vertex with RGB first; mesh.Mesh.YUp turns it as to_glb).
func (p *TrellisPipeline) DecodeMesh(g *models.TrellisGrid, slat *tensor.Tensor) (*tensor.Tensor, *mesh.Mesh, error) {
	if p.MeshDecoder == nil {
		return nil, nil, fmt.Errorf("trellis: the mesh decoder is not loaded (TrellisLoadOptions.Mesh)")
	}
	feats, fine, err := p.MeshDecoder.Forward(g, slat)
	if err != nil {
		return nil, nil, err
	}
	m, err := p.MeshDecoder.Mesh(fine, feats)
	return feats, m, err
}

// TrellisOptions control one run.
type TrellisOptions struct {
	Image image.Image // an object with its background removed (alpha)
	// Preprocessed skips preprocess_image: Image is already the S x S
	// composited input.
	Preprocessed bool
	Seed         uint64
	// Noise, when set, is stage 1's initial latent (in x R³ values, e.g.
	// torch's randn recorded from TRELLIS) and SLatNoise stage 2's (N x 8
	// for N occupied voxels); otherwise each is drawn from Seed (Light's
	// generator, not torch's).
	Noise, SLatNoise []float32
	// Steps, CFG, Interval and RescaleT override pipeline.json's
	// sparse-structure sampler; SLatSteps and SLatCFG its slat_sampler.
	Steps     int
	CFG       float64
	Interval  [2]float64
	RescaleT  float64
	SLatSteps int
	SLatCFG   float64
	// Step and SLatStep, when set, are called after each step of the two
	// samplers.
	Step, SLatStep func(i, n int)
}

// TrellisSparseStructure is stage 1's output.
type TrellisSparseStructure struct {
	Coords [][4]int32   // occupied voxels (batch, x, y, z) of the 64³ grid
	Latent []float32    // the sampled occupancy latent (8 x 16³)
	Input  *image.NRGBA // the image the encoder saw
}

// TrellisResult is a full run's output.
type TrellisResult struct {
	TrellisSparseStructure
	SLat      []float32 // the structured latent (N x 8, un-normalised)
	Gaussians *splat.Gaussians
	Mesh      *mesh.Mesh // with the mesh decoder loaded (DecodeMesh)
}

// SparseStructure runs stage 1 on one image.
func (p *TrellisPipeline) SparseStructure(o TrellisOptions) (*TrellisSparseStructure, error) {
	r, err := p.run(o, false)
	if err != nil {
		return nil, err
	}
	return &r.TrellisSparseStructure, nil
}

// Generate runs both stages on one image: its 3D Gaussians (and its mesh
// when the mesh decoder is loaded).
func (p *TrellisPipeline) Generate(o TrellisOptions) (*TrellisResult, error) {
	return p.run(o, true)
}

func (p *TrellisPipeline) run(o TrellisOptions, stage2 bool) (*TrellisResult, error) {
	if stage2 && p.SLatFlow == nil {
		return nil, fmt.Errorf("trellis: stage 2 is not loaded (SparseStructureOnly)")
	}
	S := p.ImageSize()
	var pixels []float32
	var input *image.NRGBA
	var err error
	if o.Preprocessed {
		pixels, input, err = trellisPixels(o.Image, S)
	} else {
		var matte idolMatte
		if p.BiRefNet != nil {
			matte = p.BiRefNet.RemoveBackground
		}
		pixels, input, err = trellisPreprocess(o.Image, S, matte)
	}
	if err != nil {
		return nil, err
	}
	cond, err := p.Condition(pixels, S)
	if err != nil {
		return nil, err
	}
	rng := rand.New(rand.NewPCG(o.Seed, o.Seed^0x9e3779b97f4a7c15))
	draw := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	a := p.Flow.Args
	nv := a.InChannels * a.Resolution * a.Resolution * a.Resolution
	noise := o.Noise
	if noise == nil {
		noise = draw(nv)
	} else if len(noise) != nv {
		return nil, fmt.Errorf("trellis: %d noise values, want %d", len(noise), nv)
	}
	so := p.Sampler
	if o.Steps > 0 {
		so.Steps = o.Steps
	}
	if o.CFG != 0 {
		so.CFG = o.CFG
	}
	if o.Interval != [2]float64{} {
		so.Interval = o.Interval
	}
	if o.RescaleT > 0 {
		so.RescaleT = o.RescaleT
	}
	so.Step = o.Step
	lat, _, coords, err := p.SampleSparseStructure(cond, noise, so)
	if err != nil {
		return nil, err
	}
	lh, err := lat.To(device.CPU())
	if err != nil {
		return nil, err
	}
	res := &TrellisResult{TrellisSparseStructure: TrellisSparseStructure{Coords: coords, Latent: tensor.MustSlice[float32](lh), Input: input}}
	if !stage2 {
		return res, nil
	}
	if len(coords) == 0 {
		return nil, fmt.Errorf("trellis: the sparse structure is empty (no occupied voxel)")
	}
	sn := o.SLatNoise
	if sn == nil {
		sn = draw(len(coords) * p.SLatFlow.Args.InChannels)
	}
	lo := p.SLatSampler
	if o.SLatSteps > 0 {
		lo.Steps = o.SLatSteps
	}
	if o.SLatCFG != 0 {
		lo.CFG = o.SLatCFG
	}
	if o.Interval != [2]float64{} {
		lo.Interval = o.Interval
	}
	if o.RescaleT > 0 {
		lo.RescaleT = o.RescaleT
	}
	lo.Step = o.SLatStep
	g, _, slat, err := p.SampleSLat(cond, coords, sn, lo)
	if err != nil {
		return nil, err
	}
	if _, res.Gaussians, err = p.DecodeGaussians(g, slat); err != nil {
		return nil, err
	}
	if p.MeshDecoder != nil {
		if _, res.Mesh, err = p.DecodeMesh(g, slat); err != nil {
			return nil, err
		}
	}
	sh, err := slat.To(device.CPU())
	if err != nil {
		return nil, err
	}
	if sh, err = sh.Contiguous(); err != nil {
		return nil, err
	}
	res.SLat = tensor.MustSlice[float32](sh)
	return res, nil
}

// HasAlpha reports whether img has a real alpha channel (some pixel not
// fully opaque): such images need no background removal.
func HasAlpha(img image.Image) bool { return idolHasAlpha(img) }

// trellisMattePath resolves TrellisLoadOptions.Matte to a safetensors file.
func trellisMattePath(m string) (string, error) {
	if strings.HasSuffix(m, ".safetensors") {
		if _, err := os.Stat(m); err == nil {
			return m, nil
		}
	}
	var errs []string
	for _, name := range []string{"birefnet.safetensors", "model.safetensors"} {
		dir, err := fetchFiles(m, name)
		if err == nil {
			path := filepath.Join(dir, name)
			if _, err = os.Stat(path); err == nil {
				return path, nil
			}
		}
		errs = append(errs, err.Error())
	}
	return "", fmt.Errorf("trellis: background-removal weights %q: %s", m, strings.Join(errs, "; "))
}

// trellisPreprocess is TrellisImageTo3DPipeline.preprocess_image: an image
// without alpha is shrunk to at most 1024 on its long side (Lanczos) and
// cut out by matte (BiRefNet standing in for rembg's u2net; refused when
// nil); then crop a square of 1.2 x the bounding box of alpha > 0.8
// around its centre (outside the image is transparent), resize to size x
// size (Lanczos, as Pillow), premultiply by alpha and truncate to 8 bits.
// Returns the RGB planes (3, size, size) in [0, 1] and the image.
func trellisPreprocess(img image.Image, size int, matte idolMatte) ([]float32, *image.NRGBA, error) {
	p := idolPlanesOf(img)
	if !idolHasAlpha(img) {
		if matte == nil {
			return nil, nil, fmt.Errorf("trellis: the image has no transparent background and no background remover is loaded (TrellisLoadOptions.Matte, light trellis -matte); give an RGBA image with the object cut out")
		}
		if sc := 1024 / float64(max(p.w, p.h)); sc < 1 {
			p = p.resize(int(float64(p.w)*sc), int(float64(p.h)*sc))
		}
		n := p.w * p.h
		rgb := make([]float32, 3*n)
		for i := range n {
			for c, pl := range [3][]float64{p.r, p.g, p.b} {
				pl[i] = math.Round(math.Min(math.Max(pl[i], 0), 1)*255) / 255
				rgb[c*n+i] = float32(pl[i])
			}
		}
		a, err := matte(rgb, p.w, p.h)
		if err != nil {
			return nil, nil, fmt.Errorf("trellis: background removal: %w", err)
		}
		for i := range n {
			p.a[i] = float64(a[i]) / 255
		}
	}
	left, top, right, bottom := p.w, p.h, -1, -1
	for y := range p.h {
		for x := range p.w {
			if math.Round(p.a[y*p.w+x]*255) > 0.8*255 {
				left, right = min(left, x), max(right, x)
				top, bottom = min(top, y), max(bottom, y)
			}
		}
	}
	if right < 0 {
		return nil, nil, fmt.Errorf("trellis: no object found in the image (no alpha above 0.8)")
	}
	cx, cy := float64(left+right)/2, float64(top+bottom)/2
	side := int(float64(max(right-left, bottom-top)) * 1.2)
	half := float64(side / 2)
	// PIL's crop rounds the box (Python round: half to even).
	x0, y0 := int(math.RoundToEven(cx-half)), int(math.RoundToEven(cy-half))
	x1, y1 := int(math.RoundToEven(cx+half)), int(math.RoundToEven(cy+half))
	cw, ch := x1-x0, y1-y0
	crop := &idolPlanes{w: cw, h: ch, r: make([]float64, cw*ch), g: make([]float64, cw*ch), b: make([]float64, cw*ch), a: make([]float64, cw*ch)}
	for y := range ch {
		for x := range cw {
			sx, sy := x0+x, y0+y
			if sx < 0 || sy < 0 || sx >= p.w || sy >= p.h {
				continue
			}
			i, j := y*cw+x, sy*p.w+sx
			crop.r[i], crop.g[i], crop.b[i], crop.a[i] = p.r[j], p.g[j], p.b[j], p.a[j]
		}
	}
	crop = crop.resize(size, size)
	px := make([]float32, 3*size*size)
	out := image.NewNRGBA(image.Rect(0, 0, size, size))
	q8 := func(v float64) float32 { return float32(math.Round(math.Min(math.Max(v, 0), 1)*255)) / 255 }
	for i := range size * size {
		a := q8(crop.a[i])
		for c, pl := range [3][]float64{crop.r, crop.g, crop.b} {
			v := uint8(q8(pl[i]) * a * 255) // numpy astype(uint8) truncates
			px[c*size*size+i] = float32(v) / 255
			out.Pix[4*i+c] = v
		}
		out.Pix[4*i+3] = 255
	}
	return px, out, nil
}

// trellisPixels is encode_image's own handling of an already prepared
// image: resize to size x size if needed (Lanczos), drop alpha, [0, 1].
func trellisPixels(img image.Image, size int) ([]float32, *image.NRGBA, error) {
	p := idolPlanesOf(img)
	for i := range p.a { // convert('RGB') ignores alpha
		p.a[i] = 1
	}
	if p.w != size || p.h != size {
		p = p.resize(size, size)
	}
	px := make([]float32, 3*size*size)
	out := image.NewNRGBA(image.Rect(0, 0, size, size))
	for i := range size * size {
		for c, pl := range [3][]float64{p.r, p.g, p.b} {
			v := uint8(math.Round(math.Min(math.Max(pl[i], 0), 1) * 255))
			px[c*size*size+i] = float32(v) / 255
			out.Pix[4*i+c] = v
		}
		out.Pix[4*i+3] = 255
	}
	return px, out, nil
}
