package diffusion

import (
	"encoding/json"
	"fmt"
	"image"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"

	"light/device"
	"light/dtype"
	"light/models"
	"light/nn"
	"light/tensor"
)

// TrellisPipeline is stage 1 of microsoft/TRELLIS-image-large's
// TrellisImageTo3DPipeline: an image of an object (alpha-cut) to its sparse
// structure, the occupied voxels of a 64³ grid. DINOv2 ViT-L/14 with
// registers encodes the image, the sparse-structure flow model samples the
// 16³ x 8 occupancy latent (25 Euler steps, CFG 5 for t in [0.5, 1]) and
// the decoder turns it into 64³ occupancy logits; coords = logits > 0.
// Stage 2 (the structured latent and its decoders) is not ported yet.
type TrellisPipeline struct {
	Flow    *models.TrellisFlow
	Decoder *models.TrellisSSDecoder
	DINO    *models.DINOv2
	Sampler models.TrellisSampleOptions // pipeline.json's sparse_structure_sampler.params
	Dev     device.Device
}

// TrellisLoadOptions adjust LoadTrellisPipeline.
type TrellisLoadOptions struct {
	// DINOv2 overrides the image encoder: a Hub repo or local folder with
	// transformers' Dinov2WithRegistersModel (config.json,
	// model.safetensors). Default: the HF port of pipeline.json's torch.hub
	// image_cond_model (dinov2_vitl14_reg -> facebook/dinov2-with-registers-large).
	DINOv2    string
	Storage16 bool // 16-bit linear weights (F16, as the checkpoints' torsos)
}

// trellisDINOv2Repos maps TRELLIS's torch.hub image_cond_model names to
// the transformers checkpoints holding the same weights.
var trellisDINOv2Repos = map[string]string{
	"dinov2_vits14_reg": "facebook/dinov2-with-registers-small",
	"dinov2_vitb14_reg": "facebook/dinov2-with-registers-base",
	"dinov2_vitl14_reg": "facebook/dinov2-with-registers-large",
	"dinov2_vitg14_reg": "facebook/dinov2-with-registers-giant",
}

// trellisPipelineConfig is the part of pipeline.json stage 1 uses.
type trellisPipelineConfig struct {
	Args struct {
		Models  map[string]string `json:"models"`
		Sampler struct {
			Name   string `json:"name"`
			Params struct {
				Steps       int        `json:"steps"`
				CFGStrength float64    `json:"cfg_strength"`
				CFGInterval [2]float64 `json:"cfg_interval"`
				RescaleT    float64    `json:"rescale_t"`
			} `json:"params"`
		} `json:"sparse_structure_sampler"`
		ImageCondModel string `json:"image_cond_model"`
	} `json:"args"`
}

// LoadTrellisPipeline loads TRELLIS stage 1 (a microsoft/TRELLIS-image-*
// repo or folder: pipeline.json and the sparse-structure flow model and
// decoder under ckpts/) and its DINOv2 image encoder onto dev.
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
	if a.Sampler.Name != "FlowEulerGuidanceIntervalSampler" {
		return nil, fmt.Errorf("trellis: sparse-structure sampler %q (want FlowEulerGuidanceIntervalSampler)", a.Sampler.Name)
	}
	flowName, decName := a.Models["sparse_structure_flow_model"], a.Models["sparse_structure_decoder"]
	if flowName == "" || decName == "" {
		return nil, fmt.Errorf("trellis: pipeline.json names no sparse-structure flow model / decoder")
	}
	if dir, err = fetchFiles(repo, flowName+".json", flowName+".safetensors", decName+".json", decName+".safetensors"); err != nil {
		return nil, err
	}
	load := func(f func() error) error {
		if o.Storage16 {
			return nn.WithWeightStorage16(dtype.F16, f)
		}
		return f()
	}
	sp := a.Sampler.Params
	p := &TrellisPipeline{Dev: dev, Sampler: models.TrellisSampleOptions{
		Steps: sp.Steps, CFG: sp.CFGStrength, Interval: sp.CFGInterval, RescaleT: sp.RescaleT}}
	if sp.CFGStrength == 0 {
		p.Sampler.CFG = -1 // explicit 0: no guidance
	}
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

// TrellisOptions control one stage-1 run.
type TrellisOptions struct {
	Image image.Image // an object with its background removed (alpha)
	// Preprocessed skips preprocess_image: Image is already the S x S
	// composited input.
	Preprocessed bool
	Seed         uint64
	// Noise, when set, is the initial latent (in x R³ values, e.g. torch's
	// randn recorded from TRELLIS); otherwise it is drawn from Seed (Light's
	// generator, not torch's).
	Noise []float32
	// Steps, CFG, Interval and RescaleT override pipeline.json's sampler.
	Steps    int
	CFG      float64
	Interval [2]float64
	RescaleT float64
	Step     func(i, n int)
}

// TrellisSparseStructure is stage 1's output.
type TrellisSparseStructure struct {
	Coords [][4]int32   // occupied voxels (batch, x, y, z) of the 64³ grid
	Latent []float32    // the sampled occupancy latent (8 x 16³)
	Input  *image.NRGBA // the image the encoder saw
}

// SparseStructure runs stage 1 on one image.
func (p *TrellisPipeline) SparseStructure(o TrellisOptions) (*TrellisSparseStructure, error) {
	S := p.ImageSize()
	var pixels []float32
	var input *image.NRGBA
	var err error
	if o.Preprocessed {
		pixels, input, err = trellisPixels(o.Image, S)
	} else {
		pixels, input, err = trellisPreprocess(o.Image, S)
	}
	if err != nil {
		return nil, err
	}
	cond, err := p.Condition(pixels, S)
	if err != nil {
		return nil, err
	}
	a := p.Flow.Args
	nv := a.InChannels * a.Resolution * a.Resolution * a.Resolution
	noise := o.Noise
	if noise == nil {
		rng := rand.New(rand.NewPCG(o.Seed, o.Seed^0x9e3779b97f4a7c15))
		noise = make([]float32, nv)
		for i := range noise {
			noise[i] = float32(rng.NormFloat64())
		}
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
	return &TrellisSparseStructure{Coords: coords, Latent: tensor.MustSlice[float32](lh), Input: input}, nil
}

// trellisPreprocess is TrellisImageTo3DPipeline.preprocess_image for an
// image with alpha: crop a square of 1.2 x the bounding box of alpha >
// 0.8 around its centre (outside the image is transparent), resize to size
// x size (Lanczos, as Pillow), premultiply by alpha and truncate to 8 bits.
// Returns the RGB planes (3, size, size) in [0, 1] and the image. Images
// without alpha need rembg upstream, which is not ported: they are refused.
func trellisPreprocess(img image.Image, size int) ([]float32, *image.NRGBA, error) {
	if !idolHasAlpha(img) {
		return nil, nil, fmt.Errorf("trellis: the image has no transparent background (TRELLIS removes it with rembg, which Light does not port); give an RGBA image with the object cut out")
	}
	p := idolPlanesOf(img)
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
