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
	"light/ops"
	"light/tensor"
	"light/tokenizer"
)

// Flux3 is the step-distilled (flow-map) FLUX 3 Image pipeline (diffusers'
// Flux3 distilled modular pipeline): Qwen3-VL's stacked hidden states as
// the text context, the FLUX 3 DiT, Euler steps on a constant-shift
// schedule whose next sigma is the model's target timestep, and the FLUX.2
// VAE. Text-to-image; prompts should already be full scene captions (the
// reference pipeline runs a separate prompt upsampler first).
type Flux3 struct {
	TextEncoder *models.Llama // Qwen3-VL's text model
	Tok         tokenizer.Tokenizer
	Transformer *models.Flux3Transformer
	VAE         *models.FluxVAE
	Dev         device.Device

	vaeScale int
	encDev   device.Device
	inC      int
	dir      string
	opts     Flux3LoadOptions
}

// Flux3LoadOptions adjust LoadFlux3.
type Flux3LoadOptions struct {
	TokenizerJSON string        // overrides <repo>/text_encoder/tokenizer.json
	Storage16     bool          // BF16 weight storage for the text encoder and transformer
	EncoderDevice device.Device // the text encoder's device (nil: dev)
	Sequential    bool          // load the text encoder and transformer in turn (see Flux2LoadOptions)
}

// LoadFlux3 loads a FLUX 3 Image (Klein) repo or folder: transformer/,
// vae/ and text_encoder/ in diffusers layout.
func LoadFlux3(repo string, dev device.Device, o Flux3LoadOptions) (*Flux3, error) {
	dir, err := Snapshot(repo)
	if err != nil {
		return nil, err
	}
	sub := func(s string) string { return filepath.Join(dir, s) }
	raw, err := os.ReadFile(sub("transformer/config.json"))
	if err != nil {
		return nil, err
	}
	var tc struct {
		Class   string `json:"_class_name"`
		FlowMap bool   `json:"flow_map"`
		InCh    int    `json:"in_channels"`
	}
	if err := json.Unmarshal(raw, &tc); err != nil {
		return nil, err
	}
	if tc.Class != "Flux3Transformer2DModel" {
		return nil, fmt.Errorf("flux3: %s has a %s transformer, not Flux3Transformer2DModel", repo, tc.Class)
	}
	if !tc.FlowMap {
		return nil, fmt.Errorf("flux3: %s is not a flow-map (distilled) checkpoint; only FLUX 3 Image Klein is supported", repo)
	}
	p := &Flux3{Dev: dev, encDev: dev, dir: dir, opts: o, inC: tc.InCh}
	if o.EncoderDevice != nil {
		p.encDev = o.EncoderDevice
	}
	tj := o.TokenizerJSON
	if tj == "" {
		tj = sub("text_encoder/tokenizer.json")
	}
	if p.Tok, err = tokenizer.Load(tj); err != nil {
		return nil, fmt.Errorf("flux3 tokenizer: %w", err)
	}
	if p.VAE, err = models.LoadFluxVAE(sub("vae"), dev); err != nil {
		return nil, fmt.Errorf("flux3 vae: %w", err)
	}
	if p.VAE.BNMean == nil {
		return nil, fmt.Errorf("flux3 vae: no batch-norm statistics (not an AutoencoderKLFlux2)")
	}
	if raw, err = os.ReadFile(sub("vae/config.json")); err != nil {
		return nil, err
	}
	var vc struct {
		BlockOut []int `json:"block_out_channels"`
	}
	if err := json.Unmarshal(raw, &vc); err != nil {
		return nil, err
	}
	p.vaeScale = 1 << (len(vc.BlockOut) - 1)
	if !o.Sequential {
		if err := p.loadEncoder(); err != nil {
			return nil, err
		}
		if err := p.loadTransformer(); err != nil {
			return nil, err
		}
	}
	return p, nil
}

func (p *Flux3) load16(f func() error) error {
	if p.opts.Storage16 {
		return nn.WithWeightStorage16(dtype.BF16, f)
	}
	return f()
}

func (p *Flux3) loadEncoder() error {
	return p.load16(func() (err error) {
		if p.TextEncoder, err = models.LoadCausalLMFromHF(filepath.Join(p.dir, "text_encoder"), p.encDev, dtype.F32); err != nil {
			return fmt.Errorf("flux3 text_encoder: %w", err)
		}
		return nil
	})
}

func (p *Flux3) loadTransformer() error {
	return p.load16(func() (err error) {
		if p.Transformer, err = models.LoadFlux3Transformer(filepath.Join(p.dir, "transformer"), p.Dev); err != nil {
			return fmt.Errorf("flux3 transformer: %w", err)
		}
		return nil
	})
}

// Flux3Options control one generation.
type Flux3Options struct {
	Prompt        string // a full scene caption works best (no prompt upsampler here)
	Width, Height int    // pixels, default 1024 (multiples of 16)
	Steps         int    // default 16
	Shift         float64
	MaxSeqLen     int   // prompt tokens, default 8192
	Layers        []int // Qwen3-VL hidden states, default 4, 8, …, 32
	Seed          uint64
	Latents       *tensor.Tensor // (1, in_channels, H/16, W/16) initial noise, to match a reference run
	// Images are reference images for editing (up to 10). One is kept whole:
	// the output defaults to its padded canvas and is cropped back to it.
	Images []image.Image
}

// Flux3Result is a generated image and what produced it.
type Flux3Result struct {
	Image        *tensor.Tensor // (1, 3, H, W) in [0, 1], on the CPU
	PromptEmbeds *tensor.Tensor // (1, L, len(Layers)·D)
	IDs          []int64        // prompt tokens before padding
	Sigmas       []float32
}

// qwen3VLPrompt is Qwen3-VL-Instruct's chat template for one text-only user
// turn with the generation prompt.
func qwen3VLPrompt(s string) string {
	return "<|im_start|>user\n" + s + "<|im_end|>\n<|im_start|>assistant\n"
}

// flux3PadMultiple is the text window granularity: the prompt is
// right-padded to a multiple of this many tokens; the transformer attends to
// the padding, so it is part of the recipe.
const flux3PadMultiple = 80

// encode returns the stacked hidden states (1, L, len(layers)·D) of the
// padded prompt and the unpadded ids. Inside the encoder the padding is
// masked as keys and sits at position 1, as the reference passes it.
func (p *Flux3) encode(prompt string, maxLen int, layers []int) (*tensor.Tensor, []int64, error) {
	pad := p.Tok.TokenToID("<|endoftext|>")
	if pad < 0 {
		return nil, nil, fmt.Errorf("flux3: tokenizer has no <|endoftext|> pad token")
	}
	raw := p.Tok.Encode(qwen3VLPrompt(prompt))
	if len(raw) > maxLen {
		raw = raw[:maxLen]
	}
	L := min((len(raw)+flux3PadMultiple-1)/flux3PadMultiple*flux3PadMultiple, maxLen)
	all := make([]int64, L)
	ids := make([]int64, len(raw))
	for i := range all {
		all[i] = int64(pad)
		if i < len(raw) {
			all[i], ids[i] = int64(raw[i]), int64(raw[i])
		}
	}
	x, err := tensor.New(all, []int{1, L}, p.encDev)
	if err != nil {
		return nil, nil, err
	}
	hs, err := p.TextEncoder.HiddenStatesPadPos(x, layers, len(raw), 1)
	if err != nil {
		return nil, nil, err
	}
	e, err := ops.Concat(hs, 2)
	if err != nil {
		return nil, nil, err
	}
	e, err = e.To(p.Dev)
	return e, ids, err
}

// flux3Sigmas is Flux3SetTimestepsStep's schedule as FlowMatchEuler applies
// it: linspace(1, 1/n, n) in F32, shifted s·σ / (1 + (s − 1)·σ), then 0.
func flux3Sigmas(n int, shift float64) []float32 {
	out := make([]float32, n+1)
	for i := range n {
		s := float32(1.0 + (1.0/float64(n)-1.0)*float64(i)/math.Max(float64(n-1), 1))
		if n == 1 {
			s = 1
		}
		sh := float32(shift)
		out[i] = sh * s / (1 + (sh-1)*s)
	}
	return out
}

// Generate runs text-to-image.
func (p *Flux3) Generate(o Flux3Options) (*Flux3Result, error) {
	if o.Steps == 0 {
		o.Steps = 16
	}
	if o.Shift == 0 {
		o.Shift = 6.93
	}
	if o.MaxSeqLen == 0 {
		o.MaxSeqLen = 8192
	}
	if o.Layers == nil {
		o.Layers = []int{4, 8, 12, 16, 20, 24, 28, 32}
	}
	refs, refIDs, rw, rh, box, err := p.encodeReferences(o.Images)
	if err != nil {
		return nil, err
	}
	if o.Width != 0 || o.Height != 0 {
		box = nil // an explicit size is not cropped back
	}
	if o.Width == 0 {
		o.Width = 1024
		if rw > 0 {
			o.Width = rw
		}
	}
	if o.Height == 0 {
		o.Height = 1024
		if rh > 0 {
			o.Height = rh
		}
	}
	unit := p.vaeScale * 2
	if o.Width%unit != 0 || o.Height%unit != 0 {
		return nil, fmt.Errorf("flux3: width and height must be multiples of %d", unit)
	}
	if p.TextEncoder == nil {
		if err := p.loadEncoder(); err != nil {
			return nil, err
		}
	}
	embeds, ids, err := p.encode(o.Prompt, o.MaxSeqLen, o.Layers)
	if err != nil {
		return nil, err
	}
	if p.opts.Sequential {
		p.TextEncoder = nil
		releaseMemory()
	}
	if p.Transformer == nil {
		if err := p.loadTransformer(); err != nil {
			return nil, err
		}
	}
	St := embeds.Dim(1)
	txtIDs := make([]float32, St*4)
	for l := range St {
		txtIDs[l*4+3] = float32(l)
	}
	C4 := p.inC
	h, w := o.Height/unit, o.Width/unit
	lat := o.Latents
	if lat == nil {
		rng := rand.New(rand.NewPCG(o.Seed, o.Seed^0x9e3779b97f4a7c15))
		noise := make([]float32, C4*h*w)
		for i := range noise {
			noise[i] = float32(rng.NormFloat64())
		}
		if lat, err = tensor.New(noise, []int{1, C4, h, w}, p.Dev); err != nil {
			return nil, err
		}
	}
	if lat.Dim(1) != C4 || lat.Dim(2) != h || lat.Dim(3) != w {
		return nil, fmt.Errorf("flux3: latents %v, want (1, %d, %d, %d)", lat.Shape(), C4, h, w)
	}
	x, err := lat.Reshape(1, C4, h*w)
	if err != nil {
		return nil, err
	}
	if x, err = transposeTokens(x); err != nil {
		return nil, err
	}
	// Image ids (t, h, w, l): the h / w grid scaled by the image RoPE scale 0.5.
	imgIDs := make([]float32, h*w*4)
	for r := range h {
		for c := range w {
			i := (r*w + c) * 4
			imgIDs[i+1], imgIDs[i+2] = float32(r)*0.5, float32(c)*0.5
		}
	}
	sigmas := flux3Sigmas(o.Steps, o.Shift)
	// The model sees timestep / 1000 of the scheduler's timestep = σ · 1000 (F32).
	tm := func(s float32) float32 { return (s * 1000) / 1000 }
	for i := range o.Steps {
		v, err := p.Transformer.Forward(models.Flux3Inputs{Image: x, Text: embeds,
			Timestep: []float32{tm(sigmas[i])}, TargetTimestep: []float32{tm(sigmas[i+1])},
			ImgIDs: imgIDs, TxtIDs: txtIDs, Cond: refs, CondIDs: refIDs})
		if err != nil {
			return nil, fmt.Errorf("flux3 step %d: %w", i, err)
		}
		dt, _ := tensor.New([]float32{sigmas[i+1] - sigmas[i]}, []int{1}, p.Dev)
		if v, err = ops.Mul(v, dt); err != nil {
			return nil, err
		}
		if x, err = ops.Add(x, v); err != nil {
			return nil, err
		}
	}
	if p.opts.Sequential {
		p.Transformer = nil
		releaseMemory()
	}
	img, err := p.decode(x, C4, h, w)
	if err != nil {
		return nil, err
	}
	if box != nil { // crop back to the single reference's own box
		if img, err = img.Narrow(2, box[1], box[3]-box[1]); err != nil {
			return nil, err
		}
		if img, err = img.Narrow(3, box[0], box[2]-box[0]); err != nil {
			return nil, err
		}
		if img, err = img.Contiguous(); err != nil {
			return nil, err
		}
	}
	return &Flux3Result{Image: img, PromptEmbeds: embeds, IDs: ids, Sigmas: sigmas}, nil
}

// transposeTokens maps (1, C, S) to contiguous (1, S, C) and back.
func transposeTokens(x *tensor.Tensor) (*tensor.Tensor, error) {
	t, err := x.Permute(0, 2, 1)
	if err != nil {
		return nil, err
	}
	return t.Contiguous()
}

// decode unpacks the tokens (1, h·w, C4), undoes the batch norm,
// unpatchifies 2 x 2 and runs the VAE decoder; the image is in [0, 1].
func (p *Flux3) decode(x *tensor.Tensor, C4, h, w int) (*tensor.Tensor, error) {
	u, err := transposeTokens(x)
	if err != nil {
		return nil, err
	}
	if u, err = u.Reshape(1, C4, h, w); err != nil {
		return nil, err
	}
	std, err := tensor.New(p.VAE.BNStd, []int{1, C4, 1, 1}, p.Dev)
	if err != nil {
		return nil, err
	}
	mean, err := tensor.New(p.VAE.BNMean, []int{1, C4, 1, 1}, p.Dev)
	if err != nil {
		return nil, err
	}
	if u, err = ops.Mul(u, std); err != nil {
		return nil, err
	}
	if u, err = ops.Add(u, mean); err != nil {
		return nil, err
	}
	C := C4 / 4
	if u, err = u.Reshape(1, C, 2, 2, h, w); err != nil {
		return nil, err
	}
	if u, err = u.Permute(0, 1, 4, 2, 5, 3); err != nil {
		return nil, err
	}
	if u, err = u.Contiguous(); err != nil {
		return nil, err
	}
	if u, err = u.Reshape(1, C, 2*h, 2*w); err != nil {
		return nil, err
	}
	img, err := p.VAE.Decoder.Decode(u)
	if err != nil {
		return nil, err
	}
	return toUnitRange(img)
}

// flux3Grid is the reference image grid (Flux3ImageProcessor's
// vae_scale_factor: the pipeline builds it from its default config).
const flux3Grid = 16

// flux3Ref is one preprocessed reference: CHW pixels in [-1, 1].
type flux3Ref struct {
	px   []float32
	w, h int
}

// flux3PrepareRefs is Flux3ImageProcessor.preprocess: one reference is
// kept whole ("fill": downscaled only if its 16-grid canvas exceeds 4 MP,
// then reflect-padded to the canvas, with the box it occupies returned);
// several are each downscaled to at most 1 MP and center-cropped to the
// grid ("crop"). The downscales are Lanczos in diffusers and bicubic here,
// so a reference over those caps is not bit-exact.
func flux3PrepareRefs(imgs []image.Image) ([]flux3Ref, *[4]int) {
	toPx := func(rgb []uint8, w, h int) []float32 {
		px := make([]float32, 3*h*w)
		for c := range 3 {
			for i := range h * w {
				px[c*h*w+i] = float32(rgb[i*3+c])/255*2 - 1
			}
		}
		return px
	}
	canvas := func(side int) int { return min(4096, max(64, (side+flux3Grid-1)/flux3Grid*flux3Grid)) }
	var out []flux3Ref
	if len(imgs) == 1 {
		rgb, w, h := models.ImageRGB(imgs[0])
		scale := math.Min(1, math.Min(math.Sqrt(4*1024*1024/float64(w*h)), 4096/float64(max(w, h))))
		nw, nh := w, h
		for {
			nh, nw = max(1, int(math.Round(float64(h)*scale))), max(1, int(math.Round(float64(w)*scale)))
			if canvas(nh)*canvas(nw) <= 4*1024*1024 {
				break
			}
			scale *= 0.995
		}
		if nw != w || nh != h {
			rgb, w, h = models.ResizeRGB(rgb, w, h, nw, nh), nw, nh
		}
		cw, ch := canvas(w), canvas(h)
		left, top := (cw-w)/2, (ch-h)/2
		// numpy.pad: "reflect" excludes the edge pixel; "symmetric" (images
		// smaller than one grid cell) repeats it.
		symmetric := min(w, h) <= flux3Grid
		fold := func(i, n int) int {
			if n == 1 {
				return 0
			}
			for i < 0 || i >= n {
				if symmetric {
					if i < 0 {
						i = -i - 1
					} else {
						i = 2*n - 1 - i
					}
				} else if i < 0 {
					i = -i
				} else {
					i = 2*(n-1) - i
				}
			}
			return i
		}
		pad := make([]uint8, cw*ch*3)
		for y := range ch {
			sy := fold(y-top, h)
			for x := range cw {
				sx := fold(x-left, w)
				copy(pad[(y*cw+x)*3:(y*cw+x)*3+3], rgb[(sy*w+sx)*3:(sy*w+sx)*3+3])
			}
		}
		box := [4]int{left, top, left + w, top + h}
		return []flux3Ref{{toPx(pad, cw, ch), cw, ch}}, &box
	}
	for _, img := range imgs {
		rgb, w, h := models.ImageRGB(img)
		if w*h > 1024*1024 {
			s := math.Sqrt(1024 * 1024 / float64(w*h))
			nw, nh := int(float64(w)*s), int(float64(h)*s)
			rgb, w, h = models.ResizeRGB(rgb, w, h, nw, nh), nw, nh
		}
		cw, ch := w-w%flux3Grid, h-h%flux3Grid
		left, top := (w-cw)/2, (h-ch)/2
		crop := make([]uint8, cw*ch*3)
		for y := range ch {
			copy(crop[y*cw*3:(y+1)*cw*3], rgb[((y+top)*w+left)*3:((y+top)*w+left+cw)*3])
		}
		out = append(out, flux3Ref{toPx(crop, cw, ch), cw, ch})
	}
	return out, nil
}

// encodeReferences VAE-encodes the references to batch-normalised packed
// tokens (1, Σ tokens, C4) with ids (0, row·0.5, col·0.5, −(1+i)); it also
// returns the first reference's canvas size and, for one reference, the
// box of the image inside it.
func (p *Flux3) encodeReferences(imgs []image.Image) (*tensor.Tensor, []float32, int, int, *[4]int, error) {
	if len(imgs) == 0 {
		return nil, nil, 0, 0, nil, nil
	}
	if p.VAE.Encoder == nil {
		return nil, nil, 0, 0, nil, fmt.Errorf("flux3: the VAE has no encoder weights; editing needs them")
	}
	refs, box := flux3PrepareRefs(imgs)
	C4 := p.inC
	var toks []*tensor.Tensor
	var ids []float32
	for i, r := range refs {
		x, err := tensor.New(r.px, []int{1, 3, r.h, r.w}, p.Dev)
		if err != nil {
			return nil, nil, 0, 0, nil, err
		}
		z, err := p.VAE.Encoder.Mode(x)
		if err != nil {
			return nil, nil, 0, 0, nil, err
		}
		C, zh, zw := z.Dim(1), z.Dim(2), z.Dim(3)
		if z, err = z.Reshape(1, C, zh/2, 2, zw/2, 2); err != nil {
			return nil, nil, 0, 0, nil, err
		}
		if z, err = z.Permute(0, 1, 3, 5, 2, 4); err != nil {
			return nil, nil, 0, 0, nil, err
		}
		if z, err = z.Contiguous(); err != nil {
			return nil, nil, 0, 0, nil, err
		}
		ph, pw := zh/2, zw/2
		if z, err = z.Reshape(1, C4, ph, pw); err != nil {
			return nil, nil, 0, 0, nil, err
		}
		mean, _ := tensor.New(p.VAE.BNMean, []int{1, C4, 1, 1}, p.Dev)
		std, _ := tensor.New(p.VAE.BNStd, []int{1, C4, 1, 1}, p.Dev)
		if z, err = ops.Sub(z, mean); err != nil {
			return nil, nil, 0, 0, nil, err
		}
		if z, err = ops.Div(z, std); err != nil {
			return nil, nil, 0, 0, nil, err
		}
		if z, err = z.Reshape(1, C4, ph*pw); err != nil {
			return nil, nil, 0, 0, nil, err
		}
		if z, err = transposeTokens(z); err != nil {
			return nil, nil, 0, 0, nil, err
		}
		toks = append(toks, z)
		for r := range ph {
			for c := range pw {
				ids = append(ids, 0, float32(r)*0.5, float32(c)*0.5, float32(-(1 + i)))
			}
		}
	}
	all, err := ops.Concat(toks, 1)
	if err != nil {
		return nil, nil, 0, 0, nil, err
	}
	return all, ids, refs[0].w, refs[0].h, box, nil
}
