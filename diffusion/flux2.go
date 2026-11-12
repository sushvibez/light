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
	"light/models"
	"light/nn"
	"light/ops"
	"light/tensor"
	"light/tokenizer"
)

// Flux2 is diffusers' Flux2KleinPipeline (FLUX.2 klein) and Flux2Pipeline
// (FLUX.2 [dev]): hidden states from three text-encoder layers as the text
// conditioning (klein: Qwen3; dev: Mistral-Small-3 with BFL's system
// message), the FLUX.2 transformer over 2 x 2-patchified latents,
// flow-matching Euler with an empirical time shift, guidance (klein:
// classifier-free unless step-distilled; dev: the guidance-distilled
// embedding), and the batch-normalised FLUX.2 VAE.
type Flux2 struct {
	TextEncoder *models.Llama // Qwen3 (klein) / Mistral (dev)
	Tok         tokenizer.Tokenizer
	Transformer *models.Flux2Transformer
	VAE         *models.FluxVAE
	Scheduler   *FlowMatchEuler
	Distilled   bool
	Dev         device.Device
	// Mistral is FLUX.2 [dev]'s Flux2Pipeline: the Mistral3 encoder, its
	// prompt format and an embedded guidance scale instead of CFG.
	Mistral bool

	vaeScale int
	encDev   device.Device // the text encoder's device
	inC      int           // the transformer's in_channels
	dir      string
	opts     Flux2LoadOptions
}

// Flux2LoadOptions adjust LoadFlux2.
type Flux2LoadOptions struct {
	// TokenizerJSON overrides <repo>/tokenizer/tokenizer.json.
	TokenizerJSON string
	// Storage16 keeps the text encoder's and transformer's matmul weights in
	// BF16 (F32 compute; exact for BF16 checkpoints): klein-4B then needs
	// ~16 GB instead of ~32.
	Storage16 bool
	// EncoderDevice runs the text encoder elsewhere (nil: dev). It runs
	// once per image, so the CPU frees the GPU for the transformer.
	EncoderDevice device.Device
	// Sequential keeps only the tokenizer, VAE and scheduler resident:
	// Generate loads the text encoder, encodes, frees it, then loads the
	// transformer, denoises and frees it before decoding. Peak memory is the
	// larger model instead of both, at the cost of loading per image.
	Sequential bool
}

// LoadFlux2 loads a Flux2KleinPipeline folder (or Hub repo) onto dev at F32.
func LoadFlux2(repo string, dev device.Device, o Flux2LoadOptions) (*Flux2, error) {
	dir, err := Snapshot(repo)
	if err != nil {
		return nil, err
	}
	sub := func(s string) string { return filepath.Join(dir, s) }
	raw, err := os.ReadFile(sub("model_index.json"))
	if err != nil {
		return nil, err
	}
	var idx struct {
		Class     string `json:"_class_name"`
		Distilled bool   `json:"is_distilled"`
	}
	if err := json.Unmarshal(raw, &idx); err != nil {
		return nil, err
	}
	if idx.Class != "Flux2KleinPipeline" && idx.Class != "Flux2Pipeline" {
		return nil, fmt.Errorf("flux2: %s is a %s, not a Flux2KleinPipeline / Flux2Pipeline", repo, idx.Class)
	}
	p := &Flux2{Distilled: idx.Distilled, Dev: dev, encDev: dev, dir: dir, opts: o, Mistral: idx.Class == "Flux2Pipeline"}
	if o.EncoderDevice != nil {
		p.encDev = o.EncoderDevice
	}
	if p.inC, err = transformerInChannels(sub("transformer")); err != nil {
		return nil, fmt.Errorf("flux2 transformer: %w", err)
	}
	if !o.Sequential {
		if err = p.loadEncoder(); err != nil {
			return nil, err
		}
	}
	tj := o.TokenizerJSON
	if tj == "" {
		tj = filepath.Join(sub("tokenizer"), "tokenizer.json")
	}
	if p.Tok, err = tokenizer.Load(tj); err != nil {
		return nil, fmt.Errorf("flux2 tokenizer: %w", err)
	}
	if !o.Sequential {
		if err = p.loadTransformer(); err != nil {
			return nil, err
		}
	}
	if p.VAE, err = models.LoadFluxVAE(sub("vae"), dev); err != nil {
		return nil, fmt.Errorf("flux2 vae: %w", err)
	}
	if p.VAE.BNMean == nil {
		return nil, fmt.Errorf("flux2 vae: no batch-norm statistics (not an AutoencoderKLFlux2)")
	}
	if p.Scheduler, err = LoadFlowMatchEuler(sub("scheduler")); err != nil {
		return nil, err
	}
	raw, err = os.ReadFile(filepath.Join(sub("vae"), "config.json"))
	if err != nil {
		return nil, err
	}
	var vc struct {
		BlockOut []int `json:"block_out_channels"`
	}
	if err := json.Unmarshal(raw, &vc); err != nil {
		return nil, err
	}
	p.vaeScale = 1 << (len(vc.BlockOut) - 1)
	return p, nil
}

func (p *Flux2) load16(f func() error) error {
	if p.opts.Storage16 {
		return nn.WithWeightStorage16(dtype.BF16, f)
	}
	return f()
}

func (p *Flux2) loadEncoder() (err error) {
	if err = p.load16(func() (err error) {
		p.TextEncoder, err = models.LoadCausalLMFromHF(filepath.Join(p.dir, "text_encoder"), p.encDev, dtype.F32)
		return
	}); err != nil {
		return fmt.Errorf("flux2 text_encoder: %w", err)
	}
	return nil
}

func (p *Flux2) loadTransformer() (err error) {
	if err = p.load16(func() (err error) {
		p.Transformer, err = models.LoadFlux2Transformer(filepath.Join(p.dir, "transformer"), p.Dev)
		return
	}); err != nil {
		return fmt.Errorf("flux2 transformer: %w", err)
	}
	return nil
}

// Flux2Options control one generation.
type Flux2Options struct {
	Prompt        string
	Width, Height int     // pixels, default 1024
	Steps         int     // default 4 (distilled klein) / 50
	Guidance      float32 // klein: classifier-free guidance (non-distilled); dev: the embedded guidance; default 4
	MaxSeqLen     int     // text tokens, default 512
	Layers        []int   // text encoder hidden states, default 9, 18, 27 (klein) / 10, 20, 30 (dev)
	Seed          uint64
	// Images are reference images for editing (multi-reference): each is
	// VAE-encoded and joins the sequence as extra image tokens. Width /
	// Height default to the first image's (cut to a multiple of 16).
	Images []image.Image
	// Latents, when set, is the initial noise (1, 4·C, H/2, W/2) before
	// packing (to match a reference run).
	Latents *tensor.Tensor
}

// Flux2Result is a generated image and what produced it.
type Flux2Result struct {
	Image        *tensor.Tensor // (1, 3, H, W) in [0, 1], on the CPU
	PromptEmbeds *tensor.Tensor
	IDs          []int64
	Sigmas       []float32
}

// qwen3Prompt is Qwen3's chat template for one user turn with thinking
// disabled (apply_chat_template(..., enable_thinking=False)).
func qwen3Prompt(s string) string {
	return "<|im_start|>user\n" + s + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
}

// flux2DevSystem is BFL's text-encoder system message for FLUX.2 [dev]
// (diffusers' SYSTEM_MESSAGE).
const flux2DevSystem = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation."

// mistralFlux2Prompt is Mistral-Small-3's chat template for the system
// message and one user turn, without a generation prompt (Flux2Pipeline's
// apply_chat_template(..., add_generation_prompt=False)). [IMG] markers are
// dropped from the prompt, as diffusers does.
func mistralFlux2Prompt(s string) string {
	s = strings.ReplaceAll(s, "[IMG]", "")
	return "<s>[SYSTEM_PROMPT]" + flux2DevSystem + "[/SYSTEM_PROMPT][INST]" + s + "[/INST]"
}

// encode returns the text conditioning (1, L, len(layers)·D) and the ids.
func (p *Flux2) encode(prompt string, L int, layers []int) (*tensor.Tensor, []int64, error) {
	padTok, text := "<|endoftext|>", qwen3Prompt(prompt)
	if p.Mistral {
		padTok, text = "<pad>", mistralFlux2Prompt(prompt)
	}
	pad := p.Tok.TokenToID(padTok)
	if pad < 0 {
		return nil, nil, fmt.Errorf("flux2: tokenizer has no %s pad token", padTok)
	}
	raw := p.Tok.Encode(text)
	if len(raw) > L {
		raw = raw[:L]
	}
	ids := make([]int64, L)
	for i := range ids {
		ids[i] = int64(pad)
		if i < len(raw) {
			ids[i] = int64(raw[i])
		}
	}
	x, err := tensor.New(ids, []int{1, L}, p.encDev)
	if err != nil {
		return nil, nil, err
	}
	hs, err := p.TextEncoder.HiddenStates(x, layers, len(raw))
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

// Generate runs the pipeline.
func (p *Flux2) Generate(o Flux2Options) (*Flux2Result, error) {
	if o.Steps == 0 {
		o.Steps = 50
		if p.Distilled {
			o.Steps = 4
		}
	}
	if o.Guidance == 0 {
		o.Guidance = 4
	}
	if o.MaxSeqLen == 0 {
		o.MaxSeqLen = 512
	}
	if o.Layers == nil {
		o.Layers = []int{9, 18, 27}
		if p.Mistral {
			o.Layers = []int{10, 20, 30}
		}
	}
	refs, refIDs, rw, rh, err := p.encodeReferences(o.Images)
	if err != nil {
		return nil, err
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
	cfg := !p.Distilled && !p.Mistral && o.Guidance > 1
	var guidance []float32 // dev: the guidance-distilled embedding
	if p.Mistral {
		guidance = []float32{o.Guidance}
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
	var neg *tensor.Tensor
	if cfg {
		if neg, _, err = p.encode("", o.MaxSeqLen, o.Layers); err != nil {
			return nil, err
		}
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
	txtIDs := make([]float32, o.MaxSeqLen*4)
	for l := range o.MaxSeqLen {
		txtIDs[l*4+3] = float32(l)
	}

	// Latents: (1, 4C, h, w) packed to (1, h·w, 4C) row-major.
	C4 := p.inC
	h := 2 * (o.Height / (p.vaeScale * 2)) / 2
	w := 2 * (o.Width / (p.vaeScale * 2)) / 2
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
		return nil, fmt.Errorf("flux2: latents %v, want (1, %d, %d, %d)", lat.Shape(), C4, h, w)
	}
	x, err := lat.Reshape(1, C4, h*w)
	if err != nil {
		return nil, err
	}
	if x, err = x.Permute(0, 2, 1); err != nil {
		return nil, err
	}
	if x, err = x.Contiguous(); err != nil {
		return nil, err
	}
	imgIDs := make([]float32, h*w*4)
	for r := range h {
		for c := range w {
			i := (r*w + c) * 4
			imgIDs[i+1], imgIDs[i+2] = float32(r), float32(c)
		}
	}
	sigmas := p.Scheduler.SigmasMu(o.Steps, flux2EmpiricalMu(h*w, o.Steps))
	scalar := func(v float32) (*tensor.Tensor, error) { return tensor.New([]float32{v}, []int{1}, p.Dev) }
	for i := range o.Steps {
		t := sigmas[i] * float32(p.Scheduler.NumTrainTimesteps)
		run := func(txt *tensor.Tensor) (*tensor.Tensor, error) {
			in, ids := x, imgIDs
			if refs != nil {
				var err error
				if in, err = ops.Concat([]*tensor.Tensor{x, refs}, 1); err != nil {
					return nil, err
				}
				ids = append(append([]float32(nil), imgIDs...), refIDs...)
			}
			v, err := p.Transformer.Forward(models.Flux2Inputs{Image: in, Text: txt, Timestep: []float32{t / 1000},
				Guidance: guidance, ImgIDs: ids, TxtIDs: txtIDs})
			if err != nil || refs == nil {
				return v, err
			}
			return v.Narrow(1, 0, x.Dim(1)) // the prediction for the generated tokens
		}
		v, err := run(embeds)
		if err != nil {
			return nil, fmt.Errorf("flux2 step %d: %w", i, err)
		}
		if cfg {
			u, err := run(neg)
			if err != nil {
				return nil, fmt.Errorf("flux2 step %d (negative): %w", i, err)
			}
			d, err := ops.Sub(v, u)
			if err != nil {
				return nil, err
			}
			g, _ := scalar(o.Guidance)
			if d, err = ops.Mul(d, g); err != nil {
				return nil, err
			}
			if v, err = ops.Add(u, d); err != nil {
				return nil, err
			}
		}
		dt, _ := scalar(sigmas[i+1] - sigmas[i])
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

	// Unpack, undo the batch norm, unpatchify 2 x 2, decode.
	u, err := x.Permute(0, 2, 1)
	if err != nil {
		return nil, err
	}
	if u, err = u.Contiguous(); err != nil {
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
	if img, err = toUnitRange(img); err != nil {
		return nil, err
	}
	return &Flux2Result{Image: img, PromptEmbeds: embeds, IDs: ids, Sigmas: sigmas}, nil
}

// encodeReferences turns reference images into packed, batch-normalised
// latent tokens (1, Σ tokens, 4·C) and their ids (t = 10·(i+1), row, col,
// 0); it also returns the first image's size (sides cut to a multiple of
// 2·vaeScale) for the output default.
func (p *Flux2) encodeReferences(imgs []image.Image) (*tensor.Tensor, []float32, int, int, error) {
	if len(imgs) == 0 {
		return nil, nil, 0, 0, nil
	}
	if p.VAE.Encoder == nil {
		return nil, nil, 0, 0, fmt.Errorf("flux2: the VAE has no encoder weights; editing needs them")
	}
	C4 := p.inC
	unit := p.vaeScale * 2
	var toks []*tensor.Tensor
	var ids []float32
	fw, fh := 0, 0
	for i, img := range imgs {
		px, w, h, err := flux2ImagePixels(img, unit)
		if err != nil {
			return nil, nil, 0, 0, err
		}
		if i == 0 {
			fw, fh = w, h
		}
		x, err := tensor.New(px, []int{1, 3, h, w}, p.Dev)
		if err != nil {
			return nil, nil, 0, 0, err
		}
		z, err := p.VAE.Encoder.Mode(x) // (1, C, h/s, w/s)
		if err != nil {
			return nil, nil, 0, 0, err
		}
		C, zh, zw := z.Dim(1), z.Dim(2), z.Dim(3)
		// Patchify 2 x 2 into channels: (1, C, zh/2, 2, zw/2, 2) → (1, C, 2, 2, zh/2, zw/2).
		if z, err = z.Reshape(1, C, zh/2, 2, zw/2, 2); err != nil {
			return nil, nil, 0, 0, err
		}
		if z, err = z.Permute(0, 1, 3, 5, 2, 4); err != nil {
			return nil, nil, 0, 0, err
		}
		if z, err = z.Contiguous(); err != nil {
			return nil, nil, 0, 0, err
		}
		ph, pw := zh/2, zw/2
		if z, err = z.Reshape(1, C4, ph, pw); err != nil {
			return nil, nil, 0, 0, err
		}
		mean, _ := tensor.New(p.VAE.BNMean, []int{1, C4, 1, 1}, p.Dev)
		std, _ := tensor.New(p.VAE.BNStd, []int{1, C4, 1, 1}, p.Dev)
		if z, err = ops.Sub(z, mean); err != nil {
			return nil, nil, 0, 0, err
		}
		if z, err = ops.Div(z, std); err != nil {
			return nil, nil, 0, 0, err
		}
		if z, err = z.Reshape(1, C4, ph*pw); err != nil {
			return nil, nil, 0, 0, err
		}
		if z, err = z.Permute(0, 2, 1); err != nil {
			return nil, nil, 0, 0, err
		}
		if z, err = z.Contiguous(); err != nil {
			return nil, nil, 0, 0, err
		}
		toks = append(toks, z)
		for r := range ph {
			for c := range pw {
				ids = append(ids, float32(10*(i+1)), float32(r), float32(c), 0)
			}
		}
	}
	all := toks[0]
	if len(toks) > 1 {
		var err error
		if all, err = ops.Concat(toks, 1); err != nil {
			return nil, nil, 0, 0, err
		}
	}
	return all, ids, fw, fh, nil
}

// flux2ImagePixels is the pipeline's reference-image preprocessing: images
// over 1024² pixels shrink to that area, sides are cut to a multiple of
// unit (resize to cover + centre crop), then (x / 255)·2 − 1 as (3, H, W).
// Exact when no resize is needed; resizes use Pillow bicubic where
// diffusers uses Lanczos.
func flux2ImagePixels(img image.Image, unit int) ([]float32, int, int, error) {
	rgb, w, h := models.ImageRGB(img)
	if w*h > 1024*1024 {
		s := math.Sqrt(float64(1024*1024) / float64(w*h))
		nw, nh := int(float64(w)*s), int(float64(h)*s)
		rgb, w, h = models.ResizeRGB(rgb, w, h, nw, nh), nw, nh
	}
	tw, th := (w/unit)*unit, (h/unit)*unit
	if tw == 0 || th == 0 {
		return nil, 0, 0, fmt.Errorf("flux2: reference image %dx%d smaller than %d px", w, h, unit)
	}
	if tw != w || th != h {
		// Resize so the image covers (tw, th), then crop the centre.
		sw, sh := tw, th
		if float64(tw)/float64(th) > float64(w)/float64(h) {
			sh = h * tw / w
		} else {
			sw = w * th / h
		}
		rgb = models.ResizeRGB(rgb, w, h, sw, sh)
		x0, y0 := (sw-tw)/2, (sh-th)/2
		crop := make([]uint8, tw*th*3)
		for y := range th {
			copy(crop[y*tw*3:(y+1)*tw*3], rgb[((y0+y)*sw+x0)*3:((y0+y)*sw+x0+tw)*3])
		}
		rgb, w, h = crop, tw, th
	}
	px := make([]float32, 3*h*w)
	for c := range 3 {
		for i := range h * w {
			px[c*h*w+i] = float32(rgb[i*3+c])/255*2 - 1
		}
	}
	return px, w, h, nil
}

// flux2EmpiricalMu is the pipeline's compute_empirical_mu.
func flux2EmpiricalMu(seqLen, steps int) float64 {
	const a1, b1 = 8.73809524e-05, 1.89833333
	const a2, b2 = 0.00016927, 0.45666666
	if seqLen > 4300 {
		return a2*float64(seqLen) + b2
	}
	m200 := a2*float64(seqLen) + b2
	m10 := a1*float64(seqLen) + b1
	a := (m200 - m10) / 190.0
	b := m200 - 200.0*a
	return a*float64(steps) + b
}

// toUnitRange maps a decoded image from [-1, 1] to [0, 1] on the host.
func toUnitRange(img *tensor.Tensor) (*tensor.Tensor, error) {
	cpu := device.CPU()
	img, err := img.To(cpu)
	if err != nil {
		return nil, err
	}
	half, _ := tensor.New([]float32{0.5}, []int{1}, cpu)
	two, _ := tensor.New([]float32{2}, []int{1}, cpu)
	if img, err = ops.Div(img, two); err != nil {
		return nil, err
	}
	if img, err = ops.Add(img, half); err != nil {
		return nil, err
	}
	return ops.Clamp(img, 0, 1)
}
