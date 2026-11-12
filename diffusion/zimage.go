package diffusion

import (
	"encoding/json"
	"fmt"
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

// ZImage is diffusers' ZImagePipeline (Z-Image, Z-Image-Turbo): the Qwen3
// text encoder's second-to-last hidden states of the real prompt tokens as
// the caption, the Z-Image S3-DiT, flow-matching Euler (the model predicts
// the negated velocity at time 1 − sigma) with optional classifier-free
// guidance, and the FLUX.1 VAE.
type ZImage struct {
	TextEncoder *models.Llama // Qwen3
	Tok         tokenizer.Tokenizer
	Transformer *models.ZImageTransformer
	VAE         *models.FluxVAE
	Scheduler   *FlowMatchEuler
	Dev         device.Device

	vaeScale int
	encDev   device.Device // the text encoder's device
	inC      int           // the transformer's in_channels
	dir      string
	opts     ZImageLoadOptions
}

// ZImageLoadOptions adjust LoadZImage.
type ZImageLoadOptions struct {
	TokenizerJSON string // overrides <repo>/tokenizer/tokenizer.json
	Storage16     bool   // BF16 weight storage for the text encoder and transformer
	// EncoderDevice runs the text encoder elsewhere (nil: dev). It runs
	// once per image, so the CPU frees the GPU for the transformer.
	EncoderDevice device.Device
	// Sequential keeps only the tokenizer, VAE and scheduler resident:
	// Generate loads the text encoder, encodes, frees it, then loads the
	// transformer, denoises and frees it before decoding. Peak memory is the
	// larger model instead of both, at the cost of loading per image.
	Sequential bool
}

// LoadZImage loads a ZImagePipeline folder (or Hub repo) onto dev at F32.
func LoadZImage(repo string, dev device.Device, o ZImageLoadOptions) (*ZImage, error) {
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
		Class string `json:"_class_name"`
	}
	if err := json.Unmarshal(raw, &idx); err != nil {
		return nil, err
	}
	if idx.Class != "ZImagePipeline" {
		return nil, fmt.Errorf("z-image: %s is a %s, not a ZImagePipeline", repo, idx.Class)
	}
	p := &ZImage{Dev: dev, encDev: dev, dir: dir, opts: o}
	if o.EncoderDevice != nil {
		p.encDev = o.EncoderDevice
	}
	if p.inC, err = transformerInChannels(sub("transformer")); err != nil {
		return nil, fmt.Errorf("z-image transformer: %w", err)
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
		return nil, fmt.Errorf("z-image tokenizer: %w", err)
	}
	if !o.Sequential {
		if err = p.loadTransformer(); err != nil {
			return nil, err
		}
	}
	if p.VAE, err = models.LoadFluxVAE(sub("vae"), dev); err != nil {
		return nil, fmt.Errorf("z-image vae: %w", err)
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

func (p *ZImage) load16(f func() error) error {
	if p.opts.Storage16 {
		return nn.WithWeightStorage16(dtype.BF16, f)
	}
	return f()
}

func (p *ZImage) loadEncoder() (err error) {
	if err = p.load16(func() (err error) {
		p.TextEncoder, err = models.LoadCausalLMFromHF(filepath.Join(p.dir, "text_encoder"), p.encDev, dtype.F32)
		return
	}); err != nil {
		return fmt.Errorf("z-image text_encoder: %w", err)
	}
	return nil
}

func (p *ZImage) loadTransformer() (err error) {
	if err = p.load16(func() (err error) {
		p.Transformer, err = models.LoadZImageTransformer(filepath.Join(p.dir, "transformer"), p.Dev)
		return
	}); err != nil {
		return fmt.Errorf("z-image transformer: %w", err)
	}
	return nil
}

// ZImageOptions control one generation.
type ZImageOptions struct {
	Prompt         string
	NegativePrompt string
	Width, Height  int     // pixels, default 1024
	Steps          int     // default 8 (Turbo)
	Guidance       float32 // classifier-free guidance; 0 (Turbo) disables it
	MaxSeqLen      int     // prompt tokens kept at most, default 512
	Seed           uint64
	Latents        *tensor.Tensor // (C, H/vae·2, W/vae·2) initial noise, to match a reference run
	// Img2Img, when its Image is set, starts from that image noised to the
	// schedule's strength point (ZImageImg2ImgPipeline); Width / Height then
	// default to the image's. Noise is the (C, h, w) noise (seeded when nil).
	Img2Img Img2Img
	Noise   *tensor.Tensor
}

// ZImageResult is a generated image and what produced it.
type ZImageResult struct {
	Image   *tensor.Tensor // (1, 3, H, W) in [0, 1], on the CPU
	Caption *tensor.Tensor // (L, D) text conditioning
	IDs     []int64
	Sigmas  []float32
}

// qwen3ThinkingPrompt is Qwen3's chat template for one user turn with
// thinking enabled (Z-Image's apply_chat_template(..., enable_thinking=True)).
func qwen3ThinkingPrompt(s string) string {
	return "<|im_start|>user\n" + s + "<|im_end|>\n<|im_start|>assistant\n"
}

// encode returns the caption (L, D): the hidden states after the
// second-to-last layer for the real tokens. Real tokens never attend to
// the padding the pipeline adds, so the unpadded run is exact.
func (p *ZImage) encode(prompt string, maxLen int) (*tensor.Tensor, []int64, error) {
	raw := p.Tok.Encode(qwen3ThinkingPrompt(prompt))
	if len(raw) > maxLen {
		raw = raw[:maxLen]
	}
	ids := make([]int64, len(raw))
	for i, v := range raw {
		ids[i] = int64(v)
	}
	x, err := tensor.New(ids, []int{1, len(ids)}, p.encDev)
	if err != nil {
		return nil, nil, err
	}
	hs, err := p.TextEncoder.HiddenStates(x, []int{len(p.TextEncoder.Blocks) - 1}, 0)
	if err != nil {
		return nil, nil, err
	}
	c, err := hs[0].Reshape(len(ids), hs[0].Dim(2))
	if err != nil {
		return nil, nil, err
	}
	c, err = c.To(p.Dev)
	return c, ids, err
}

// Generate runs the pipeline.
func (p *ZImage) Generate(o ZImageOptions) (*ZImageResult, error) {
	if o.Steps == 0 {
		o.Steps = 8
	}
	if o.MaxSeqLen == 0 {
		o.MaxSeqLen = 512
	}
	if o.Img2Img.Image != nil && o.Width == 0 && o.Height == 0 {
		o.Width, o.Height = defaultSize(o.Img2Img.Image, p.vaeScale*2)
	}
	if o.Width == 0 {
		o.Width = 1024
	}
	if o.Height == 0 {
		o.Height = 1024
	}
	pt := newPhaseTimer()
	if p.TextEncoder == nil {
		if err := p.loadEncoder(); err != nil {
			return nil, err
		}
		pt.mark("load encoder", nil)
	}
	cap, ids, err := p.encode(o.Prompt, o.MaxSeqLen)
	if err != nil {
		return nil, err
	}
	pt.mark("encode", cap)
	var neg *tensor.Tensor
	if o.Guidance > 0 {
		if neg, _, err = p.encode(o.NegativePrompt, o.MaxSeqLen); err != nil {
			return nil, err
		}
	}
	if p.opts.Sequential {
		p.TextEncoder = nil
		releaseMemory()
	}
	pt.mark("free encoder", nil)
	if p.Transformer == nil {
		if err := p.loadTransformer(); err != nil {
			return nil, err
		}
		pt.mark("load transformer", nil)
	}
	C := p.inC
	h := 2 * (o.Height / (p.vaeScale * 2))
	w := 2 * (o.Width / (p.vaeScale * 2))
	sigmas := p.Scheduler.Sigmas(o.Steps, (h/2)*(w/2))
	first := 0
	x := o.Latents
	if x == nil {
		if x = o.Noise; x == nil {
			rng := rand.New(rand.NewPCG(o.Seed, o.Seed^0x9e3779b97f4a7c15))
			noise := make([]float32, C*h*w)
			for i := range noise {
				noise[i] = float32(rng.NormFloat64())
			}
			if x, err = tensor.New(noise, []int{C, h, w}, p.Dev); err != nil {
				return nil, err
			}
		}
		if o.Img2Img.Image != nil {
			z, err := imageLatents(p.VAE, o.Img2Img.Image, w*p.vaeScale, h*p.vaeScale, p.Dev)
			if err != nil {
				return nil, err
			}
			if z, err = z.Reshape(C, h, w); err != nil {
				return nil, err
			}
			first = startStep(o.Steps, o.Img2Img.Strength)
			if first >= o.Steps {
				return nil, fmt.Errorf("z-image: strength %g leaves no steps", o.Img2Img.Strength)
			}
			if x, err = noised(z, x, sigmas[first]); err != nil {
				return nil, err
			}
		}
	}
	if x.Dim(0) != C || x.Dim(1) != h || x.Dim(2) != w {
		return nil, fmt.Errorf("z-image: latents %v, want (%d, %d, %d)", x.Shape(), C, h, w)
	}
	scalar := func(v float32) (*tensor.Tensor, error) { return tensor.New([]float32{v}, []int{1}, p.Dev) }
	for i := first; i < o.Steps; i++ {
		t := sigmas[i] * float32(p.Scheduler.NumTrainTimesteps)
		tm := (1000 - t) / 1000
		endProfile := stepProfile(i, p.Dev)
		v, err := p.Transformer.Forward(x, cap, tm)
		if err != nil {
			return nil, fmt.Errorf("z-image step %d: %w", i, err)
		}
		if neg != nil {
			u, err := p.Transformer.Forward(x, neg, tm)
			if err != nil {
				return nil, fmt.Errorf("z-image step %d (negative): %w", i, err)
			}
			d, err := ops.Sub(v, u)
			if err != nil {
				return nil, err
			}
			g, _ := scalar(o.Guidance)
			if d, err = ops.Mul(d, g); err != nil {
				return nil, err
			}
			if v, err = ops.Add(v, d); err != nil {
				return nil, err
			}
		}
		// The model predicts the negated velocity.
		dt, _ := scalar(sigmas[i] - sigmas[i+1])
		if v, err = ops.Mul(v, dt); err != nil {
			return nil, err
		}
		if x, err = ops.Add(x, v); err != nil {
			return nil, err
		}
		endProfile()
		pt.mark(fmt.Sprintf("step %d", i), x)
	}
	if p.opts.Sequential {
		p.Transformer = nil
		releaseMemory()
	}
	pt.mark("free transformer", nil)
	z, err := x.Reshape(1, C, h, w)
	if err != nil {
		return nil, err
	}
	sc, _ := scalar(p.VAE.ScalingFactor)
	if z, err = ops.Div(z, sc); err != nil {
		return nil, err
	}
	sh, _ := scalar(p.VAE.ShiftFactor)
	if z, err = ops.Add(z, sh); err != nil {
		return nil, err
	}
	img, err := p.VAE.Decoder.Decode(z)
	if err != nil {
		return nil, err
	}
	if img, err = toUnitRange(img); err != nil {
		return nil, err
	}
	pt.mark("vae decode", img)
	return &ZImageResult{Image: img, Caption: cap, IDs: ids, Sigmas: sigmas}, nil
}
