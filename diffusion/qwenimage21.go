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

// QwenImage21 is diffusers' QwenImage21Pipeline (Qwen-Image 2.1), text to
// image without guidance (true_cfg_scale 1, the default): the Qwen3-VL text
// encoder's last-layer states before its final norm (system prompt
// dropped), the single-stream Qwen-Image 2.1 DiT, flow-matching Euler with
// a dynamic exponential shift and a terminal stretch, and the RGBA
// Qwen-Image 2.1 VAE.
type QwenImage21 struct {
	TextEncoder *models.Llama // Qwen3-VL's text model
	Tok         tokenizer.Tokenizer
	Transformer *models.QwenImage21Transformer
	VAE         *models.QwenImageVAE
	Scheduler   *FlowMatchEuler
	Dev         device.Device

	encDev device.Device
	dir    string
	opts   QwenImage21LoadOptions
}

// QwenImage21LoadOptions adjust LoadQwenImage21.
type QwenImage21LoadOptions struct {
	TokenizerJSON string        // overrides <repo>/processor/tokenizer.json
	Storage16     bool          // BF16 weight storage for the text encoder and transformer
	EncoderDevice device.Device // the text encoder's device (nil: dev)
	Sequential    bool          // load the text encoder and transformer in turn
}

// LoadQwenImage21 loads a QwenImage21Pipeline folder (or Hub repo).
func LoadQwenImage21(repo string, dev device.Device, o QwenImage21LoadOptions) (*QwenImage21, error) {
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
		Class        string    `json:"_class_name"`
		SampleSigmas []float64 `json:"sample_sigmas"`
	}
	if err := json.Unmarshal(raw, &idx); err != nil {
		return nil, err
	}
	if idx.Class != "QwenImage21Pipeline" {
		return nil, fmt.Errorf("qwen-image 2.1: %s is a %s, not a QwenImage21Pipeline", repo, idx.Class)
	}
	if len(idx.SampleSigmas) > 0 {
		return nil, fmt.Errorf("qwen-image 2.1: pipeline sample_sigmas are not supported")
	}
	p := &QwenImage21{Dev: dev, encDev: dev, dir: dir, opts: o}
	if o.EncoderDevice != nil {
		p.encDev = o.EncoderDevice
	}
	tj := o.TokenizerJSON
	if tj == "" {
		tj = sub("processor/tokenizer.json")
	}
	if p.Tok, err = tokenizer.Load(tj); err != nil {
		return nil, fmt.Errorf("qwen-image 2.1 tokenizer: %w", err)
	}
	if p.VAE, err = models.LoadQwenImageVAE(sub("vae"), dev); err != nil {
		return nil, fmt.Errorf("qwen-image 2.1 vae: %w", err)
	}
	if p.Scheduler, err = LoadFlowMatchEuler(sub("scheduler")); err != nil {
		return nil, err
	}
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

func (p *QwenImage21) load16(f func() error) error {
	if p.opts.Storage16 {
		return nn.WithWeightStorage16(dtype.BF16, f)
	}
	return f()
}

func (p *QwenImage21) loadEncoder() error {
	return p.load16(func() (err error) {
		if p.TextEncoder, err = models.LoadCausalLMFromHF(filepath.Join(p.dir, "text_encoder"), p.encDev, dtype.F32); err != nil {
			return fmt.Errorf("qwen-image 2.1 text_encoder: %w", err)
		}
		return nil
	})
}

func (p *QwenImage21) loadTransformer() error {
	return p.load16(func() (err error) {
		if p.Transformer, err = models.LoadQwenImage21Transformer(filepath.Join(p.dir, "transformer"), p.Dev); err != nil {
			return fmt.Errorf("qwen-image 2.1 transformer: %w", err)
		}
		return nil
	})
}

// QwenImage21Options control one generation.
type QwenImage21Options struct {
	Prompt        string
	Width, Height int // pixels, default 1024 (multiples of 32)
	Steps         int // default 40
	Seed          uint64
	Latents       *tensor.Tensor // (1, 64, H/16, W/16) initial noise, to match a reference run
}

// QwenImage21Result is a generated image and what produced it.
type QwenImage21Result struct {
	Image        *tensor.Tensor // (1, 4, H, W) RGBA in [0, 1], on the CPU
	PromptEmbeds *tensor.Tensor // (1, L, D)
	IDs          []int64        // the full templated prompt
	Sigmas       []float32
}

const qwenImage21System = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"

// encode returns the prompt embeddings: the last layer's states before the
// final norm, without the system-prompt tokens.
func (p *QwenImage21) encode(prompt string) (*tensor.Tensor, []int64, error) {
	if prompt == "" {
		prompt = " " // Qwen has no BOS: an empty prompt would leave nothing to read
	}
	raw := p.Tok.Encode(qwenImage21System + "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n")
	drop := len(p.Tok.Encode(qwenImage21System))
	if drop >= len(raw) {
		return nil, nil, fmt.Errorf("qwen-image 2.1: empty prompt after the system tokens")
	}
	ids := make([]int64, len(raw))
	for i, v := range raw {
		ids[i] = int64(v)
	}
	x, err := tensor.New(ids, []int{1, len(ids)}, p.encDev)
	if err != nil {
		return nil, nil, err
	}
	hs, err := p.TextEncoder.HiddenStatesPreNorm(x, []int{len(p.TextEncoder.Blocks)})
	if err != nil {
		return nil, nil, err
	}
	e, err := hs[0].Narrow(1, drop, len(ids)-drop)
	if err != nil {
		return nil, nil, err
	}
	if e, err = e.Contiguous(); err != nil {
		return nil, nil, err
	}
	e, err = e.To(p.Dev)
	return e, ids, err
}

// Generate runs text-to-image.
func (p *QwenImage21) Generate(o QwenImage21Options) (*QwenImage21Result, error) {
	if o.Steps == 0 {
		o.Steps = 40
	}
	if o.Width == 0 {
		o.Width = 1024
	}
	if o.Height == 0 {
		o.Height = 1024
	}
	unit := p.VAE.Scale()
	// One vision-language image slot stands for 2 x 2 latent tokens: the grid is even.
	if o.Width%(2*unit) != 0 || o.Height%(2*unit) != 0 {
		return nil, fmt.Errorf("qwen-image 2.1: width and height must be multiples of %d", 2*unit)
	}
	if p.TextEncoder == nil {
		if err := p.loadEncoder(); err != nil {
			return nil, err
		}
	}
	embeds, ids, err := p.encode(o.Prompt)
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
	C := p.Transformer.Cfg.InChannels
	h, w := o.Height/unit, o.Width/unit
	lat := o.Latents
	if lat == nil {
		rng := rand.New(rand.NewPCG(o.Seed, o.Seed^0x9e3779b97f4a7c15))
		noise := make([]float32, C*h*w)
		for i := range noise {
			noise[i] = float32(rng.NormFloat64())
		}
		if lat, err = tensor.New(noise, []int{1, C, h, w}, p.Dev); err != nil {
			return nil, err
		}
	}
	if lat.Dim(1) != C || lat.Dim(2) != h || lat.Dim(3) != w {
		return nil, fmt.Errorf("qwen-image 2.1: latents %v, want (1, %d, %d, %d)", lat.Shape(), C, h, w)
	}
	x, err := lat.Reshape(1, C, h*w)
	if err != nil {
		return nil, err
	}
	if x, err = transposeTokens(x); err != nil {
		return nil, err
	}
	sigmas := p.Scheduler.Sigmas(o.Steps, h*w)
	for i := range o.Steps {
		t := (sigmas[i] * 1000) / 1000
		v, err := p.Transformer.Forward(models.QwenImage21Inputs{Image: x, Text: embeds, TextLen: embeds.Dim(1),
			H: h, W: w, Timestep: t})
		if err != nil {
			return nil, fmt.Errorf("qwen-image 2.1 step %d: %w", i, err)
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
	// Unpack, de-normalise with the latents' mean / std, decode.
	z, err := transposeTokens(x)
	if err != nil {
		return nil, err
	}
	if z, err = z.Reshape(1, C, h, w); err != nil {
		return nil, err
	}
	std, err := tensor.New(p.VAE.Cfg.LatentsStd, []int{1, C, 1, 1}, p.Dev)
	if err != nil {
		return nil, err
	}
	mean, err := tensor.New(p.VAE.Cfg.LatentsMean, []int{1, C, 1, 1}, p.Dev)
	if err != nil {
		return nil, err
	}
	if z, err = ops.Mul(z, std); err != nil {
		return nil, err
	}
	if z, err = ops.Add(z, mean); err != nil {
		return nil, err
	}
	img, err := p.VAE.Decode(z)
	if err != nil {
		return nil, err
	}
	if img, err = toUnitRange(img); err != nil {
		return nil, err
	}
	return &QwenImage21Result{Image: img, PromptEmbeds: embeds, IDs: ids, Sigmas: sigmas}, nil
}
