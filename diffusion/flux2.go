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
	"light/ops"
	"light/tensor"
	"light/tokenizer"
)

// Flux2 is diffusers' Flux2KleinPipeline (FLUX.2 klein): Qwen3 hidden
// states from three layers as the text conditioning, the FLUX.2
// transformer over 2 x 2-patchified latents, flow-matching Euler with an
// empirical time shift, classifier-free guidance unless the model is
// step-distilled, and the batch-normalised FLUX.2 VAE.
type Flux2 struct {
	TextEncoder *models.Llama // Qwen3
	Tok         tokenizer.Tokenizer
	Transformer *models.Flux2Transformer
	VAE         *models.FluxVAE
	Scheduler   *FlowMatchEuler
	Distilled   bool
	Dev         device.Device

	vaeScale int
}

// Flux2LoadOptions adjust LoadFlux2.
type Flux2LoadOptions struct {
	// TokenizerJSON overrides <repo>/tokenizer/tokenizer.json.
	TokenizerJSON string
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
	if idx.Class != "Flux2KleinPipeline" {
		return nil, fmt.Errorf("flux2: %s is a %s, not a Flux2KleinPipeline", repo, idx.Class)
	}
	p := &Flux2{Distilled: idx.Distilled, Dev: dev}
	if p.TextEncoder, err = models.LoadCausalLMFromHF(sub("text_encoder"), dev, dtype.F32); err != nil {
		return nil, fmt.Errorf("flux2 text_encoder: %w", err)
	}
	tj := o.TokenizerJSON
	if tj == "" {
		tj = filepath.Join(sub("tokenizer"), "tokenizer.json")
	}
	if p.Tok, err = tokenizer.Load(tj); err != nil {
		return nil, fmt.Errorf("flux2 tokenizer: %w", err)
	}
	if p.Transformer, err = models.LoadFlux2Transformer(sub("transformer"), dev); err != nil {
		return nil, fmt.Errorf("flux2 transformer: %w", err)
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

// Flux2Options control one generation.
type Flux2Options struct {
	Prompt        string
	Width, Height int     // pixels, default 1024
	Steps         int     // default 4 (distilled klein) / 50
	Guidance      float32 // classifier-free guidance for non-distilled models, default 4
	MaxSeqLen     int     // text tokens, default 512
	Layers        []int   // text encoder hidden states, default 9, 18, 27
	Seed          uint64
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

// encode returns the text conditioning (1, L, len(layers)·D) and the ids.
func (p *Flux2) encode(prompt string, L int, layers []int) (*tensor.Tensor, []int64, error) {
	pad := p.Tok.TokenToID("<|endoftext|>")
	if pad < 0 {
		return nil, nil, fmt.Errorf("flux2: tokenizer has no <|endoftext|> pad token")
	}
	raw := p.Tok.Encode(qwen3Prompt(prompt))
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
	x, err := tensor.New(ids, []int{1, L}, p.Dev)
	if err != nil {
		return nil, nil, err
	}
	hs, err := p.TextEncoder.HiddenStates(x, layers, len(raw))
	if err != nil {
		return nil, nil, err
	}
	e, err := ops.Concat(hs, 2)
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
	}
	if o.Width == 0 {
		o.Width = 1024
	}
	if o.Height == 0 {
		o.Height = 1024
	}
	cfg := !p.Distilled && o.Guidance > 1
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
	txtIDs := make([]float32, o.MaxSeqLen*4)
	for l := range o.MaxSeqLen {
		txtIDs[l*4+3] = float32(l)
	}

	// Latents: (1, 4C, h, w) packed to (1, h·w, 4C) row-major.
	C4 := p.Transformer.Cfg.InChannels
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
			return p.Transformer.Forward(models.Flux2Inputs{Image: x, Text: txt, Timestep: []float32{t / 1000},
				ImgIDs: imgIDs, TxtIDs: txtIDs})
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
