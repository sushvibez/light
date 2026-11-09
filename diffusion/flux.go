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

// Flux is diffusers' FluxPipeline: CLIP-L pooled and T5 text embeddings,
// the Flux transformer stepped by flow-matching Euler over packed 2 × 2
// latent patches, and the 16-channel VAE.
type Flux struct {
	CLIP        *models.CLIPText
	CLIPTok     *tokenizer.CLIPTokenizer
	T5          *models.T5Encoder
	T5Tok       tokenizer.Tokenizer
	Transformer *models.FluxTransformer
	VAE         *models.FluxVAE
	Scheduler   *FlowMatchEuler
	Dev         device.Device

	vaeScale int // 2^(len(block_out_channels) − 1): pixels per latent
}

// LoadFlux loads a FluxPipeline folder (or Hub repo) onto dev at F32.
func LoadFlux(repo string, dev device.Device) (*Flux, error) {
	dir, err := Snapshot(repo)
	if err != nil {
		return nil, err
	}
	p := &Flux{Dev: dev}
	sub := func(s string) string { return filepath.Join(dir, s) }
	if p.CLIP, err = models.LoadCLIPTextFromDir(sub("text_encoder"), dev); err != nil {
		return nil, fmt.Errorf("flux text_encoder: %w", err)
	}
	vocab, err := os.Open(filepath.Join(sub("tokenizer"), "vocab.json"))
	if err != nil {
		return nil, err
	}
	defer vocab.Close()
	merges, err := os.Open(filepath.Join(sub("tokenizer"), "merges.txt"))
	if err != nil {
		return nil, err
	}
	defer merges.Close()
	if p.CLIPTok, err = tokenizer.LoadCLIPTokenizer(vocab, merges); err != nil {
		return nil, fmt.Errorf("flux tokenizer: %w", err)
	}
	if p.T5, err = models.LoadT5EncoderFromHF(sub("text_encoder_2"), dev, dtype.F32); err != nil {
		return nil, fmt.Errorf("flux text_encoder_2: %w", err)
	}
	if p.T5Tok, err = tokenizer.Load(filepath.Join(sub("tokenizer_2"), "tokenizer.json")); err != nil {
		return nil, fmt.Errorf("flux tokenizer_2: %w", err)
	}
	if p.Transformer, err = models.LoadFluxTransformer(sub("transformer"), dev); err != nil {
		return nil, fmt.Errorf("flux transformer: %w", err)
	}
	if p.VAE, err = models.LoadFluxVAE(sub("vae"), dev); err != nil {
		return nil, fmt.Errorf("flux vae: %w", err)
	}
	if p.Scheduler, err = LoadFlowMatchEuler(sub("scheduler")); err != nil {
		return nil, err
	}
	raw, err := os.ReadFile(filepath.Join(sub("vae"), "config.json"))
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

// FluxOptions control one generation.
type FluxOptions struct {
	Prompt        string
	Width, Height int     // pixels (rounded down to a multiple of 2·vaeScale)
	Steps         int     // default 4 (schnell); 28–50 for dev
	Guidance      float32 // distilled guidance (models with guidance_embeds); default 3.5
	MaxSeqLen     int     // T5 tokens, default 512 (schnell uses 256)
	Seed          uint64  // initial noise (Light's generator, not torch's)
	// Latents, when set, is the initial packed noise (1, S_img, C·4) instead
	// of the seeded draw (used to match a reference run).
	Latents *tensor.Tensor
	// Img2Img, when its Image is set, starts from that image noised to the
	// schedule's strength point (FluxImg2ImgPipeline). Noise is then the
	// unpacked (1, C, h, w) noise (seeded when nil).
	Img2Img Img2Img
	Noise   *tensor.Tensor
}

// FluxResult is a generated image and what produced it.
type FluxResult struct {
	Image        *tensor.Tensor // (1, 3, H, W) in [0, 1], on the CPU
	PromptEmbeds *tensor.Tensor // T5 states (1, MaxSeqLen, D)
	Pooled       *tensor.Tensor // CLIP pooled (1, D)
	Sigmas       []float32
}

// t5IDs tokenizes like T5TokenizerFast(padding="max_length",
// truncation=True): content, </s>, then <pad> up to n.
func (p *Flux) t5IDs(text string, n int) ([]int64, error) {
	eos, pad := p.T5Tok.TokenToID("</s>"), p.T5Tok.TokenToID("<pad>")
	if eos < 0 || pad < 0 {
		return nil, fmt.Errorf("flux: T5 tokenizer lacks </s> / <pad>")
	}
	ids := p.T5Tok.Encode(text)
	if len(ids) > n-1 {
		ids = ids[:n-1]
	}
	out := make([]int64, n)
	for i := range out {
		switch {
		case i < len(ids):
			out[i] = int64(ids[i])
		case i == len(ids):
			out[i] = int64(eos)
		default:
			out[i] = int64(pad)
		}
	}
	return out, nil
}

func (p *Flux) scalar(v float32) (*tensor.Tensor, error) {
	return tensor.New([]float32{v}, []int{1}, p.Dev)
}

// Generate runs the pipeline.
func (p *Flux) Generate(o FluxOptions) (*FluxResult, error) {
	if o.Steps == 0 {
		o.Steps = 4
	}
	if o.Guidance == 0 {
		o.Guidance = 3.5
	}
	if o.MaxSeqLen == 0 {
		o.MaxSeqLen = 512
	}
	if o.Width == 0 {
		o.Width = 1024
	}
	if o.Height == 0 {
		o.Height = 1024
	}
	// Text: CLIP pooled + T5 states.
	cids := p.CLIPTok.Encode(o.Prompt, p.CLIP.Cfg.MaxPosition)
	c64 := make([]int64, len(cids))
	for i, v := range cids {
		c64[i] = int64(v)
	}
	ct, err := tensor.New(c64, []int{1, len(c64)}, p.Dev)
	if err != nil {
		return nil, err
	}
	ch, err := p.CLIP.Forward(ct)
	if err != nil {
		return nil, err
	}
	pooled, err := models.PooledCLIP(ch, [][]int32{cids})
	if err != nil {
		return nil, err
	}
	tids, err := p.t5IDs(o.Prompt, o.MaxSeqLen)
	if err != nil {
		return nil, err
	}
	tt, err := tensor.New(tids, []int{1, len(tids)}, p.Dev)
	if err != nil {
		return nil, err
	}
	embeds, err := p.T5.Forward(tt)
	if err != nil {
		return nil, err
	}

	// Latents: (1, C, h, w) noise packed into (1, h/2·w/2, 4C) 2 × 2 patches.
	cfg := p.Transformer.Cfg
	C := cfg.InChannels / 4
	h := 2 * (o.Height / (p.vaeScale * 2))
	w := 2 * (o.Width / (p.vaeScale * 2))
	S := (h / 2) * (w / 2)
	sigmas := p.Scheduler.Sigmas(o.Steps, S)
	first := 0
	x := o.Latents
	if x == nil {
		n := o.Noise
		if n == nil {
			rng := rand.New(rand.NewPCG(o.Seed, o.Seed^0x9e3779b97f4a7c15))
			noise := make([]float32, C*h*w)
			for i := range noise {
				noise[i] = float32(rng.NormFloat64())
			}
			if n, err = tensor.New(noise, []int{1, C, h, w}, p.Dev); err != nil {
				return nil, err
			}
		}
		if o.Img2Img.Image != nil {
			z, err := imageLatents(p.VAE, o.Img2Img.Image, w*p.vaeScale, h*p.vaeScale, p.Dev)
			if err != nil {
				return nil, err
			}
			first = startStep(o.Steps, o.Img2Img.Strength)
			if first >= o.Steps {
				return nil, fmt.Errorf("flux: strength %g leaves no steps", o.Img2Img.Strength)
			}
			if n, err = noised(z, n, sigmas[first]); err != nil {
				return nil, err
			}
		}
		if n, err = n.Reshape(1, C, h/2, 2, w/2, 2); err != nil {
			return nil, err
		}
		if n, err = n.Permute(0, 2, 4, 1, 3, 5); err != nil {
			return nil, err
		}
		if n, err = n.Contiguous(); err != nil {
			return nil, err
		}
		if x, err = n.Reshape(1, S, 4*C); err != nil {
			return nil, err
		}
	}
	if x.Dim(1) != S || x.Dim(2) != 4*C {
		return nil, fmt.Errorf("flux: latents %v, want (1, %d, %d)", x.Shape(), S, 4*C)
	}
	imgIDs := make([]float32, S*3)
	for r := range h / 2 {
		for c := range w / 2 {
			i := (r*(w/2) + c) * 3
			imgIDs[i+1], imgIDs[i+2] = float32(r), float32(c)
		}
	}
	txtIDs := make([]float32, o.MaxSeqLen*3)

	var guidance []float32
	if cfg.GuidanceEmbeds {
		guidance = []float32{o.Guidance}
	}
	for i := first; i < o.Steps; i++ {
		t := sigmas[i] * float32(p.Scheduler.NumTrainTimesteps) // scheduler timesteps
		v, err := p.Transformer.Forward(models.FluxInputs{
			Image: x, Text: embeds, Pooled: pooled,
			Timestep: []float32{t / 1000}, Guidance: guidance,
			ImgIDs: imgIDs, TxtIDs: txtIDs,
		})
		if err != nil {
			return nil, fmt.Errorf("flux step %d: %w", i, err)
		}
		dt, err := p.scalar(sigmas[i+1] - sigmas[i])
		if err != nil {
			return nil, err
		}
		if v, err = ops.Mul(v, dt); err != nil {
			return nil, err
		}
		if x, err = ops.Add(x, v); err != nil {
			return nil, err
		}
	}

	// Unpack to (1, C, h, w), undo the latent normalisation, decode.
	u, err := x.Reshape(1, h/2, w/2, C, 2, 2)
	if err != nil {
		return nil, err
	}
	if u, err = u.Permute(0, 3, 1, 4, 2, 5); err != nil {
		return nil, err
	}
	if u, err = u.Contiguous(); err != nil {
		return nil, err
	}
	if u, err = u.Reshape(1, C, h, w); err != nil {
		return nil, err
	}
	sc, err := p.scalar(p.VAE.ScalingFactor)
	if err != nil {
		return nil, err
	}
	if u, err = ops.Div(u, sc); err != nil {
		return nil, err
	}
	sh, err := p.scalar(p.VAE.ShiftFactor)
	if err != nil {
		return nil, err
	}
	if u, err = ops.Add(u, sh); err != nil {
		return nil, err
	}
	img, err := p.VAE.Decoder.Decode(u)
	if err != nil {
		return nil, err
	}
	// [-1, 1] → [0, 1], on the host (the image is read there anyway).
	cpu := device.CPU()
	if img, err = img.To(cpu); err != nil {
		return nil, err
	}
	half, err := tensor.New([]float32{0.5}, []int{1}, cpu)
	if err != nil {
		return nil, err
	}
	two, err := tensor.New([]float32{2}, []int{1}, cpu)
	if err != nil {
		return nil, err
	}
	if img, err = ops.Div(img, two); err != nil {
		return nil, err
	}
	if img, err = ops.Add(img, half); err != nil {
		return nil, err
	}
	if img, err = ops.Clamp(img, 0, 1); err != nil {
		return nil, err
	}
	return &FluxResult{Image: img, PromptEmbeds: embeds, Pooled: pooled, Sigmas: sigmas}, nil
}
