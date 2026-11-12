package models

import (
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"time"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
	"light/tokenizer"
)

// SDXL base: stabilityai/stable-diffusion-xl-base-1.0 through the same
// primitives as SD 1.5 (CLIP, UNet2DConditionModel, VAE, DDIM), with
//   - a three-level UNet (320 / 640 / 1280) whose transformer depth is
//     1 / 2 / 10 per level, 64-wide heads, linear projections and a
//     "text_time" addition embedding (pooled text ⊕ six embedded size /
//     crop ids),
//   - two text encoders: CLIP-L and OpenCLIP bigG, whose penultimate hidden
//     states concatenate into the 2048-wide cross-attention context, bigG's
//     projected pooled output feeding the addition embedding,
//   - the SDXL VAE (scaling factor 0.13025), run in F32 (its fp16 weights
//     overflow; diffusers' force_upcast).

const (
	SDXLRepo          = "stabilityai/stable-diffusion-xl-base-1.0"
	SDXLScalingFactor = 0.13025
	sdxlAddTimeDim    = 256 // addition_time_embed_dim
	sdxlNumTimeIDs    = 6   // original size (2), crop (2), target size (2)
)

// SDUNetSDXL returns the SDXL base UNet config
// (stabilityai/stable-diffusion-xl-base-1.0/unet/config.json).
func SDUNetSDXL() SDUNetConfig {
	return SDUNetConfig{
		InChannels: 4, OutChannels: 4,
		BlockOutChannels:  []int{320, 640, 1280},
		LayersPerBlock:    2,
		HeadDim:           64,
		CrossAttentionDim: 2048,
		TimeEmbDim:        1280,
		NormGroups:        32,
		CrossAttn:         []bool{false, true, true},
		TransformerDepth:  []int{1, 2, 10},
		AddEmbedDim:       1280 + sdxlNumTimeIDs*sdxlAddTimeDim, // 2816
	}
}

// StableDiffusionXL is the SDXL base text-to-image pipeline.
type StableDiffusionXL struct {
	TextEncoder1 *CLIPText // CLIP-L
	TextEncoder2 *CLIPText // OpenCLIP bigG, with projection
	Tokenizer1   *tokenizer.CLIPTokenizer
	Tokenizer2   *tokenizer.CLIPTokenizer
	UNet         *SDUNet
	VAE          *VAEDecoder
	Scheduler    *DDIMScheduler
}

// DefaultSDXLGenerateOpts returns SDXL's canonical defaults: 1024², 30 DDIM
// steps, guidance 5.
func DefaultSDXLGenerateOpts(prompt string) SDGenerateOpts {
	o := DefaultSDGenerateOpts(prompt)
	o.Width, o.Height = 1024, 1024
	o.NumInference = 30
	o.GuidanceScale = 5
	o.ScalingFactor = SDXLScalingFactor
	return o
}

// SDXLCondition is what one prompt contributes to a UNet forward: the
// (1, 77, 2048) cross-attention context and the (1, 2816) addition
// embedding (bigG pooled ⊕ time ids).
type SDXLCondition struct {
	Ctx    *tensor.Tensor
	AddEmb *tensor.Tensor
}

// Encode runs a prompt through both text encoders and builds the UNet
// conditioning for an image of the given size (SDXL's size / crop
// conditioning: original and target size = the output, crop (0, 0)).
func (s *StableDiffusionXL) Encode(prompt string, width, height int, dev device.Device) (*SDXLCondition, error) {
	ids1 := s.Tokenizer1.Encode(prompt, s.TextEncoder1.Cfg.MaxPosition)
	ids2 := s.Tokenizer2.EncodeWithPad(prompt, s.TextEncoder2.Cfg.MaxPosition, 0)
	t1, err := tokensToI64Tensor(ids1, dev)
	if err != nil {
		return nil, err
	}
	t2, err := tokensToI64Tensor(ids2, dev)
	if err != nil {
		return nil, err
	}
	h1, _, err := s.TextEncoder1.ForwardPenultimate(t1)
	if err != nil {
		return nil, fmt.Errorf("text encoder 1: %w", err)
	}
	h2, pooled, err := s.TextEncoder2.ForwardPenultimate(t2)
	if err != nil {
		return nil, fmt.Errorf("text encoder 2: %w", err)
	}
	ctx, err := ops.Concat([]*tensor.Tensor{h1, h2}, 2) // (1, 77, 768 + 1280)
	if err != nil {
		return nil, err
	}
	addEmb, err := SDXLAddEmbedding(pooled, width, height, 0, 0, width, height, dev)
	if err != nil {
		return nil, err
	}
	return &SDXLCondition{Ctx: ctx, AddEmb: addEmb}, nil
}

// SDXLAddEmbedding is pooled (B, 1280) ⊕ the six time ids (original h / w,
// crop top / left, target h / w), each sinusoidally embedded to 256 dims
// (flip_sin_to_cos, no shift: cos first) → (B, 2816).
func SDXLAddEmbedding(pooled *tensor.Tensor, origH, origW, cropTop, cropLeft, targetH, targetW int, dev device.Device) (*tensor.Tensor, error) {
	ids := []float32{float32(origH), float32(origW), float32(cropTop), float32(cropLeft), float32(targetH), float32(targetW)}
	te, err := TimestepEmbedding(ids, sdxlAddTimeDim, 10000, dev) // (6, 256)
	if err != nil {
		return nil, err
	}
	flat, err := te.Reshape(1, sdxlNumTimeIDs*sdxlAddTimeDim)
	if err != nil {
		return nil, err
	}
	B := pooled.Dim(0)
	if B > 1 {
		parts := make([]*tensor.Tensor, B)
		for i := range parts {
			parts[i] = flat
		}
		if flat, err = ops.Concat(parts, 0); err != nil {
			return nil, err
		}
	}
	if flat.DType() != pooled.DType() {
		if flat, err = ops.Cast(flat, pooled.DType()); err != nil {
			return nil, err
		}
	}
	return ops.Concat([]*tensor.Tensor{pooled, flat}, 1)
}

// Generate runs text-to-image. Returns an (H, W, 3) F32 image in [0, 1].
func (s *StableDiffusionXL) Generate(opts SDGenerateOpts, dev device.Device) (*tensor.Tensor, error) {
	if opts.Width%8 != 0 || opts.Height%8 != 0 {
		return nil, fmt.Errorf("SDXL.Generate: width/height must be multiples of 8 (got %dx%d)", opts.Width, opts.Height)
	}
	if opts.ScalingFactor == 0 {
		opts.ScalingFactor = SDXLScalingFactor
	}
	latH, latW := opts.Height/8, opts.Width/8
	timeEmbDim := s.UNet.Cfg.BlockOutChannels[0]
	unetDT := s.UNet.ConvIn.Weight.DType()
	phase := func(name string) {
		if opts.OnPhase != nil {
			opts.OnPhase(name)
		}
	}
	castTo := func(t *tensor.Tensor, dt dtype.DType) (*tensor.Tensor, error) {
		if t.DType() == dt {
			return t, nil
		}
		return ops.Cast(t, dt)
	}
	phase("encode prompt")
	cond, err := s.Encode(opts.Prompt, opts.Width, opts.Height, dev)
	if err != nil {
		return nil, err
	}
	uncond, err := s.Encode(opts.Negative, opts.Width, opts.Height, dev)
	if err != nil {
		return nil, err
	}
	for _, c := range []*SDXLCondition{cond, uncond} {
		if c.Ctx, err = castTo(c.Ctx, unetDT); err != nil {
			return nil, err
		}
		if c.AddEmb, err = castTo(c.AddEmb, unetDT); err != nil {
			return nil, err
		}
	}
	runtime.GC()

	phase("init noise")
	seed := opts.Seed
	if seed == 0 {
		seed = rand.Uint64()
	}
	rng := rand.New(rand.NewPCG(seed, seed^0x9E3779B97F4A7C15))
	noise := make([]float32, 4*latH*latW)
	for i := range noise {
		u1, u2 := rng.Float64(), rng.Float64()
		if u1 < 1e-30 {
			u1 = 1e-30
		}
		noise[i] = float32(math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2))
	}
	latent, err := tensor.New(noise, []int{1, 4, latH, latW}, dev)
	if err != nil {
		return nil, err
	}
	if s.Scheduler == nil || len(s.Scheduler.Timesteps) != opts.NumInference {
		s.Scheduler = NewDDIMScheduler(opts.NumInference)
	}

	phase("diffuse")
	for step := range opts.NumInference {
		stepStart := time.Now()
		t := s.Scheduler.Timesteps[step]
		temb, err := TimestepEmbedding([]float32{float32(t)}, timeEmbDim, 10000, dev)
		if err != nil {
			return nil, err
		}
		if temb, err = castTo(temb, unetDT); err != nil {
			return nil, err
		}
		xin := s.Scheduler.ScaleModelInput(latent, step)
		if xin, err = castTo(xin, unetDT); err != nil {
			return nil, err
		}
		epsUncond, err := s.UNet.ForwardAug(xin, temb, uncond.Ctx, uncond.AddEmb)
		if err != nil {
			return nil, fmt.Errorf("unet step %d (uncond): %w", step, err)
		}
		runtime.GC()
		epsCond, err := s.UNet.ForwardAug(xin, temb, cond.Ctx, cond.AddEmb)
		if err != nil {
			return nil, fmt.Errorf("unet step %d (cond): %w", step, err)
		}
		runtime.GC()
		if epsUncond, err = castTo(epsUncond, dtype.F32); err != nil {
			return nil, err
		}
		if epsCond, err = castTo(epsCond, dtype.F32); err != nil {
			return nil, err
		}
		guided, err := cfgGuide(epsUncond, epsCond, opts.GuidanceScale)
		if err != nil {
			return nil, err
		}
		if latent, err = s.Scheduler.Step(step, latent, guided); err != nil {
			return nil, err
		}
		if opts.OnStep != nil {
			opts.OnStep(step, opts.NumInference, t, time.Since(stepStart))
		}
	}

	phase("vae decode")
	runtime.GC()
	lat, err := castTo(latent, s.VAE.ConvIn.Weight.DType())
	if err != nil {
		return nil, err
	}
	decoded, err := s.VAE.DecodeWithScaling(lat, opts.ScalingFactor)
	if err != nil {
		return nil, err
	}
	return sdImageFromDecoded(decoded)
}
