package models

import (
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"time"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
	"light/tokenizer"
)

// StableDiffusion bundles the four pieces of an SD 1.x text-to-image
// pipeline: CLIP text encoder, UNet noise predictor, VAE decoder, and the
// diffusion scheduler. The tokenizer lives outside (caller supplies it) so
// users can swap CLIP variants without rebuilding the model.
type StableDiffusion struct {
	TextEncoder *CLIPText
	UNet        *SDUNet
	VAE         *VAEDecoder
	Tokenizer   *tokenizer.CLIPTokenizer
	Scheduler   *DDIMScheduler
}

// SDGenerateOpts controls a single text-to-image run.
type SDGenerateOpts struct {
	Prompt        string
	Negative      string  // empty → ""
	Width         int     // pixels; must be a multiple of 8 (latent factor)
	Height        int     // pixels; multiple of 8
	NumInference  int     // sampler steps; 50 typical
	GuidanceScale float32 // CFG scale; 7.5 typical
	Seed          uint64  // 0 → nondeterministic
	ScalingFactor float32 // SD 1.x = 0.18215

	// OnPhase fires at coarse pipeline phases ("tokenize", "encode prompt",
	// "init noise", "diffuse", "vae decode") so callers can log progress
	// without scraping stdout. May be nil.
	OnPhase func(phase string)

	// OnStep fires after each diffusion step. `step` is 0-indexed; `total`
	// is opts.NumInference; `t` is the absolute scheduler timestep used;
	// `elapsed` is wall time spent on this step. May be nil.
	OnStep func(step, total, t int, elapsed time.Duration)
}

// DefaultSDGenerateOpts returns the canonical SD 1.5 inference defaults.
func DefaultSDGenerateOpts(prompt string) SDGenerateOpts {
	return SDGenerateOpts{
		Prompt:        prompt,
		Negative:      "",
		Width:         512,
		Height:        512,
		NumInference:  50,
		GuidanceScale: 7.5,
		Seed:          0,
		ScalingFactor: 0.18215,
	}
}

// Generate runs text-to-image. Returns an (H, W, 3) F32 image in [0, 1].
// Caller is responsible for converting to PNG / JPEG.
func (s *StableDiffusion) Generate(opts SDGenerateOpts, dev device.Device) (*tensor.Tensor, error) {
	if opts.Width%8 != 0 || opts.Height%8 != 0 {
		return nil, fmt.Errorf("SD.Generate: width/height must be multiples of 8 (got %dx%d)", opts.Width, opts.Height)
	}
	latH, latW := opts.Height/8, opts.Width/8
	timeEmbDim := s.UNet.Cfg.BlockOutChannels[0] // 320
	// UNet's storage dtype drives the activation flow. CLIP/scheduler stay
	// F32; we cast at the boundaries.
	unetDT := s.UNet.ConvIn.Weight.DType()
	phase := func(name string) {
		if opts.OnPhase != nil {
			opts.OnPhase(name)
		}
	}
	maybeCast := func(t *tensor.Tensor, target dtype.DType) (*tensor.Tensor, error) {
		if t.DType() == target {
			return t, nil
		}
		return ops.Cast(t, target)
	}

	// 1. Tokenize cond + uncond.
	phase("tokenize")
	condIDs := s.Tokenizer.Encode(opts.Prompt, s.TextEncoder.Cfg.MaxPosition)
	uncondIDs := s.Tokenizer.Encode(opts.Negative, s.TextEncoder.Cfg.MaxPosition)

	// 2. Encode through CLIP. Run uncond first then cond and concat along
	//    the batch dim → (2, 77, 768). The diffusion loop slices it back
	//    into the two B=1 ctx halves.
	phase("encode prompt")
	uncondTok, err := tokensToI64Tensor(uncondIDs, dev)
	if err != nil {
		return nil, err
	}
	condTok, err := tokensToI64Tensor(condIDs, dev)
	if err != nil {
		return nil, err
	}
	uncondCtx, err := s.TextEncoder.Forward(uncondTok)
	if err != nil {
		return nil, fmt.Errorf("text-encode uncond: %w", err)
	}
	condCtx, err := s.TextEncoder.Forward(condTok)
	if err != nil {
		return nil, fmt.Errorf("text-encode cond: %w", err)
	}
	ctx, err := concatBatch(uncondCtx, condCtx)
	if err != nil {
		return nil, err
	}
	// CLIP runs at F32; if the UNet is BF16 cast the context once so every
	// per-step UNet forward gets the right dtype (the cast is a single
	// (2, 77, 768) tensor — 230 KB — vs running it 50 × 2 = 100 times).
	if ctx, err = maybeCast(ctx, unetDT); err != nil {
		return nil, err
	}

	// 3. Initialise noise on CPU (so the seed is reproducible) then move.
	phase("init noise")
	rngSeed := opts.Seed
	if rngSeed == 0 {
		rngSeed = rand.Uint64()
	}
	rng := rand.New(rand.NewPCG(rngSeed, rngSeed^0x9E3779B97F4A7C15))
	noise := make([]float32, 1*4*latH*latW)
	for i := range noise {
		// Box-Muller for N(0, 1).
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

	// 4. Diffusion loop.
	//
	// We run uncond and cond as two separate B=1 forwards (rather than one
	// B=2 forward). The UNet's attention scores tensor is (B*heads, S, S);
	// at 64×64 input that's 256 MB per copy at B=1, so doubling B doubles
	// the attention tensor too. Two B=1 forwards halve peak GPU memory at
	// the cost of running the model twice per step. Worth it to fit on
	// 16 GB cards.
	phase("diffuse")
	for step := range opts.NumInference {
		stepStart := time.Now()
		t := s.Scheduler.Timesteps[step]
		// (1, 320) timestep embedding — same `t` for both forwards. Cast
		// to UNet's dtype if needed.
		temb, err := TimestepEmbedding([]float32{float32(t)}, timeEmbDim, 10000, dev)
		if err != nil {
			return nil, err
		}
		if temb, err = maybeCast(temb, unetDT); err != nil {
			return nil, err
		}
		uncondCtxSplit, err := ctx.Narrow(0, 0, 1)
		if err != nil {
			return nil, err
		}
		condCtxSplit, err := ctx.Narrow(0, 1, 1)
		if err != nil {
			return nil, err
		}
		// Latent stays at F32 across DDIM steps for stability — cast in
		// for UNet, cast back out.
		latentInDT, err := maybeCast(latent, unetDT)
		if err != nil {
			return nil, err
		}
		epsUncond, err := s.UNet.Forward(latentInDT, temb, uncondCtxSplit)
		if err != nil {
			return nil, fmt.Errorf("unet step %d (uncond): %w", step, err)
		}
		runtime.GC() // free uncond intermediates before the cond forward
		epsCond, err := s.UNet.Forward(latentInDT, temb, condCtxSplit)
		if err != nil {
			return nil, fmt.Errorf("unet step %d (cond): %w", step, err)
		}
		// Cast eps back to F32 for CFG + scheduler step.
		if epsUncond, err = maybeCast(epsUncond, dtype.F32); err != nil {
			return nil, err
		}
		if epsCond, err = maybeCast(epsCond, dtype.F32); err != nil {
			return nil, err
		}
		guided, err := cfgGuide(epsUncond, epsCond, opts.GuidanceScale)
		if err != nil {
			return nil, err
		}
		latent, err = s.Scheduler.Step(step, latent, guided)
		if err != nil {
			return nil, err
		}
		runtime.GC()
		if opts.OnStep != nil {
			opts.OnStep(step, opts.NumInference, t, time.Since(stepStart))
		}
	}

	// 5. Decode through VAE (1, 3, H, W) in [-1, 1] roughly.
	phase("vae decode")
	vaeDT := s.VAE.ConvIn.Weight.DType()
	latentForVAE, err := maybeCast(latent, vaeDT)
	if err != nil {
		return nil, err
	}
	decoded, err := s.VAE.DecodeWithScaling(latentForVAE, opts.ScalingFactor)
	if err != nil {
		return nil, fmt.Errorf("vae decode: %w", err)
	}
	return sdImageFromDecoded(decoded)
}

// sdImageFromDecoded clamps a VAE output (1, 3, H, W) in [-1, 1] to an
// (H, W, 3) F32 image in [0, 1] on the CPU.
func sdImageFromDecoded(decoded *tensor.Tensor) (*tensor.Tensor, error) {
	var err error
	// VAE may produce BF16; cast to F32 for the CPU pixel-pack loop below.
	if decoded.DType() != dtype.F32 {
		if decoded, err = ops.Cast(decoded, dtype.F32); err != nil {
			return nil, err
		}
	}
	// Clamp to [-1, 1] then rescale to [0, 1] and reshape (B, C, H, W) → (H, W, C).
	cpu, err := decoded.To(device.CPU())
	if err != nil {
		return nil, err
	}
	flat := tensor.MustSlice[float32](cpu)
	C, H, W := decoded.Dim(1), decoded.Dim(2), decoded.Dim(3)
	if C != 3 {
		return nil, fmt.Errorf("SD.Generate: VAE returned %d channels, expected 3", C)
	}
	out := make([]float32, H*W*3)
	for c := 0; c < 3; c++ {
		for y := 0; y < H; y++ {
			for x := 0; x < W; x++ {
				v := (flat[c*H*W+y*W+x] + 1) * 0.5
				if v < 0 {
					v = 0
				}
				if v > 1 {
					v = 1
				}
				out[(y*W+x)*3+c] = v
			}
		}
	}
	return tensor.New(out, []int{H, W, 3}, device.CPU())
}

// tokensToI64Tensor takes (T,) int32 and returns a (1, T) I64 tensor on dev.
func tokensToI64Tensor(ids []int32, dev device.Device) (*tensor.Tensor, error) {
	tokens := make([]int64, len(ids))
	for i, v := range ids {
		tokens[i] = int64(v)
	}
	return tensor.New(tokens, []int{1, len(ids)}, dev)
}

// concatBatch concatenates two same-shape rank-N tensors along batch dim 0.
// Used to stack CFG uncond + cond pairs.
func concatBatch(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	if a.Rank() != b.Rank() {
		return nil, fmt.Errorf("concatBatch: rank mismatch")
	}
	for i := 1; i < a.Rank(); i++ {
		if a.Dim(i) != b.Dim(i) {
			return nil, fmt.Errorf("concatBatch: dim %d mismatch (%d vs %d)", i, a.Dim(i), b.Dim(i))
		}
	}
	return ops.Concat([]*tensor.Tensor{a, b}, 0)
}

// cfgGuide blends `eps_uncond + scale * (eps_cond - eps_uncond)`.
func cfgGuide(epsUncond, epsCond *tensor.Tensor, scale float32) (*tensor.Tensor, error) {
	// guided = eps_uncond * (1 - scale) + eps_cond * scale.
	// Fold to two scalar muls + one add.
	uncondScale, err := tensor.New([]float32{1 - scale}, []int{1}, epsUncond.Device())
	if err != nil {
		return nil, err
	}
	condScale, err := tensor.New([]float32{scale}, []int{1}, epsCond.Device())
	if err != nil {
		return nil, err
	}
	a, err := autograd.Mul(epsUncond, uncondScale)
	if err != nil {
		return nil, err
	}
	b, err := autograd.Mul(epsCond, condScale)
	if err != nil {
		return nil, err
	}
	return autograd.Add(a, b)
}

// silence unused-imports placeholder.
var _ = dtype.F32
