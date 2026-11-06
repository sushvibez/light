package models

import (
	"fmt"
	"math"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// TimestepEmbedding computes the sinusoidal positional embedding used by
// diffusion U-Nets. Output shape is (B, dim) F32 on `dev`.
//
//	freqs[i] = 1 / (max_period ** (2i / dim))   for i ∈ [0, dim/2)
//	emb[b, i]            = cos(t[b] * freqs[i])
//	emb[b, dim/2 + i]    = sin(t[b] * freqs[i])
//
// Note: HuggingFace Diffusers swaps cos/sin order ("flip_sin_to_cos=True")
// so the first half is cos and the second half is sin. We follow that
// convention because the SD 1.5 weights expect it.
func TimestepEmbedding(timesteps []float32, dim int, maxPeriod float64, dev device.Device) (*tensor.Tensor, error) {
	if dim%2 != 0 {
		return nil, fmt.Errorf("TimestepEmbedding: dim (%d) must be even", dim)
	}
	half := dim / 2
	B := len(timesteps)
	data := make([]float32, B*dim)
	for b, t := range timesteps {
		for i := 0; i < half; i++ {
			freq := math.Exp(-math.Log(maxPeriod) * float64(i) / float64(half))
			theta := float64(t) * freq
			data[b*dim+i] = float32(math.Cos(theta))
			data[b*dim+half+i] = float32(math.Sin(theta))
		}
	}
	return tensor.New(data, []int{B, dim}, dev)
}

// DDIMScheduler implements the deterministic ε-prediction DDIM sampler used
// by Stable Diffusion 1.x. Given a noise prediction ε and the current
// noisy latent x_t, it produces x_{t-1}.
//
// Pre-computes the linear-β schedule's α_cumprod table and the discrete
// timestep grid up front; `Step` is allocation-light.
type DDIMScheduler struct {
	NumTrainTimesteps int       // 1000 for SD 1.5
	BetaStart         float64   // 0.00085
	BetaEnd           float64   // 0.012
	AlphasCumprod     []float64 // length NumTrainTimesteps
	Timesteps         []int     // descending list of `NumInferenceSteps` integers in [0, NumTrainTimesteps)
}

// NewDDIMScheduler builds a scheduler with the SD 1.5 default schedule
// ("scaled_linear" β-schedule between [BetaStart, BetaEnd], 1000 train
// timesteps). `numInference` controls the sampling grid (50 is typical;
// fewer = faster + lower quality).
func NewDDIMScheduler(numInference int) *DDIMScheduler {
	const (
		train     = 1000
		betaStart = 0.00085
		betaEnd   = 0.012
	)
	// "Scaled-linear" schedule: β = (linspace(sqrt(b0), sqrt(b1), N))²
	alphasCumprod := make([]float64, train)
	cum := 1.0
	sb0 := math.Sqrt(betaStart)
	sb1 := math.Sqrt(betaEnd)
	for i := 0; i < train; i++ {
		t := float64(i) / float64(train-1)
		s := sb0 + (sb1-sb0)*t
		beta := s * s
		alpha := 1 - beta
		cum *= alpha
		alphasCumprod[i] = cum
	}
	// Inference timesteps: evenly spaced descending. SD 1.5 uses HF
	// Diffusers' "leading" mode (the default), which produces
	// `(arange(N) * (train/N)).round()[::-1]` — i.e. the first inference
	// step uses train timestep `(N-1)*step`, NOT `(N-1)*step + (step-1)`.
	// An off-by-`step-1` here means we sample at slightly noisier timesteps
	// than the model expected, which compounds across the loop and shows
	// up as visibly degraded image quality.
	step := train / numInference
	timesteps := make([]int, numInference)
	for i := 0; i < numInference; i++ {
		timesteps[i] = (numInference - 1 - i) * step
	}
	return &DDIMScheduler{
		NumTrainTimesteps: train,
		BetaStart:         betaStart,
		BetaEnd:           betaEnd,
		AlphasCumprod:     alphasCumprod,
		Timesteps:         timesteps,
	}
}

// Step applies one DDIM update.
//
//	x_{t-1} = sqrt(α_prev) · x0_pred + sqrt(1-α_prev) · ε
//	x0_pred = (x_t - sqrt(1-α_t) · ε) / sqrt(α_t)
//
// `step` is the index into Timesteps (0 = first / noisiest). `xt` and `eps`
// must share shape and dtype. Returns x_{t-1} as a fresh tensor on the same
// device.
func (s *DDIMScheduler) Step(step int, xt, eps *tensor.Tensor) (*tensor.Tensor, error) {
	if step < 0 || step >= len(s.Timesteps) {
		return nil, fmt.Errorf("DDIMScheduler.Step: step %d out of range [0, %d)", step, len(s.Timesteps))
	}
	t := s.Timesteps[step]
	alphaT := s.AlphasCumprod[t]
	var alphaPrev float64 = 1
	if step+1 < len(s.Timesteps) {
		alphaPrev = s.AlphasCumprod[s.Timesteps[step+1]]
	}
	sqrtAlphaT := math.Sqrt(alphaT)
	sqrtOneMinusAlphaT := math.Sqrt(1 - alphaT)
	sqrtAlphaPrev := math.Sqrt(alphaPrev)
	sqrtOneMinusAlphaPrev := math.Sqrt(1 - alphaPrev)

	// Compute x0_pred = (xt - sqrtOneMinusAlphaT * eps) / sqrtAlphaT, then
	// next = sqrtAlphaPrev * x0_pred + sqrtOneMinusAlphaPrev * eps.
	// Equivalent: next = a * xt + b * eps with
	//   a = sqrtAlphaPrev / sqrtAlphaT
	//   b = sqrtOneMinusAlphaPrev - a * sqrtOneMinusAlphaT
	a := float32(sqrtAlphaPrev / sqrtAlphaT)
	b := float32(sqrtOneMinusAlphaPrev - (sqrtAlphaPrev/sqrtAlphaT)*sqrtOneMinusAlphaT)
	return ddimAxBy(xt, eps, a, b)
}

// ScaleModelInput pre-conditions x_t for the model (DDIM doesn't change the
// scale, but Euler schedulers do — kept on the type so callers can stay
// agnostic to the exact scheduler).
func (s *DDIMScheduler) ScaleModelInput(x *tensor.Tensor, step int) *tensor.Tensor {
	return x
}

// ddimAxBy returns a*xt + b*eps as a fresh tensor of x's dtype on x's device.
func ddimAxBy(xt, eps *tensor.Tensor, a, b float32) (*tensor.Tensor, error) {
	if xt.DType() != eps.DType() {
		return nil, fmt.Errorf("DDIM step: dtype mismatch (xt=%v, eps=%v)", xt.DType(), eps.DType())
	}
	if !xt.Shape().Equal(eps.Shape()) {
		return nil, fmt.Errorf("DDIM step: shape mismatch (%v vs %v)", xt.Shape(), eps.Shape())
	}
	if xt.DType() != dtype.F32 {
		return nil, fmt.Errorf("DDIM step: only F32 supported today (got %v)", xt.DType())
	}
	// Plain CPU loop — the latent is small (1*4*64*64 = 16K floats) and
	// keeping this simple avoids needing axpby/fused-mul-add in the kernel
	// surface yet. Promote to a CUDA kernel if it shows up in profiles.
	xtCPU, err := xt.To(device.CPU())
	if err != nil {
		return nil, err
	}
	epsCPU, err := eps.To(device.CPU())
	if err != nil {
		return nil, err
	}
	xs := tensor.MustSlice[float32](xtCPU)
	es := tensor.MustSlice[float32](epsCPU)
	out := make([]float32, len(xs))
	for i := range xs {
		out[i] = a*xs[i] + b*es[i]
	}
	return tensor.New(out, xt.Shape().Dims(), xt.Device())
}
