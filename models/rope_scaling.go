package models

import (
	"fmt"
	"math"

	"light/device"
	"light/tensor"
)

// RopeScaling describes a frequency-scaling scheme applied to the RoPE
// inverse-frequency table, i.e. HuggingFace's `rope_scaling` config block.
// Supported: "llama3" (Llama 3.1 / 3.2), "longrope" (Phi-3 / 3.5) and
// "yarn" (Mellum 2.1 full-attention layers, Qwen long-context configs).
type RopeScaling struct {
	Type                 string  // "llama3", "longrope" or "yarn"
	Factor               float64 // llama3: e.g. 32 for Llama 3.2, 8 for Llama 3.1; yarn: context extension factor
	LowFreqFactor        float64 // llama3: e.g. 1
	HighFreqFactor       float64 // llama3: e.g. 4
	OriginalMaxPositions int     // context the base table was trained for (8192 Llama-3, 4096 Phi-3)

	// longrope: per-frequency divisors (len headDim/2); short is used while
	// the sequence fits OriginalMaxPositions, long beyond it. MaxPositions is
	// config max_position_embeddings; it sets the attention factor (and the
	// yarn factor when Factor is 0, as DeepSeek-V3 configs do).
	ShortFactor  []float64
	LongFactor   []float64
	MaxPositions int

	// yarn: ramp bounds in rotations (HF defaults 32 / 1 when 0), an
	// explicit cos/sin multiplier (0 = derive from Factor, MScale and
	// MScaleAllDim), and NoTruncate for configs with truncate=false.
	BetaFast, BetaSlow   float64
	AttnFactor           float64
	MScale, MScaleAllDim float64
	NoTruncate           bool
}

// yarnFactor is the yarn context-extension factor (Factor, or
// max/original positions when the config leaves it unset).
func (s RopeScaling) yarnFactor() float64 {
	if s.Factor > 0 {
		return s.Factor
	}
	if s.OriginalMaxPositions > 0 && s.MaxPositions > 0 {
		return float64(s.MaxPositions) / float64(s.OriginalMaxPositions)
	}
	return 1
}

// YarnInvFreq is transformers' `_compute_yarn_parameters` table: per pair i,
// a blend of the original frequency (extrapolation, kept for the
// high-frequency pairs that rotate more than BetaFast times over the
// original context) and the frequency divided by the factor (interpolation,
// for the pairs rotating fewer than BetaSlow times), with a linear ramp in
// between.
func YarnInvFreq(headDim int, base float64, s RopeScaling) []float32 {
	factor := s.yarnFactor()
	betaFast, betaSlow := s.BetaFast, s.BetaSlow
	if betaFast == 0 {
		betaFast = 32
	}
	if betaSlow == 0 {
		betaSlow = 1
	}
	dim := float64(headDim)
	orig := float64(s.OriginalMaxPositions)
	correctionDim := func(rotations float64) float64 {
		return dim * math.Log(orig/(rotations*2*math.Pi)) / (2 * math.Log(base))
	}
	low, high := correctionDim(betaFast), correctionDim(betaSlow)
	if !s.NoTruncate {
		low, high = math.Floor(low), math.Ceil(high)
	}
	low, high = math.Max(low, 0), math.Min(high, dim-1)
	if low == high {
		high += 0.001 // HF: prevent a zero-width ramp
	}
	half := headDim / 2
	out := make([]float32, half)
	for i := range half {
		posFreq := math.Pow(base, float64(2*i)/dim)
		ramp := math.Min(math.Max((float64(i)-low)/(high-low), 0), 1)
		extrapolation := 1 - ramp // weight of the unscaled frequency
		out[i] = float32((1/(factor*posFreq))*(1-extrapolation) + (1/posFreq)*extrapolation)
	}
	return out
}

// yarnMScale is HF's get_mscale: 0.1·m·ln(scale) + 1 for scale > 1, else 1.
func yarnMScale(scale, m float64) float64 {
	if scale <= 1 {
		return 1
	}
	return 0.1*m*math.Log(scale) + 1
}

// LongRopeInvFreq divides the base table element-wise by factors (Phi-3
// `_compute_longrope_parameters`): invFreq[i] = base^(-2i/headDim) / factors[i].
func LongRopeInvFreq(headDim int, base float64, factors []float64) []float32 {
	inv := BaseInvFreq(headDim, base)
	for i := range inv {
		inv[i] = float32(float64(inv[i]) / factors[i])
	}
	return inv
}

// AttentionFactor is the multiplier the scheme applies to cos/sin (and hence
// to the rotated q and k). LongRoPE: sqrt(1 + ln(max/orig) / ln(orig)) when
// the context was extended. YaRN: the configured attention_factor, else
// get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim) when both
// are set, else get_mscale(factor). 1 for every other scheme.
func (s RopeScaling) AttentionFactor() float64 {
	if s.Type == "yarn" {
		if s.AttnFactor > 0 {
			return s.AttnFactor
		}
		f := s.yarnFactor()
		if s.MScale != 0 && s.MScaleAllDim != 0 {
			return yarnMScale(f, s.MScale) / yarnMScale(f, s.MScaleAllDim)
		}
		return yarnMScale(f, 1)
	}
	if s.Type != "longrope" || s.OriginalMaxPositions <= 0 || s.MaxPositions <= s.OriginalMaxPositions {
		return 1
	}
	scale := float64(s.MaxPositions) / float64(s.OriginalMaxPositions)
	return math.Sqrt(1 + math.Log(scale)/math.Log(float64(s.OriginalMaxPositions)))
}

// BaseInvFreq returns the unscaled RoPE table: invFreq[i] = base^(-2i/headDim)
// for i in [0, headDim/2).
func BaseInvFreq(headDim int, base float64) []float32 {
	half := headDim / 2
	out := make([]float32, half)
	for i := range half {
		out[i] = float32(math.Pow(base, -float64(2*i)/float64(headDim)))
	}
	return out
}

// Llama3InvFreq applies Meta's Llama-3 frequency scaling to the base table.
// Mirrors transformers' `_compute_llama3_parameters`:
//
//	wavelen = 2π / invFreq
//	wavelen > origCtx/lowFreqFactor  → invFreq / factor           (low freqs)
//	wavelen < origCtx/highFreqFactor → invFreq                    (high freqs)
//	otherwise                        → smooth interpolation of the two
func Llama3InvFreq(headDim int, base float64, s RopeScaling) []float32 {
	inv := BaseInvFreq(headDim, base)
	lowWavelen := float64(s.OriginalMaxPositions) / s.LowFreqFactor
	highWavelen := float64(s.OriginalMaxPositions) / s.HighFreqFactor
	for i, f := range inv {
		f64 := float64(f)
		wavelen := 2 * math.Pi / f64
		switch {
		case wavelen > lowWavelen:
			inv[i] = float32(f64 / s.Factor)
		case wavelen < highWavelen:
			// unchanged
		default:
			smooth := (float64(s.OriginalMaxPositions)/wavelen - s.LowFreqFactor) / (s.HighFreqFactor - s.LowFreqFactor)
			inv[i] = float32((1-smooth)*f64/s.Factor + smooth*f64)
		}
	}
	return inv
}

// InvFreq returns the scaled inverse-frequency table for this config (the
// short-factor table for longrope), or an error for an unknown scheme.
func (s RopeScaling) InvFreq(headDim int, base float64) ([]float32, error) {
	return s.InvFreqFor(headDim, base, false)
}

// InvFreqFor is InvFreq with an explicit choice of the longrope long-factor
// table; `long` is ignored for other schemes.
func (s RopeScaling) InvFreqFor(headDim int, base float64, long bool) ([]float32, error) {
	switch s.Type {
	case "llama3":
		if s.Factor <= 0 || s.HighFreqFactor <= s.LowFreqFactor || s.OriginalMaxPositions <= 0 {
			return nil, fmt.Errorf("rope_scaling llama3: invalid parameters %+v", s)
		}
		return Llama3InvFreq(headDim, base, s), nil
	case "longrope":
		f := s.ShortFactor
		if long {
			f = s.LongFactor
		}
		if len(f) != headDim/2 {
			return nil, fmt.Errorf("rope_scaling longrope: %d factors for headDim %d (want %d)", len(f), headDim, headDim/2)
		}
		return LongRopeInvFreq(headDim, base, f), nil
	case "yarn":
		if s.OriginalMaxPositions <= 0 {
			return nil, fmt.Errorf("rope_scaling yarn: original_max_position_embeddings is required (%+v)", s)
		}
		return YarnInvFreq(headDim, base, s), nil
	default:
		return nil, fmt.Errorf("rope_scaling: unsupported type %q (have: llama3, longrope, yarn)", s.Type)
	}
}

// invFreqTensor materialises a table as a (headDim/2,) F32 tensor on dev.
func (s RopeScaling) invFreqTensor(headDim int, base float64, long bool, dev device.Device) (*tensor.Tensor, error) {
	vals, err := s.InvFreqFor(headDim, base, long)
	if err != nil {
		return nil, err
	}
	return tensor.New(vals, []int{len(vals)}, dev)
}
