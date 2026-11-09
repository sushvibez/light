package ops

import (
	"fmt"

	"light/internal/registry"
	"light/tensor"
)

// GroupNorm normalises a (N, C, H, W) tensor by splitting C into `groups`
// equal channel groups and computing mean/variance per (n, g) over the
// (C/groups) * H * W elements that fall in that group. Per-channel affine
// (gamma, beta) follows. Used by Stable Diffusion's UNet/VAE.
//
//	x:      (N, C, H, W) F32 / F64 / BF16
//	gamma:  (C,)         same dtype as x
//	beta:   (C,)         same dtype as x
//	groups: int — must divide C
//	eps:    float64
//
// Returns y with the same shape and dtype as x. Caller is responsible for
// landing weights at the right dtype at load time — no in-kernel cast.
func GroupNorm(x, gamma, beta *tensor.Tensor, groups int, eps float64) (*tensor.Tensor, error) {
	if x.Rank() != 4 {
		return nil, fmt.Errorf("GroupNorm: x must be (N, C, H, W), got %v", x.Shape())
	}
	if gamma.Rank() != 1 || beta.Rank() != 1 || gamma.Dim(0) != x.Dim(1) || beta.Dim(0) != x.Dim(1) {
		return nil, fmt.Errorf("GroupNorm: gamma/beta must be (C,) with C=%d", x.Dim(1))
	}
	if !x.Device().SameAs(gamma.Device()) || !x.Device().SameAs(beta.Device()) {
		return nil, fmt.Errorf("GroupNorm: device mismatch")
	}
	if x.DType() != gamma.DType() || x.DType() != beta.DType() {
		return nil, fmt.Errorf("GroupNorm: dtype mismatch (x=%v gamma=%v beta=%v)", x.DType(), gamma.DType(), beta.DType())
	}
	if groups <= 0 || x.Dim(1)%groups != 0 {
		return nil, fmt.Errorf("GroupNorm: groups (%d) must divide C (%d)", groups, x.Dim(1))
	}
	x, err := ensureContig(x)
	if err != nil {
		return nil, err
	}
	gamma, err = ensureContig(gamma)
	if err != nil {
		return nil, err
	}
	beta, err = ensureContig(beta)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Empty(x.DType(), x.Shape().Dims(), x.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "groupnorm", Dev: x.Device().Kind(), DType: x.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{x, gamma, beta},
		Output: out,
		Attrs:  map[string]any{"groups": groups, "eps": eps},
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// RMSNorm computes y = x * rsqrt(mean(x², axis=-1) + eps) * weight along the
// last axis. Used by Llama-family models.
//
//	x: (..., hidden)
//	weight: (hidden,)
//	returns y with same shape as x.
func RMSNorm(x, weight *tensor.Tensor, eps float64) (*tensor.Tensor, error) {
	if !x.Device().SameAs(weight.Device()) {
		return nil, fmt.Errorf("RMSNorm: device mismatch")
	}
	if x.DType() != weight.DType() {
		return nil, fmt.Errorf("RMSNorm: dtype mismatch (x=%v, weight=%v) — cast at load time", x.DType(), weight.DType())
	}
	if weight.Rank() != 1 || weight.Dim(0) != x.Dim(-1) {
		return nil, fmt.Errorf("RMSNorm: weight shape %v incompatible with x last dim %d", weight.Shape(), x.Dim(-1))
	}
	x, err := ensureContig(x)
	if err != nil {
		return nil, err
	}
	weight, err = ensureContig(weight)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Empty(x.DType(), x.Shape().Dims(), x.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "rmsnorm", Dev: x.Device().Kind(), DType: x.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{x, weight},
		Output: out,
		Attrs:  map[string]any{"eps": eps},
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// LayerNorm computes y = gamma * (x - mean) / sqrt(var + eps) + beta along the
// last axis.
func LayerNorm(x, gamma, beta *tensor.Tensor, eps float64) (*tensor.Tensor, error) {
	if !x.Device().SameAs(gamma.Device()) || !x.Device().SameAs(beta.Device()) {
		return nil, fmt.Errorf("LayerNorm: device mismatch")
	}
	if x.DType() != gamma.DType() || x.DType() != beta.DType() {
		return nil, fmt.Errorf("LayerNorm: dtype mismatch (x=%v gamma=%v beta=%v) — cast at load time", x.DType(), gamma.DType(), beta.DType())
	}
	last := x.Dim(-1)
	if gamma.Rank() != 1 || gamma.Dim(0) != last || beta.Rank() != 1 || beta.Dim(0) != last {
		return nil, fmt.Errorf("LayerNorm: gamma/beta shape mismatch")
	}
	x, err := ensureContig(x)
	if err != nil {
		return nil, err
	}
	gamma, err = ensureContig(gamma)
	if err != nil {
		return nil, err
	}
	beta, err = ensureContig(beta)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Empty(x.DType(), x.Shape().Dims(), x.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "layernorm", Dev: x.Device().Kind(), DType: x.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{x, gamma, beta},
		Output: out,
		Attrs:  map[string]any{"eps": eps},
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// ensureContig is a tiny helper so multi-arg ops don't repeat the same check.
func ensureContig(t *tensor.Tensor) (*tensor.Tensor, error) {
	if t.IsContiguous() && t.Offset() == 0 {
		return t, nil
	}
	return t.Contiguous()
}
