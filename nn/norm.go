package nn

import (
	"fmt"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// wantsGrad reports whether any of the tensors participates in autograd.
// Norm modules take the fused single-kernel inference op otherwise, which
// also covers BF16 inputs the autograd decomposition cannot handle yet.
func wantsGrad(ts ...*tensor.Tensor) bool {
	for _, t := range ts {
		if t != nil && t.RequiresGrad() {
			return true
		}
	}
	return false
}

// RMSNorm implements Llama-style root-mean-square normalization along the
// last axis:  y = x * rsqrt(mean(x², axis=-1) + eps) * weight.
type RMSNorm struct {
	Weight *tensor.Tensor
	Eps    float64
}

// NewRMSNorm initializes weight to ones and takes the given eps.
func NewRMSNorm(hidden int, eps float64, dev device.Device) (*RMSNorm, error) {
	return NewRMSNormWithDType(hidden, eps, dtype.F32, dev)
}

// NewRMSNormWithDType allocates the weight at the given dtype. Use this on
// the BF16 inference path so the weight matches the input — the fused
// rmsnorm kernel requires identical weight/input dtype.
func NewRMSNormWithDType(hidden int, eps float64, wdt dtype.DType, dev device.Device) (*RMSNorm, error) {
	w, err := tensor.Ones(wdt, []int{hidden}, dev)
	if err != nil {
		return nil, err
	}
	return &RMSNorm{Weight: w, Eps: eps}, nil
}

func (n *RMSNorm) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	// Training: differentiable decomposition. Inference: the fused
	// single-kernel ops.RMSNorm (also handles BF16 input + F32 weight).
	if wantsGrad(x, n.Weight) {
		return autograd.RMSNorm(x, n.Weight, float32(n.Eps))
	}
	return ops.RMSNorm(x, n.Weight, n.Eps)
}

func (n *RMSNorm) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{n.Weight}
}

// GroupNorm normalises (N, C, H, W) tensors by splitting C into `Groups`
// equal channel groups, computing mean/variance per (n, g) over (C/G)*H*W
// elements, and applying per-channel affine (Gamma, Beta).
//
// Used by Stable Diffusion's UNet and VAE (always with Groups=32). Weight
// dtype matches the input dtype — the fused groupnorm kernel insists on it.
type GroupNorm struct {
	Gamma  *tensor.Tensor
	Beta   *tensor.Tensor
	Groups int
	Eps    float64
}

// NewGroupNorm initialises gamma to ones and beta to zeros (F32).
func NewGroupNorm(channels, groups int, eps float64, dev device.Device) (*GroupNorm, error) {
	return NewGroupNormWithDType(channels, groups, eps, dtype.F32, dev)
}

// NewGroupNormWithDType allocates gamma/beta at the given dtype. Use the
// BF16 variant so the SD UNet/VAE inference path stays single-dtype.
func NewGroupNormWithDType(channels, groups int, eps float64, wdt dtype.DType, dev device.Device) (*GroupNorm, error) {
	if groups <= 0 || channels%groups != 0 {
		return nil, fmt.Errorf("nn.GroupNorm: groups (%d) must divide channels (%d)", groups, channels)
	}
	g, err := tensor.Ones(wdt, []int{channels}, dev)
	if err != nil {
		return nil, err
	}
	b, err := tensor.Zeros(wdt, []int{channels}, dev)
	if err != nil {
		return nil, err
	}
	return &GroupNorm{Gamma: g, Beta: b, Groups: groups, Eps: eps}, nil
}

func (n *GroupNorm) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	return ops.GroupNorm(x, n.Gamma, n.Beta, n.Groups, n.Eps)
}

func (n *GroupNorm) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{n.Gamma, n.Beta}
}

// LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta, last-axis.
type LayerNorm struct {
	Gamma *tensor.Tensor
	Beta  *tensor.Tensor
	Eps   float64
}

// NewLayerNorm initializes gamma to ones and beta to zeros (F32).
func NewLayerNorm(hidden int, eps float64, dev device.Device) (*LayerNorm, error) {
	return NewLayerNormWithDType(hidden, eps, dtype.F32, dev)
}

// NewLayerNormWithDType allocates gamma/beta at the given dtype. The fused
// layernorm kernel requires gamma/beta dtype to match the input dtype, so
// the BF16 inference path uses this with wdt=BF16.
func NewLayerNormWithDType(hidden int, eps float64, wdt dtype.DType, dev device.Device) (*LayerNorm, error) {
	g, err := tensor.Ones(wdt, []int{hidden}, dev)
	if err != nil {
		return nil, err
	}
	b, err := tensor.Zeros(wdt, []int{hidden}, dev)
	if err != nil {
		return nil, err
	}
	return &LayerNorm{Gamma: g, Beta: b, Eps: eps}, nil
}

func (n *LayerNorm) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	// Training: differentiable decomposition. Inference: the fused
	// `ops.LayerNorm` (also handles BF16 input + F32 gamma/beta — the
	// autograd decomposition reaches into ops.Mean/Sub/Sqrt, which are not
	// registered for BF16, so it is only taken when a gradient is wanted).
	if wantsGrad(x, n.Gamma, n.Beta) {
		return autograd.LayerNorm(x, n.Gamma, n.Beta, float32(n.Eps))
	}
	return ops.LayerNorm(x, n.Gamma, n.Beta, n.Eps)
}

func (n *LayerNorm) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{n.Gamma, n.Beta}
}
