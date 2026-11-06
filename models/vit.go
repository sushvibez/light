package models

import (
	"fmt"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// ViTConfig captures one Vision Transformer variant.
//
//	ImgSize=224  PatchSize=16  ⇒ 14·14 = 196 patches  ⇒ 197 tokens (with CLS)
type ViTConfig struct {
	ImgSize    int
	PatchSize  int
	InChannels int
	NumClasses int
	NEmbd      int
	NLayer     int
	NHead      int
	MlpRatio   float64
	LayerNorm  float64
}

// ViTBase16 is the canonical ViT-B/16 from Dosovitskiy et al. 2020.
func ViTBase16() ViTConfig {
	return ViTConfig{
		ImgSize: 224, PatchSize: 16, InChannels: 3, NumClasses: 1000,
		NEmbd: 768, NLayer: 12, NHead: 12, MlpRatio: 4.0, LayerNorm: 1e-6,
	}
}

// ViTLarge16 is ViT-L/16 (≈304M params).
func ViTLarge16() ViTConfig {
	return ViTConfig{
		ImgSize: 224, PatchSize: 16, InChannels: 3, NumClasses: 1000,
		NEmbd: 1024, NLayer: 24, NHead: 16, MlpRatio: 4.0, LayerNorm: 1e-6,
	}
}

// ViTHuge14 is ViT-H/14.
func ViTHuge14() ViTConfig {
	return ViTConfig{
		ImgSize: 224, PatchSize: 14, InChannels: 3, NumClasses: 1000,
		NEmbd: 1280, NLayer: 32, NHead: 16, MlpRatio: 4.0, LayerNorm: 1e-6,
	}
}

// ViT is a standard Vision Transformer encoder + classification head.
type ViT struct {
	Cfg       ViTConfig
	PatchConv *nn.Conv2d     // (Cin, NEmbd, k=PatchSize, stride=PatchSize, no padding)
	ClsToken  *tensor.Tensor // (1, 1, NEmbd) — broadcast across batch
	PosEmb    *tensor.Tensor // (1, NumPatches+1, NEmbd)
	Blocks    []*vitBlock
	NormFinal *nn.LayerNorm
	Head      *nn.Linear
	training  bool
}

type vitBlock struct {
	Norm1 *nn.LayerNorm
	Attn  *nn.MultiHeadAttention
	Norm2 *nn.LayerNorm
	FC1   *nn.Linear
	FC2   *nn.Linear
}

// NewViT builds a ViT with the given config and random weights.
func NewViT(cfg ViTConfig, dev device.Device) (*ViT, error) {
	if cfg.ImgSize%cfg.PatchSize != 0 {
		return nil, fmt.Errorf("models.ViT: ImgSize %d not divisible by PatchSize %d", cfg.ImgSize, cfg.PatchSize)
	}
	gridSize := cfg.ImgSize / cfg.PatchSize
	numPatches := gridSize * gridSize

	patchConv, err := nn.NewConv2d(cfg.InChannels, cfg.NEmbd, cfg.PatchSize, cfg.PatchSize,
		[2]int{cfg.PatchSize, cfg.PatchSize}, [2]int{0, 0}, true, dev)
	if err != nil {
		return nil, err
	}
	cls, err := tensor.Zeros(dtype.F32, []int{1, 1, cfg.NEmbd}, dev)
	if err != nil {
		return nil, err
	}
	pos, err := tensor.Zeros(dtype.F32, []int{1, numPatches + 1, cfg.NEmbd}, dev)
	if err != nil {
		return nil, err
	}
	mlpHidden := int(float64(cfg.NEmbd) * cfg.MlpRatio)
	blocks := make([]*vitBlock, cfg.NLayer)
	for i := range blocks {
		ln1, err := nn.NewLayerNorm(cfg.NEmbd, cfg.LayerNorm, dev)
		if err != nil {
			return nil, err
		}
		attn, err := nn.NewMultiHeadAttention(cfg.NEmbd, cfg.NHead, false, true, dev)
		if err != nil {
			return nil, err
		}
		ln2, err := nn.NewLayerNorm(cfg.NEmbd, cfg.LayerNorm, dev)
		if err != nil {
			return nil, err
		}
		fc1, err := nn.NewLinear(cfg.NEmbd, mlpHidden, true, dev)
		if err != nil {
			return nil, err
		}
		fc2, err := nn.NewLinear(mlpHidden, cfg.NEmbd, true, dev)
		if err != nil {
			return nil, err
		}
		blocks[i] = &vitBlock{Norm1: ln1, Attn: attn, Norm2: ln2, FC1: fc1, FC2: fc2}
	}
	normFinal, err := nn.NewLayerNorm(cfg.NEmbd, cfg.LayerNorm, dev)
	if err != nil {
		return nil, err
	}
	head, err := nn.NewLinear(cfg.NEmbd, cfg.NumClasses, true, dev)
	if err != nil {
		return nil, err
	}
	return &ViT{
		Cfg: cfg, PatchConv: patchConv,
		ClsToken: cls, PosEmb: pos,
		Blocks: blocks, NormFinal: normFinal, Head: head,
	}, nil
}

// Forward classifies a batch of (B, C, H, W) RGB images.
func (v *ViT) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Rank() != 4 {
		return nil, fmt.Errorf("ViT: expected (B, C, H, W), got %v", x.Shape())
	}
	B := x.Dim(0)
	D := v.Cfg.NEmbd

	// Patch embed: Conv with kernel=stride=PatchSize → (B, D, gH, gW).
	h, err := v.PatchConv.Forward(x)
	if err != nil {
		return nil, err
	}
	gH := h.Dim(2)
	gW := h.Dim(3)
	N := gH * gW

	// (B, D, gH, gW) → (B, D, N) → permute (B, N, D)
	h, err = autograd.Reshape(h, B, D, N)
	if err != nil {
		return nil, err
	}
	h, err = autograd.Permute(h, 0, 2, 1) // (B, N, D)
	if err != nil {
		return nil, err
	}
	h, err = autograd.Contiguous(h)
	if err != nil {
		return nil, err
	}

	// Prepend CLS token: broadcast (1, 1, D) → (B, 1, D), then concat.
	clsB, err := v.ClsToken.Expand(B, 1, D)
	if err != nil {
		return nil, err
	}
	clsBC, err := clsB.Contiguous()
	if err != nil {
		return nil, err
	}
	tokens, err := ops.Concat([]*tensor.Tensor{clsBC, h}, 1) // (B, N+1, D)
	if err != nil {
		return nil, err
	}
	// Add position embedding (broadcast across batch).
	tokens, err = autograd.Add(tokens, v.PosEmb)
	if err != nil {
		return nil, err
	}
	_ = N

	// Transformer encoder.
	for _, blk := range v.Blocks {
		tokens, err = blk.forward(tokens)
		if err != nil {
			return nil, err
		}
	}
	tokens, err = v.NormFinal.Forward(tokens)
	if err != nil {
		return nil, err
	}
	// Take CLS token output: tokens[:, 0, :] → (B, D).
	cls, err := tokens.Narrow(1, 0, 1)
	if err != nil {
		return nil, err
	}
	cls, err = cls.Contiguous()
	if err != nil {
		return nil, err
	}
	cls, err = autograd.Reshape(cls, B, D)
	if err != nil {
		return nil, err
	}
	return v.Head.Forward(cls)
}

func (b *vitBlock) forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	// Pre-norm attention.
	h, err := b.Norm1.Forward(x)
	if err != nil {
		return nil, err
	}
	h, err = b.Attn.Forward(h)
	if err != nil {
		return nil, err
	}
	x, err = autograd.Add(x, h)
	if err != nil {
		return nil, err
	}
	// Pre-norm MLP (GELU).
	h, err = b.Norm2.Forward(x)
	if err != nil {
		return nil, err
	}
	h, err = b.FC1.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = autograd.GELU(h)
	if err != nil {
		return nil, err
	}
	h, err = b.FC2.Forward(h)
	if err != nil {
		return nil, err
	}
	return autograd.Add(x, h)
}

func (v *ViT) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, v.PatchConv.Parameters()...)
	ps = append(ps, v.ClsToken)
	ps = append(ps, v.PosEmb)
	for _, b := range v.Blocks {
		ps = append(ps, b.Norm1.Parameters()...)
		ps = append(ps, b.Attn.Parameters()...)
		ps = append(ps, b.Norm2.Parameters()...)
		ps = append(ps, b.FC1.Parameters()...)
		ps = append(ps, b.FC2.Parameters()...)
	}
	ps = append(ps, v.NormFinal.Parameters()...)
	ps = append(ps, v.Head.Parameters()...)
	return ps
}

func (v *ViT) ParamCount() int {
	total := 0
	for _, p := range v.Parameters() {
		total += p.NumElements()
	}
	return total
}

func (v *ViT) SetTraining(t bool) { v.training = t }
