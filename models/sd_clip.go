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

// CLIPTextConfig describes the CLIP-L/14 text encoder used by Stable
// Diffusion 1.x. Defaults match `openai/clip-vit-large-patch14`:
//
//	VocabSize=49408, MaxPosition=77, Hidden=768, Layers=12, Heads=12.
type CLIPTextConfig struct {
	VocabSize        int
	MaxPosition      int
	HiddenSize       int
	NumLayers        int
	NumHeads         int
	ProjectionDim    int         // unused by SD's CLIPTextModel — kept for parity
	IntermediateSize int         // MLP width; 0 means 4·HiddenSize
	LayerNormEps     float64     // 1e-5
	UseQuickGELU     bool        // true for OpenAI/SD CLIP; false for HF "openai" port checkpoints
	WeightDType      dtype.DType // BF16 for half-memory inference; defaults to F32
}

// CLIPTextSD15 returns the config used by SD 1.5's text encoder.
func CLIPTextSD15() CLIPTextConfig {
	return CLIPTextConfig{
		VocabSize: 49408, MaxPosition: 77,
		HiddenSize: 768, NumLayers: 12, NumHeads: 12,
		LayerNormEps: 1e-5, UseQuickGELU: true,
	}
}

type clipBlock struct {
	N1   *nn.LayerNorm
	Attn *nn.MultiHeadAttention
	N2   *nn.LayerNorm
	FC1  *nn.Linear
	FC2  *nn.Linear
}

func newCLIPBlock(cfg CLIPTextConfig, dev device.Device) (*clipBlock, error) {
	n1, err := nn.NewLayerNorm(cfg.HiddenSize, cfg.LayerNormEps, dev)
	if err != nil {
		return nil, err
	}
	// CLIP uses bias=true on Q/K/V/O and is causal (to mask padding tokens).
	attn, err := nn.NewMultiHeadAttention(cfg.HiddenSize, cfg.NumHeads, true, true, dev)
	if err != nil {
		return nil, err
	}
	n2, err := nn.NewLayerNorm(cfg.HiddenSize, cfg.LayerNormEps, dev)
	if err != nil {
		return nil, err
	}
	inter := cfg.IntermediateSize
	if inter == 0 {
		inter = 4 * cfg.HiddenSize
	}
	fc1, err := nn.NewLinear(cfg.HiddenSize, inter, true, dev)
	if err != nil {
		return nil, err
	}
	fc2, err := nn.NewLinear(inter, cfg.HiddenSize, true, dev)
	if err != nil {
		return nil, err
	}
	return &clipBlock{N1: n1, Attn: attn, N2: n2, FC1: fc1, FC2: fc2}, nil
}

func (b *clipBlock) forward(x *tensor.Tensor, useQuickGELU bool) (*tensor.Tensor, error) {
	h, err := b.N1.Forward(x)
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
	h, err = b.N2.Forward(x)
	if err != nil {
		return nil, err
	}
	h, err = b.FC1.Forward(h)
	if err != nil {
		return nil, err
	}
	if useQuickGELU {
		h, err = quickGELU(h)
	} else {
		h, err = ops.GeluErf(h) // HF "gelu" is the exact erf form (CLIP-G, the HF CLIP ports)
	}
	if err != nil {
		return nil, err
	}
	h, err = b.FC2.Forward(h)
	if err != nil {
		return nil, err
	}
	return autograd.Add(x, h)
}

func (b *clipBlock) parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, b.N1.Parameters()...)
	ps = append(ps, b.Attn.Parameters()...)
	ps = append(ps, b.N2.Parameters()...)
	ps = append(ps, b.FC1.Parameters()...)
	ps = append(ps, b.FC2.Parameters()...)
	return ps
}

// quickGELU computes x * sigmoid(1.702 * x). OpenAI's CLIP and SD's text
// encoder use this in the FFN; HF Transformers calls it `gelu_quick`.
func quickGELU(x *tensor.Tensor) (*tensor.Tensor, error) {
	scale, err := tensor.New([]float32{1.702}, []int{1}, x.Device())
	if err != nil {
		return nil, err
	}
	scaled, err := autograd.Mul(x, scale)
	if err != nil {
		return nil, err
	}
	sig, err := autograd.Sigmoid(scaled)
	if err != nil {
		return nil, err
	}
	return autograd.Mul(x, sig)
}

// CLIPText is the SD 1.5 text encoder: token+position embeddings → 12
// pre-norm causal-attention blocks → final LayerNorm. Forward consumes
// (B, T) I64 tokens and returns (B, T, HiddenSize) F32 hidden states.
type CLIPText struct {
	Cfg       CLIPTextConfig
	TokenEmb  *nn.Embedding
	PosEmb    *nn.Embedding
	Blocks    []*clipBlock
	FinalNorm *nn.LayerNorm
}

// NewCLIPText constructs the text encoder with random init. Loaders
// overwrite values; see SkipInit for a no-init fast path.
func NewCLIPText(cfg CLIPTextConfig, skipInit bool, dev device.Device) (*CLIPText, error) {
	var tok, pos *nn.Embedding
	var err error
	if skipInit {
		tok, err = nn.NewEmbeddingUninit(cfg.VocabSize, cfg.HiddenSize, dtype.F32, dev)
		if err != nil {
			return nil, err
		}
		pos, err = nn.NewEmbeddingUninit(cfg.MaxPosition, cfg.HiddenSize, dtype.F32, dev)
	} else {
		tok, err = nn.NewEmbedding(cfg.VocabSize, cfg.HiddenSize, dev)
		if err != nil {
			return nil, err
		}
		pos, err = nn.NewEmbedding(cfg.MaxPosition, cfg.HiddenSize, dev)
	}
	if err != nil {
		return nil, err
	}
	blocks := make([]*clipBlock, cfg.NumLayers)
	for i := range blocks {
		b, err := newCLIPBlock(cfg, dev)
		if err != nil {
			return nil, fmt.Errorf("CLIP layer %d: %w", i, err)
		}
		blocks[i] = b
	}
	fn, err := nn.NewLayerNorm(cfg.HiddenSize, cfg.LayerNormEps, dev)
	if err != nil {
		return nil, err
	}
	return &CLIPText{Cfg: cfg, TokenEmb: tok, PosEmb: pos, Blocks: blocks, FinalNorm: fn}, nil
}

// Forward runs the text encoder. `tokens` is (B, T) I64 — for SD T must be
// 77 (`Cfg.MaxPosition`), and the input is typically padded with the EOT
// token.
func (m *CLIPText) Forward(tokens *tensor.Tensor) (*tensor.Tensor, error) {
	if tokens.Rank() != 2 || tokens.DType() != dtype.I64 {
		return nil, fmt.Errorf("CLIPText: tokens must be (B, T) I64, got %v %v", tokens.Shape(), tokens.DType())
	}
	B := tokens.Dim(0)
	T := tokens.Dim(1)
	if T > m.Cfg.MaxPosition {
		return nil, fmt.Errorf("CLIPText: sequence length %d > MaxPosition %d", T, m.Cfg.MaxPosition)
	}
	tokH, err := m.TokenEmb.Forward(tokens)
	if err != nil {
		return nil, err
	}
	posIdx, err := makePosIdx(T, B, tokens.Device())
	if err != nil {
		return nil, err
	}
	posH, err := m.PosEmb.Forward(posIdx)
	if err != nil {
		return nil, err
	}
	h, err := autograd.Add(tokH, posH)
	if err != nil {
		return nil, err
	}
	for _, blk := range m.Blocks {
		h, err = blk.forward(h, m.Cfg.UseQuickGELU)
		if err != nil {
			return nil, err
		}
	}
	return m.FinalNorm.Forward(h)
}

// Parameters returns every learnable tensor.
func (m *CLIPText) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, m.TokenEmb.Parameters()...)
	ps = append(ps, m.PosEmb.Parameters()...)
	for _, b := range m.Blocks {
		ps = append(ps, b.parameters()...)
	}
	ps = append(ps, m.FinalNorm.Parameters()...)
	return ps
}

// ParamCount returns the total trainable element count.
func (m *CLIPText) ParamCount() int {
	total := 0
	for _, p := range m.Parameters() {
		total += p.NumElements()
	}
	return total
}

// silence unused-import for ops when the only consumer is quickGELU's
// deeper forms; keeps the file tidy as more activations land.
var _ = ops.SiLU
