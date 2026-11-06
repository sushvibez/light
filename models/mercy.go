package models

// MercyLM is a port of https://github.com/ravipatib/mercylm — a small
// (~15.4 M param) GPT-style causal LM trained for conversational response.
//
// Architecture (reference config):
//
//	vocab_size:  4096 (BPE, dynamic)
//	context_len: 128
//	embed_dim:   384
//	num_heads:   8  (head_dim = 48)
//	num_layers:  10
//	ffn_dim:     1024
//
// Layout: TokenEmb + PosEmb → 10× TransformerBlock → LayerNorm → LM head
// (LM head weight is tied to the token embedding).
//
// Each block uses pre-norm:
//
//	x = x + Attn(LN(x))
//	x = x + FFN(LN(x))
//
// Attention projections carry no bias; FFN linears carry bias.
// FFN activation is ReLU (not GELU/SwiGLU).

import (
	"fmt"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/nn"
	"light/tensor"
)

// MercyConfig holds every hyper-parameter needed to build the model.
type MercyConfig struct {
	VocabSize  int
	ContextLen int
	EmbedDim   int
	NumHeads   int
	NumLayers  int
	FFNDim     int
	LNEps      float64
}

// DefaultMercyConfig returns the reference MercyLM hyper-parameters. Pass
// the actual vocab size from `vocab_size.txt` (default 4096 when training
// from scratch with the reference BPE tokenizer).
func DefaultMercyConfig(vocabSize int) MercyConfig {
	return MercyConfig{
		VocabSize:  vocabSize,
		ContextLen: 128,
		EmbedDim:   384,
		NumHeads:   8,
		NumLayers:  10,
		FFNDim:     1024,
		LNEps:      1e-5,
	}
}

// mercyBlock is one pre-norm transformer block.
type mercyBlock struct {
	Norm1 *nn.LayerNorm
	Attn  *nn.MultiHeadAttention
	Norm2 *nn.LayerNorm
	FC1   *nn.Linear
	FC2   *nn.Linear
}

func newMercyBlock(cfg MercyConfig, dev device.Device) (*mercyBlock, error) {
	n1, err := nn.NewLayerNorm(cfg.EmbedDim, cfg.LNEps, dev)
	if err != nil {
		return nil, err
	}
	// bias=false on attention projections — matches the reference.
	attn, err := nn.NewMultiHeadAttention(cfg.EmbedDim, cfg.NumHeads, true, false, dev)
	if err != nil {
		return nil, err
	}
	n2, err := nn.NewLayerNorm(cfg.EmbedDim, cfg.LNEps, dev)
	if err != nil {
		return nil, err
	}
	// FFN has bias (PyTorch nn.Linear default).
	fc1, err := nn.NewLinear(cfg.EmbedDim, cfg.FFNDim, true, dev)
	if err != nil {
		return nil, err
	}
	fc2, err := nn.NewLinear(cfg.FFNDim, cfg.EmbedDim, true, dev)
	if err != nil {
		return nil, err
	}
	return &mercyBlock{Norm1: n1, Attn: attn, Norm2: n2, FC1: fc1, FC2: fc2}, nil
}

func (b *mercyBlock) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	// Attention sub-layer (pre-norm)
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
	// FFN sub-layer (pre-norm): LN → FC1 → ReLU → FC2
	h, err = b.Norm2.Forward(x)
	if err != nil {
		return nil, err
	}
	h, err = b.FC1.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = autograd.ReLU(h)
	if err != nil {
		return nil, err
	}
	h, err = b.FC2.Forward(h)
	if err != nil {
		return nil, err
	}
	return autograd.Add(x, h)
}

func (b *mercyBlock) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, b.Norm1.Parameters()...)
	ps = append(ps, b.Attn.Parameters()...)
	ps = append(ps, b.Norm2.Parameters()...)
	ps = append(ps, b.FC1.Parameters()...)
	ps = append(ps, b.FC2.Parameters()...)
	return ps
}

// MercyLM is the full model. Takes (B, T) I64 token IDs and returns
// (B, T, VocabSize) F32 logits.
type MercyLM struct {
	Cfg       MercyConfig
	TokenEmb  *nn.Embedding
	PosEmb    *nn.Embedding
	Blocks    []*mercyBlock
	FinalNorm *nn.LayerNorm
	LMHead    *nn.Linear // weight tied to TokenEmb.Weight
}

// NewMercyLM constructs a MercyLM with Xavier/normal random init on CPU
// then moves weights to `dev`. For training from scratch. To load
// pretrained weights use LoadMercyFromSafetensors.
func NewMercyLM(cfg MercyConfig, dev device.Device) (*MercyLM, error) {
	if cfg.EmbedDim%cfg.NumHeads != 0 {
		return nil, fmt.Errorf("MercyLM: EmbedDim (%d) not divisible by NumHeads (%d)", cfg.EmbedDim, cfg.NumHeads)
	}
	tokEmb, err := nn.NewEmbedding(cfg.VocabSize, cfg.EmbedDim, dev)
	if err != nil {
		return nil, err
	}
	posEmb, err := nn.NewEmbedding(cfg.ContextLen, cfg.EmbedDim, dev)
	if err != nil {
		return nil, err
	}
	blocks := make([]*mercyBlock, cfg.NumLayers)
	for i := range blocks {
		b, err := newMercyBlock(cfg, dev)
		if err != nil {
			return nil, err
		}
		blocks[i] = b
	}
	ln, err := nn.NewLayerNorm(cfg.EmbedDim, cfg.LNEps, dev)
	if err != nil {
		return nil, err
	}
	head, err := nn.NewLinear(cfg.EmbedDim, cfg.VocabSize, false, dev)
	if err != nil {
		return nil, err
	}
	// Tie LM-head weight to the token embedding — same convention as GPT-2.
	head.W = tokEmb.Weight

	return &MercyLM{
		Cfg: cfg, TokenEmb: tokEmb, PosEmb: posEmb,
		Blocks: blocks, FinalNorm: ln, LMHead: head,
	}, nil
}

// NewMercyLMUninit constructs a MercyLM with zero-init weights (use when a
// loader is about to overwrite all values).
func NewMercyLMUninit(cfg MercyConfig, dev device.Device) (*MercyLM, error) {
	if cfg.EmbedDim%cfg.NumHeads != 0 {
		return nil, fmt.Errorf("MercyLM: EmbedDim (%d) not divisible by NumHeads (%d)", cfg.EmbedDim, cfg.NumHeads)
	}
	tokEmb, err := nn.NewEmbeddingUninit(cfg.VocabSize, cfg.EmbedDim, dtype.F32, dev)
	if err != nil {
		return nil, err
	}
	posEmb, err := nn.NewEmbeddingUninit(cfg.ContextLen, cfg.EmbedDim, dtype.F32, dev)
	if err != nil {
		return nil, err
	}
	blocks := make([]*mercyBlock, cfg.NumLayers)
	for i := range blocks {
		b, err := newMercyBlock(cfg, dev)
		if err != nil {
			return nil, err
		}
		blocks[i] = b
	}
	ln, err := nn.NewLayerNorm(cfg.EmbedDim, cfg.LNEps, dev)
	if err != nil {
		return nil, err
	}
	head, err := nn.NewLinear(cfg.EmbedDim, cfg.VocabSize, false, dev)
	if err != nil {
		return nil, err
	}
	head.W = tokEmb.Weight

	return &MercyLM{
		Cfg: cfg, TokenEmb: tokEmb, PosEmb: posEmb,
		Blocks: blocks, FinalNorm: ln, LMHead: head,
	}, nil
}

// Forward takes tokens (B, T) I64 and returns logits (B, T, VocabSize) F32.
func (m *MercyLM) Forward(tokens *tensor.Tensor) (*tensor.Tensor, error) {
	if tokens.Rank() != 2 {
		return nil, fmt.Errorf("MercyLM: tokens must be (B, T), got %v", tokens.Shape())
	}
	if tokens.DType() != dtype.I64 {
		return nil, fmt.Errorf("MercyLM: tokens must be I64, got %v", tokens.DType())
	}
	T := tokens.Dim(1)
	if T > m.Cfg.ContextLen {
		return nil, fmt.Errorf("MercyLM: sequence length %d > ContextLen %d", T, m.Cfg.ContextLen)
	}
	B := tokens.Dim(0)

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
		h, err = blk.Forward(h)
		if err != nil {
			return nil, err
		}
	}
	h, err = m.FinalNorm.Forward(h)
	if err != nil {
		return nil, err
	}
	return m.LMHead.Forward(h)
}

// Parameters returns all unique trainable tensors (LM head excluded when
// it is tied to the token embedding).
func (m *MercyLM) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, m.TokenEmb.Parameters()...)
	ps = append(ps, m.PosEmb.Parameters()...)
	for _, b := range m.Blocks {
		ps = append(ps, b.Parameters()...)
	}
	ps = append(ps, m.FinalNorm.Parameters()...)
	for _, p := range m.LMHead.Parameters() {
		if p == m.TokenEmb.Weight {
			continue
		}
		ps = append(ps, p)
	}
	return ps
}

// ParamCount returns total trainable element count (tied weight counted once).
func (m *MercyLM) ParamCount() int {
	n := 0
	for _, p := range m.Parameters() {
		n += p.NumElements()
	}
	return n
}
