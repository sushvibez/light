package nn

import (
	"fmt"

	"light/autograd"
	"light/device"
	"light/tensor"
)

// TransformerBlock is one pre-norm transformer layer:
//
//	h = x + SelfAttention( LayerNorm1(x) )
//	y = h + FFN( LayerNorm2(h) )
//
// FFN is Linear(H, 4H) → SiLU → Linear(4H, H).
// Uses SelfAttention (single-head) and LayerNorm; a Llama-style variant using
// RMSNorm is a trivial variant of this template.
type TransformerBlock struct {
	Norm1  *LayerNorm
	Attn   *SelfAttention
	Norm2  *LayerNorm
	FFN    *Sequential
	hidden int
}

// NewTransformerBlock builds a pre-norm transformer block with `hidden` as
// the model width. Expansion ratio for FFN is fixed at 4.
func NewTransformerBlock(hidden int, causal bool, dev device.Device) (*TransformerBlock, error) {
	if hidden <= 0 {
		return nil, fmt.Errorf("nn.TransformerBlock: hidden must be positive")
	}
	n1, err := NewLayerNorm(hidden, 1e-5, dev)
	if err != nil {
		return nil, err
	}
	attn, err := NewSelfAttention(hidden, causal, dev)
	if err != nil {
		return nil, err
	}
	n2, err := NewLayerNorm(hidden, 1e-5, dev)
	if err != nil {
		return nil, err
	}
	up, err := NewLinear(hidden, hidden*4, true, dev)
	if err != nil {
		return nil, err
	}
	down, err := NewLinear(hidden*4, hidden, true, dev)
	if err != nil {
		return nil, err
	}
	ffn := NewSequential(up, SiLU{}, down)
	return &TransformerBlock{
		Norm1: n1, Attn: attn, Norm2: n2, FFN: ffn, hidden: hidden,
	}, nil
}

// Forward applies pre-norm attention + residual, then pre-norm FFN + residual.
func (b *TransformerBlock) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	// Attention sub-layer: x = x + Attn(LN1(x))
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

	// FFN sub-layer: x = x + FFN(LN2(x))
	h, err = b.Norm2.Forward(x)
	if err != nil {
		return nil, err
	}
	h, err = b.FFN.Forward(h)
	if err != nil {
		return nil, err
	}
	return autograd.Add(x, h)
}

// Parameters returns all trainable tensors in this block.
func (b *TransformerBlock) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, b.Norm1.Parameters()...)
	ps = append(ps, b.Attn.Parameters()...)
	ps = append(ps, b.Norm2.Parameters()...)
	ps = append(ps, b.FFN.Parameters()...)
	return ps
}

// Hidden returns the model width.
func (b *TransformerBlock) Hidden() int { return b.hidden }
