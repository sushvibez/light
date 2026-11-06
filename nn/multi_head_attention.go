package nn

import (
	"fmt"
	"math"

	"light/autograd"
	"light/device"
	"light/tensor"
)

// MultiHeadAttention implements standard multi-head self-attention with
// optional causal masking.
//
// Input x: (S, H) where S=seqLen, H=hidden. Internally reshapes to
// (S, NumHeads, HeadDim) → permutes to (NumHeads, S, HeadDim), uses
// batched matmul for the per-head (Q·Kᵀ) and (attn·V), then permutes and
// reshapes back to (S, H) before the output projection.
//
// hidden must be divisible by numHeads.
type MultiHeadAttention struct {
	QProj, KProj, VProj, OProj *Linear
	numHeads                   int
	headDim                    int
	hidden                     int
	causal                     bool
	scale                      *tensor.Tensor
	maskCache                  *tensor.Tensor
	maskDev                    device.Device
	maskLen                    int
}

// NewMultiHeadAttention constructs the module. `hidden` must be divisible by
// `numHeads`. `bias` toggles whether Q/K/V/O linears include a bias term —
// GPT-2 uses bias=true; Llama uses bias=false.
func NewMultiHeadAttention(hidden, numHeads int, causal, bias bool, dev device.Device) (*MultiHeadAttention, error) {
	if hidden <= 0 || numHeads <= 0 {
		return nil, fmt.Errorf("nn.MultiHeadAttention: hidden and numHeads must be positive")
	}
	if hidden%numHeads != 0 {
		return nil, fmt.Errorf("nn.MultiHeadAttention: hidden (%d) not divisible by numHeads (%d)", hidden, numHeads)
	}
	headDim := hidden / numHeads
	q, err := NewLinear(hidden, hidden, bias, dev)
	if err != nil {
		return nil, err
	}
	k, err := NewLinear(hidden, hidden, bias, dev)
	if err != nil {
		return nil, err
	}
	v, err := NewLinear(hidden, hidden, bias, dev)
	if err != nil {
		return nil, err
	}
	o, err := NewLinear(hidden, hidden, bias, dev)
	if err != nil {
		return nil, err
	}
	scaleVal := float32(1.0 / math.Sqrt(float64(headDim)))
	scaleT, err := tensor.New([]float32{scaleVal}, []int{1}, dev)
	if err != nil {
		return nil, err
	}
	return &MultiHeadAttention{
		QProj: q, KProj: k, VProj: v, OProj: o,
		numHeads: numHeads, headDim: headDim, hidden: hidden,
		causal: causal, scale: scaleT,
	}, nil
}

// Forward runs multi-head self-attention on x of shape (S, H) or (B, S, H).
// When a batch dim is present, each batch element is processed independently;
// gradients still flow back through shared Q/K/V/O projections.
func (m *MultiHeadAttention) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Rank() < 2 || x.Dim(-1) != m.hidden {
		return nil, fmt.Errorf("nn.MultiHeadAttention: expected (..., S, %d), got %v", m.hidden, x.Shape())
	}
	// Normalise: always work with (B, S, H) internally.
	unbatched := x.Rank() == 2
	if unbatched {
		var err error
		x, err = autograd.Reshape(x, 1, x.Dim(0), x.Dim(1))
		if err != nil {
			return nil, err
		}
	}
	B := x.Dim(0)
	S := x.Dim(1)
	NH := m.numHeads
	HD := m.headDim

	q, err := m.QProj.Forward(x)
	if err != nil {
		return nil, err
	}
	k, err := m.KProj.Forward(x)
	if err != nil {
		return nil, err
	}
	v, err := m.VProj.Forward(x)
	if err != nil {
		return nil, err
	}

	// Reshape (B, S, H) → (B, S, NH, HD) → permute (B, NH, S, HD) → reshape
	// to (B*NH, S, HD) so the batched matmul stays rank-3.
	qP, err := reshapeAndPermuteForMHA(q, B, S, NH, HD)
	if err != nil {
		return nil, err
	}
	kP, err := reshapeAndPermuteForMHA(k, B, S, NH, HD)
	if err != nil {
		return nil, err
	}
	vP, err := reshapeAndPermuteForMHA(v, B, S, NH, HD)
	if err != nil {
		return nil, err
	}

	// kPT: (B*NH, HD, S) via last-two-axis transpose + contiguify.
	kPT, err := autograd.Transpose(kP, 1, 2)
	if err != nil {
		return nil, err
	}
	kPTc, err := autograd.Contiguous(kPT)
	if err != nil {
		return nil, err
	}

	// scores: (B*NH, S, S)
	scores, err := autograd.MatMul(qP, kPTc)
	if err != nil {
		return nil, err
	}
	scaled, err := autograd.Mul(scores, m.scale)
	if err != nil {
		return nil, err
	}
	if m.causal {
		mask, err := m.causalMask(S, x.Device())
		if err != nil {
			return nil, err
		}
		// (S, S) broadcasts to (B*NH, S, S).
		scaled, err = autograd.Add(scaled, mask)
		if err != nil {
			return nil, err
		}
	}
	attn, err := autograd.Softmax(scaled, -1)
	if err != nil {
		return nil, err
	}
	ctx, err := autograd.MatMul(attn, vP) // (B*NH, S, HD)
	if err != nil {
		return nil, err
	}

	// Un-collapse: (B*NH, S, HD) → (B, NH, S, HD) → permute (B, S, NH, HD)
	// → reshape (B, S, H).
	ctxU, err := autograd.Reshape(ctx, B, NH, S, HD)
	if err != nil {
		return nil, err
	}
	ctxP, err := autograd.Permute(ctxU, 0, 2, 1, 3)
	if err != nil {
		return nil, err
	}
	ctxR, err := autograd.Reshape(ctxP, B, S, m.hidden)
	if err != nil {
		return nil, err
	}
	out, err := m.OProj.Forward(ctxR)
	if err != nil {
		return nil, err
	}
	if unbatched {
		return autograd.Reshape(out, S, m.hidden)
	}
	return out, nil
}

// Parameters returns Q/K/V/O linear parameters.
func (m *MultiHeadAttention) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, m.QProj.Parameters()...)
	ps = append(ps, m.KProj.Parameters()...)
	ps = append(ps, m.VProj.Parameters()...)
	ps = append(ps, m.OProj.Parameters()...)
	return ps
}

// NumHeads returns the number of attention heads.
func (m *MultiHeadAttention) NumHeads() int { return m.numHeads }

// HeadDim returns the per-head dimension (hidden / numHeads).
func (m *MultiHeadAttention) HeadDim() int { return m.headDim }

// Hidden returns the model width.
func (m *MultiHeadAttention) Hidden() int { return m.hidden }

// Causal reports whether causal masking is enabled.
func (m *MultiHeadAttention) Causal() bool { return m.causal }

// reshapeAndPermuteForMHA takes (B, S, H) and returns (B*NH, S, HD) with
// the batch and head dims collapsed so the batched matmul stays rank-3.
func reshapeAndPermuteForMHA(t *tensor.Tensor, B, S, NH, HD int) (*tensor.Tensor, error) {
	r, err := autograd.Reshape(t, B, S, NH, HD)
	if err != nil {
		return nil, err
	}
	p, err := autograd.Permute(r, 0, 2, 1, 3)
	if err != nil {
		return nil, err
	}
	c, err := autograd.Contiguous(p)
	if err != nil {
		return nil, err
	}
	return autograd.Reshape(c, B*NH, S, HD)
}

func (m *MultiHeadAttention) causalMask(s int, dev device.Device) (*tensor.Tensor, error) {
	if m.maskCache != nil && m.maskLen == s && m.maskDev != nil && m.maskDev.SameAs(dev) {
		return m.maskCache, nil
	}
	data := make([]float32, s*s)
	negInf := float32(math.Inf(-1))
	for i := range s {
		for j := i + 1; j < s; j++ {
			data[i*s+j] = negInf
		}
	}
	mask, err := tensor.New(data, []int{s, s}, dev)
	if err != nil {
		return nil, err
	}
	m.maskCache = mask
	m.maskDev = dev
	m.maskLen = s
	return mask, nil
}
