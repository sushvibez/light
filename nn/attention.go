package nn

import (
	"fmt"
	"math"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/tensor"
)

// SelfAttention is the simplest form of transformer self-attention: one head,
// optional causal masking, no dropout. Built entirely out of existing ops
// (Linear, MatMul, Softmax, Add, Mul) — no new CUDA kernels required.
//
// Forward(x) where x: (seqLen, hidden):
//
//	q = QProj(x); k = KProj(x); v = VProj(x)
//	scores = q · kᵀ / √hidden          → (S, S)
//	(+ causal mask if enabled)
//	attn = softmax(scores, axis=-1)   → (S, S)
//	ctx  = attn · v                    → (S, H)
//	out  = OProj(ctx)                  → (S, H)
//
// Multi-head attention is layered on top in a later phase (needs Narrow + a
// batched matmul or Concat).
type SelfAttention struct {
	QProj, KProj, VProj, OProj *Linear

	hidden    int
	scale     *tensor.Tensor // (1,) scalar = 1/√hidden on the correct device
	causal    bool
	maskCache *tensor.Tensor // cached causal mask for the most recent seq-len
	maskDev   device.Device
	maskLen   int
}

// NewSelfAttention constructs a module with 4 Linear layers (Q/K/V/O).
// Weights get Kaiming-uniform init; biases are omitted (matches LLaMA).
func NewSelfAttention(hidden int, causal bool, dev device.Device) (*SelfAttention, error) {
	if hidden <= 0 {
		return nil, fmt.Errorf("nn.SelfAttention: hidden must be positive (got %d)", hidden)
	}
	q, err := NewLinear(hidden, hidden, false, dev)
	if err != nil {
		return nil, err
	}
	k, err := NewLinear(hidden, hidden, false, dev)
	if err != nil {
		return nil, err
	}
	v, err := NewLinear(hidden, hidden, false, dev)
	if err != nil {
		return nil, err
	}
	o, err := NewLinear(hidden, hidden, false, dev)
	if err != nil {
		return nil, err
	}
	scaleVal := float32(1.0 / math.Sqrt(float64(hidden)))
	scale, err := tensor.New([]float32{scaleVal}, []int{1}, dev)
	if err != nil {
		return nil, err
	}
	return &SelfAttention{
		QProj: q, KProj: k, VProj: v, OProj: o,
		hidden: hidden,
		scale:  scale,
		causal: causal,
	}, nil
}

// Forward runs self-attention on x of shape (..., seqLen, hidden). Rank must
// be ≥ 2. Any leading dims are treated as batch axes and preserved through
// the attention block.
func (a *SelfAttention) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Rank() < 2 || x.Dim(-1) != a.hidden {
		return nil, fmt.Errorf("nn.SelfAttention: expected (..., S, %d), got %v", a.hidden, x.Shape())
	}
	r := x.Rank()
	s := x.Dim(-2)

	q, err := a.QProj.Forward(x)
	if err != nil {
		return nil, err
	}
	k, err := a.KProj.Forward(x)
	if err != nil {
		return nil, err
	}
	v, err := a.VProj.Forward(x)
	if err != nil {
		return nil, err
	}

	// Transpose last two axes of K: (..., S, H) → (..., H, S).
	kT, err := autograd.Transpose(k, r-2, r-1)
	if err != nil {
		return nil, err
	}
	// For rank-2, autograd.MatMul is 2-D. For rank ≥ 3, batched matmul.
	// Both paths materialise transposed views via autograd.Contiguous so
	// the kernel sees default strides.
	kTc, err := autograd.Contiguous(kT)
	if err != nil {
		return nil, err
	}
	qc, err := autograd.Contiguous(q)
	if err != nil {
		return nil, err
	}
	scores, err := autograd.MatMul(qc, kTc) // (..., S, S)
	if err != nil {
		return nil, err
	}
	scaled, err := autograd.Mul(scores, a.scale)
	if err != nil {
		return nil, err
	}

	if a.causal {
		mask, err := a.causalMask(s, x.Device())
		if err != nil {
			return nil, err
		}
		scaled, err = autograd.Add(scaled, mask)
		if err != nil {
			return nil, err
		}
	}

	attn, err := autograd.Softmax(scaled, -1)
	if err != nil {
		return nil, err
	}
	vc, err := autograd.Contiguous(v)
	if err != nil {
		return nil, err
	}
	ctx, err := autograd.MatMul(attn, vc) // (..., S, H)
	if err != nil {
		return nil, err
	}
	return a.OProj.Forward(ctx)
}

// Parameters returns the concatenated parameters of Q/K/V/O projections.
func (a *SelfAttention) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, a.QProj.Parameters()...)
	ps = append(ps, a.KProj.Parameters()...)
	ps = append(ps, a.VProj.Parameters()...)
	ps = append(ps, a.OProj.Parameters()...)
	return ps
}

// causalMask returns (and caches) an (S, S) tensor with 0 at (i, j) for j <= i
// and -∞ for j > i — added to the pre-softmax scores to zero out future tokens.
func (a *SelfAttention) causalMask(s int, dev device.Device) (*tensor.Tensor, error) {
	if a.maskCache != nil && a.maskLen == s && a.maskDev != nil && a.maskDev.SameAs(dev) {
		return a.maskCache, nil
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
	a.maskCache = mask
	a.maskDev = dev
	a.maskLen = s
	return mask, nil
}

// Hidden returns the hidden dimension.
func (a *SelfAttention) Hidden() int { return a.hidden }

// Causal reports whether this attention masks future positions.
func (a *SelfAttention) Causal() bool { return a.causal }

// Ensure dtype stays imported in case future changes need it directly.
var _ = dtype.F32
