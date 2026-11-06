package nn

import (
	"fmt"
	"math"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// CrossAttention implements multi-head attention where queries come from one
// stream (typically the latents in a diffusion U-Net) and keys/values come
// from another (the text-encoder hidden states). Stable Diffusion's
// `BasicTransformerBlock` interleaves a self-attention CrossAttention(x, x)
// and a cross-attention CrossAttention(x, ctx) — same module, different
// `ctx` argument.
//
// `q_dim` is the query/output width; `kv_dim` is the context width
// (CLIP hidden size for SD 1.5 = 768). Internally projects to NumHeads ×
// HeadDim with HeadDim = q_dim / NumHeads. No bias on Q/K/V/O (HF SD).
//
// Shapes:
//
//	x:   (B, T_q, q_dim)
//	ctx: (B, T_kv, kv_dim)   — same B; T_kv may differ from T_q
//	out: (B, T_q, q_dim)
//
// No causal mask (cross-attn is acausal). The same kernel works for the
// self-attention case when callers pass `ctx == x`.
type CrossAttention struct {
	QProj, KProj, VProj, OProj *Linear
	numHeads                   int
	headDim                    int
	qDim                       int
	kvDim                      int
	scale                      *tensor.Tensor // (1,) F32 = 1/sqrt(HeadDim)
}

// NewCrossAttention constructs the module with F32 weights. See
// NewCrossAttentionWithDType for BF16 / mixed-precision construction.
func NewCrossAttention(qDim, kvDim, numHeads int, dev device.Device) (*CrossAttention, error) {
	return NewCrossAttentionWithDType(qDim, kvDim, numHeads, dtype.F32, dev)
}

// NewCrossAttentionWithDType constructs the module with weights stored at
// `wdt`. `qDim` must be divisible by `numHeads`. Q/K/V have no bias (HF SD
// convention), but the output projection (`to_out.0` in HF) DOES carry a
// bias — silently dropping it produces visible quality regressions in
// generated images.
func NewCrossAttentionWithDType(qDim, kvDim, numHeads int, wdt dtype.DType, dev device.Device) (*CrossAttention, error) {
	if qDim <= 0 || kvDim <= 0 || numHeads <= 0 {
		return nil, fmt.Errorf("nn.CrossAttention: dims and heads must be positive")
	}
	if qDim%numHeads != 0 {
		return nil, fmt.Errorf("nn.CrossAttention: qDim (%d) not divisible by numHeads (%d)", qDim, numHeads)
	}
	headDim := qDim / numHeads
	q, err := NewLinearUninit(qDim, qDim, false, wdt, dev)
	if err != nil {
		return nil, err
	}
	k, err := NewLinearUninit(kvDim, qDim, false, wdt, dev)
	if err != nil {
		return nil, err
	}
	v, err := NewLinearUninit(kvDim, qDim, false, wdt, dev)
	if err != nil {
		return nil, err
	}
	o, err := NewLinearUninit(qDim, qDim, true, wdt, dev) // bias=true: HF's to_out.0 has bias
	if err != nil {
		return nil, err
	}
	scaleVal := float32(1.0 / math.Sqrt(float64(headDim)))
	scaleT, err := tensor.New([]float32{scaleVal}, []int{1}, dev)
	if err != nil {
		return nil, err
	}
	// Scale must match the activation/weight dtype — otherwise the
	// `Mul(scores, scale)` later either errors on dtype mismatch or, worse,
	// silently produces garbage. Cast eagerly so the runtime path is
	// dtype-uniform.
	if wdt != dtype.F32 {
		scaleT, err = ops.Cast(scaleT, wdt)
		if err != nil {
			return nil, err
		}
	}
	return &CrossAttention{
		QProj: q, KProj: k, VProj: v, OProj: o,
		numHeads: numHeads, headDim: headDim, qDim: qDim, kvDim: kvDim,
		scale: scaleT,
	}, nil
}

// Forward runs cross-attention. `x` and `ctx` may have different sequence
// lengths but must share the batch dim and the activation dtype.
func (a *CrossAttention) Forward(x, ctx *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Rank() != 3 || ctx.Rank() != 3 {
		return nil, fmt.Errorf("nn.CrossAttention: expected rank-3 (B, T, D), got %v / %v", x.Shape(), ctx.Shape())
	}
	if x.Dim(0) != ctx.Dim(0) {
		return nil, fmt.Errorf("nn.CrossAttention: batch mismatch (%d vs %d)", x.Dim(0), ctx.Dim(0))
	}
	if x.Dim(-1) != a.qDim {
		return nil, fmt.Errorf("nn.CrossAttention: x last dim %d != qDim %d", x.Dim(-1), a.qDim)
	}
	if ctx.Dim(-1) != a.kvDim {
		return nil, fmt.Errorf("nn.CrossAttention: ctx last dim %d != kvDim %d", ctx.Dim(-1), a.kvDim)
	}
	B := x.Dim(0)
	Tq := x.Dim(1)
	Tkv := ctx.Dim(1)
	NH := a.numHeads
	HD := a.headDim

	q, err := a.QProj.Forward(x) // (B, Tq, qDim)
	if err != nil {
		return nil, err
	}
	k, err := a.KProj.Forward(ctx) // (B, Tkv, qDim)
	if err != nil {
		return nil, err
	}
	v, err := a.VProj.Forward(ctx) // (B, Tkv, qDim)
	if err != nil {
		return nil, err
	}
	qP, err := reshapeAndPermuteForMHATq(q, B, Tq, NH, HD)
	if err != nil {
		return nil, err
	}
	kP, err := reshapeAndPermuteForMHATq(k, B, Tkv, NH, HD)
	if err != nil {
		return nil, err
	}
	vP, err := reshapeAndPermuteForMHATq(v, B, Tkv, NH, HD)
	if err != nil {
		return nil, err
	}
	kPT, err := autograd.Transpose(kP, 1, 2)
	if err != nil {
		return nil, err
	}
	kPTc, err := autograd.Contiguous(kPT)
	if err != nil {
		return nil, err
	}
	scores, err := autograd.MatMul(qP, kPTc) // (B*NH, Tq, Tkv)
	if err != nil {
		return nil, err
	}
	scaled, err := autograd.Mul(scores, a.scale)
	if err != nil {
		return nil, err
	}
	attn, err := autograd.Softmax(scaled, -1)
	if err != nil {
		return nil, err
	}
	c, err := autograd.MatMul(attn, vP) // (B*NH, Tq, HD)
	if err != nil {
		return nil, err
	}
	cU, err := autograd.Reshape(c, B, NH, Tq, HD)
	if err != nil {
		return nil, err
	}
	cP, err := autograd.Permute(cU, 0, 2, 1, 3)
	if err != nil {
		return nil, err
	}
	cR, err := autograd.Contiguous(cP)
	if err != nil {
		return nil, err
	}
	cR, err = autograd.Reshape(cR, B, Tq, a.qDim)
	if err != nil {
		return nil, err
	}
	return a.OProj.Forward(cR)
}

// Parameters returns Q/K/V/O linear parameters.
func (a *CrossAttention) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, a.QProj.Parameters()...)
	ps = append(ps, a.KProj.Parameters()...)
	ps = append(ps, a.VProj.Parameters()...)
	ps = append(ps, a.OProj.Parameters()...)
	return ps
}

// reshapeAndPermuteForMHATq is a copy of reshapeAndPermuteForMHA that
// accepts an explicit T (sequence length) rather than reading it from the
// shape. CrossAttention's Q and K/V have different T values, so we need to
// pass them explicitly per call.
func reshapeAndPermuteForMHATq(t *tensor.Tensor, B, T, NH, HD int) (*tensor.Tensor, error) {
	r, err := autograd.Reshape(t, B, T, NH, HD)
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
	return autograd.Reshape(c, B*NH, T, HD)
}

// silence unused-import warnings when the dtype package isn't directly
// referenced (kept for future BF16 expansions of cross-attention).
var _ = dtype.F32
