// Package ops — RoPE (Rotary Position Embedding).
//
// For a tensor x with shape (..., seqLen, headDim) where headDim is even,
// RoPE rotates the feature pair (x[..., i], x[..., i+half]), half = headDim/2,
// by an angle θ_{i,pos} = pos * base^(-2i / headDim) for i in [0, half):
//
//	x'[..., i]      =  x[..., i]      * cos(θ) - x[..., i+half] * sin(θ)
//	x'[..., i+half] =  x[..., i]      * sin(θ) + x[..., i+half] * cos(θ)
//
// This is the "rotate-half" (GPT-NeoX) convention used by HuggingFace's
// Llama, Qwen, Mistral, Phi and Gemma implementations, i.e. what their
// checkpoints were trained with. GPT-J style interleaved pairs (2i, 2i+1)
// are a different, incompatible convention, available as RoPEInterleaved
// (Candle `rope_i`); until 2026-10-27 RoPE itself used interleaved pairs,
// which made every Llama-family model subtly wrong (top-1 often survived,
// logits did not).
// Dispatches to CPU or CUDA via the op registry.
package ops

import (
	"fmt"

	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// RoPE applies rotary position embeddings to the last axis of x. x:
// (..., seqLen, headDim) where headDim must be even; positions: (seqLen,) I64.
func RoPE(x *tensor.Tensor, positions *tensor.Tensor, base float64) (*tensor.Tensor, error) {
	if x.Rank() < 2 {
		return nil, fmt.Errorf("RoPE: x rank must be >= 2 (got %d)", x.Rank())
	}
	if x.DType() != dtype.F32 && x.DType() != dtype.F64 && x.DType() != dtype.BF16 {
		return nil, fmt.Errorf("RoPE: dtype %v unsupported (F32/F64/BF16)", x.DType())
	}
	if positions.DType() != dtype.I64 || positions.Rank() != 1 {
		return nil, fmt.Errorf("RoPE: positions must be 1-D I64")
	}
	headDim := x.Dim(-1)
	seqLen := x.Dim(-2)
	if headDim%2 != 0 {
		return nil, fmt.Errorf("RoPE: headDim must be even (got %d)", headDim)
	}
	if positions.Dim(0) != seqLen {
		return nil, fmt.Errorf("RoPE: positions length %d != seqLen %d", positions.Dim(0), seqLen)
	}
	src, err := ensureContig(x)
	if err != nil {
		return nil, err
	}
	posC, err := ensureContig(positions)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(x.DType(), x.Shape().Dims(), x.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "rope", Dev: x.Device().Kind(), DType: x.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{src, posC},
		Output: out,
		Attrs:  map[string]any{"base": base},
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// RoPEWithInvFreq is RoPE with an explicit per-pair inverse-frequency table:
// θ_{i,pos} = pos * invFreq[i]. invFreq must be a (headDim/2,) F32 tensor on
// x's device. Passing base^(-2i/headDim) values reproduces RoPE(x, pos, base);
// the point of this variant is frequency scaling (Llama 3 "llama3" scaling,
// YaRN, linear) computed once on the host and reused for every layer.
func RoPEWithInvFreq(x, positions, invFreq *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Rank() < 2 {
		return nil, fmt.Errorf("RoPEWithInvFreq: x rank must be >= 2 (got %d)", x.Rank())
	}
	if x.DType() != dtype.F32 && x.DType() != dtype.F64 && x.DType() != dtype.BF16 {
		return nil, fmt.Errorf("RoPEWithInvFreq: dtype %v unsupported (F32/F64/BF16)", x.DType())
	}
	if positions.DType() != dtype.I64 || positions.Rank() != 1 {
		return nil, fmt.Errorf("RoPEWithInvFreq: positions must be 1-D I64")
	}
	headDim := x.Dim(-1)
	seqLen := x.Dim(-2)
	if headDim%2 != 0 {
		return nil, fmt.Errorf("RoPEWithInvFreq: headDim must be even (got %d)", headDim)
	}
	if positions.Dim(0) != seqLen {
		return nil, fmt.Errorf("RoPEWithInvFreq: positions length %d != seqLen %d", positions.Dim(0), seqLen)
	}
	if invFreq.DType() != dtype.F32 || invFreq.Rank() != 1 || invFreq.Dim(0) != headDim/2 {
		return nil, fmt.Errorf("RoPEWithInvFreq: invFreq must be (headDim/2=%d,) F32, got %v %v", headDim/2, invFreq.Shape(), invFreq.DType())
	}
	if !invFreq.Device().SameAs(x.Device()) {
		return nil, fmt.Errorf("RoPEWithInvFreq: invFreq on %v, x on %v", invFreq.Device(), x.Device())
	}
	src, err := ensureContig(x)
	if err != nil {
		return nil, err
	}
	posC, err := ensureContig(positions)
	if err != nil {
		return nil, err
	}
	freqC, err := ensureContig(invFreq)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(x.DType(), x.Shape().Dims(), x.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "rope_scaled", Dev: x.Device().Kind(), DType: x.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{src, posC, freqC},
		Output: out,
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// RoPEInterleaved applies rotary embeddings with GPT-J interleaved pairs
// (dims 2i and 2i+1 rotate together; Candle `rope_i`), the convention of
// GPT-J, CodeGen and Phi-1/2 checkpoints. Arguments as RoPE.
func RoPEInterleaved(x, positions *tensor.Tensor, base float64) (*tensor.Tensor, error) {
	if x.Rank() < 2 {
		return nil, fmt.Errorf("RoPEInterleaved: x rank must be >= 2 (got %d)", x.Rank())
	}
	if x.DType() != dtype.F32 && x.DType() != dtype.F64 && x.DType() != dtype.BF16 {
		return nil, fmt.Errorf("RoPEInterleaved: dtype %v unsupported (F32/F64/BF16)", x.DType())
	}
	if positions.DType() != dtype.I64 || positions.Rank() != 1 {
		return nil, fmt.Errorf("RoPEInterleaved: positions must be 1-D I64")
	}
	if x.Dim(-1)%2 != 0 {
		return nil, fmt.Errorf("RoPEInterleaved: headDim must be even (got %d)", x.Dim(-1))
	}
	if positions.Dim(0) != x.Dim(-2) {
		return nil, fmt.Errorf("RoPEInterleaved: positions length %d != seqLen %d", positions.Dim(0), x.Dim(-2))
	}
	src, err := ensureContig(x)
	if err != nil {
		return nil, err
	}
	posC, err := ensureContig(positions)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(x.DType(), x.Shape().Dims(), x.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "rope_i", Dev: x.Device().Kind(), DType: x.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{src, posC},
		Output: out,
		Attrs:  map[string]any{"base": base},
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// RoPETHD is rotate-half RoPE on the (batch, seq, heads, headDim) layout
// (Candle `rope_thd`): the tensor is viewed as (batch, heads, seq, headDim)
// for the kernel and returned in the original layout. positions is (seq,).
func RoPETHD(x, positions *tensor.Tensor, base float64) (*tensor.Tensor, error) {
	if x.Rank() != 4 {
		return nil, fmt.Errorf("RoPETHD: x must be (B, T, H, D), got %v", x.Shape())
	}
	bhtd, err := x.Permute(0, 2, 1, 3)
	if err != nil {
		return nil, err
	}
	if bhtd, err = ensureContig(bhtd); err != nil {
		return nil, err
	}
	rot, err := RoPE(bhtd, positions, base)
	if err != nil {
		return nil, err
	}
	back, err := rot.Permute(0, 2, 1, 3)
	if err != nil {
		return nil, err
	}
	return ensureContig(back)
}
