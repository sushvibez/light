package ops

import (
	"fmt"

	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// GatedDeltaMode picks how GatedDeltaRule walks the sequence.
type GatedDeltaMode int

const (
	// GatedDeltaAuto uses the recurrent form for a single token (decode) and
	// the chunked form otherwise (prefill).
	GatedDeltaAuto GatedDeltaMode = iota
	// GatedDeltaRecurrent updates the state token by token.
	GatedDeltaRecurrent
	// GatedDeltaChunked processes ChunkSize tokens at a time: an intra-chunk
	// unit-lower-triangular solve plus the recurrent carry between chunks.
	GatedDeltaChunked
)

// GatedDeltaOptions configures GatedDeltaRule.
type GatedDeltaOptions struct {
	Mode      GatedDeltaMode
	ChunkSize int  // chunked form block length; 0 = 64 (transformers' default)
	QKL2Norm  bool // L2-normalise q and k over the head dim first (eps 1e-6); Qwen3.5 / Qwen3-Next set it
}

// GatedDeltaRule is the gated delta rule of Gated DeltaNet linear attention
// (Qwen3-Next, Qwen3.5 / 3.6 / 3.8), transformers'
// torch_recurrent_gated_delta_rule / torch_chunk_gated_delta_rule. Per batch
// and head, with a state S of shape (dk, dv), for every token t:
//
//	S ← exp(g_t) · S
//	S ← S + k_t ⊗ β_t (v_t − Sᵀ k_t)
//	o_t = Sᵀ (q_t / √dk)
//
// Shapes (all F32, the precision the reference runs this in whatever the
// model dtype):
//
//	q, k  (B, T, H, dk)   query / key heads, already repeated to H value heads
//	v     (B, T, H, dv)
//	g     (B, T, H)       log-space decay, ≤ 0
//	beta  (B, T, H)       write strength in (0, 1)
//	state (B, H, dk, dv)  initial state, or nil for zeros
//
// It returns the output (B, T, H, dv) and the final state (B, H, dk, dv),
// which is the initial state for the next call (decode after prefill).
func GatedDeltaRule(q, k, v, g, beta, state *tensor.Tensor, opts GatedDeltaOptions) (out, final *tensor.Tensor, err error) {
	if q.Rank() != 4 || k.Rank() != 4 || v.Rank() != 4 {
		return nil, nil, fmt.Errorf("GatedDeltaRule: q, k, v must be rank 4 (B, T, H, d), got %v %v %v", q.Shape(), k.Shape(), v.Shape())
	}
	B, T, H, dk := q.Dim(0), q.Dim(1), q.Dim(2), q.Dim(3)
	dv := v.Dim(3)
	if !k.Shape().Equal(q.Shape()) || v.Dim(0) != B || v.Dim(1) != T || v.Dim(2) != H {
		return nil, nil, fmt.Errorf("GatedDeltaRule: shape mismatch q %v k %v v %v", q.Shape(), k.Shape(), v.Shape())
	}
	for name, t := range map[string]*tensor.Tensor{"g": g, "beta": beta} {
		if t.Rank() != 3 || t.Dim(0) != B || t.Dim(1) != T || t.Dim(2) != H {
			return nil, nil, fmt.Errorf("GatedDeltaRule: %s must be (B, T, H) = (%d, %d, %d), got %v", name, B, T, H, t.Shape())
		}
	}
	if state != nil && (state.Rank() != 4 || state.Dim(0) != B || state.Dim(1) != H || state.Dim(2) != dk || state.Dim(3) != dv) {
		return nil, nil, fmt.Errorf("GatedDeltaRule: state must be (B, H, dk, dv) = (%d, %d, %d, %d), got %v", B, H, dk, dv, state.Shape())
	}
	dev := q.Device()
	ins := []*tensor.Tensor{q, k, v, g, beta}
	if state != nil {
		ins = append(ins, state)
	}
	for i, t := range ins {
		if t.DType() != dtype.F32 {
			return nil, nil, fmt.Errorf("GatedDeltaRule: input %d is %v, want F32", i, t.DType())
		}
		if !t.Device().SameAs(dev) {
			return nil, nil, fmt.Errorf("GatedDeltaRule: input %d on %v, want %v", i, t.Device(), dev)
		}
		if ins[i], err = ensureContig(t); err != nil {
			return nil, nil, err
		}
	}
	if state == nil {
		zero, err := tensor.Zeros(dtype.F32, []int{B, H, dk, dv}, dev)
		if err != nil {
			return nil, nil, err
		}
		ins = append(ins, zero)
	}
	chunk := opts.ChunkSize
	if chunk <= 0 {
		chunk = 64
	}
	chunked := opts.Mode == GatedDeltaChunked || (opts.Mode == GatedDeltaAuto && T > 1)

	if out, err = tensor.Zeros(dtype.F32, []int{B, T, H, dv}, dev); err != nil {
		return nil, nil, err
	}
	if final, err = tensor.Zeros(dtype.F32, []int{B, H, dk, dv}, dev); err != nil {
		return nil, nil, err
	}
	kern, err := registry.Lookup(registry.Key{Op: "gated_delta_rule", Dev: dev.Kind(), DType: dtype.F32})
	if err != nil {
		return nil, nil, err
	}
	// Inputs: q, k, v, g, beta, initial state, final state (written).
	if err := kern(&registry.Context{
		Inputs: append(ins, final),
		Output: out,
		Attrs:  map[string]any{"chunked": chunked, "chunk": chunk, "l2norm": opts.QKL2Norm},
	}); err != nil {
		return nil, nil, err
	}
	return out, final, nil
}

// CausalConv1dDepthwise is the short causal depthwise convolution in front
// of Gated DeltaNet (transformers' causal_conv1d_fn / causal_conv1d_update):
// channel c of the output at time t is Σ_j w[c, j] · x[c, t − K + 1 + j],
// optionally followed by SiLU. Inputs before t = 0 come from state, the last
// K−1 inputs of the previous call (zeros when state is nil), so a sequence
// can be fed in pieces (prefill, then one token at a time).
//
//	x     (B, C, T)   F32
//	w     (C, K)      F32 (the HF conv1d.weight (C, 1, K) squeezed)
//	state (B, C, K−1) F32 or nil
//
// It returns the output (B, C, T) and the new state (B, C, K−1).
func CausalConv1dDepthwise(x, w, state *tensor.Tensor, silu bool) (out, newState *tensor.Tensor, err error) {
	if x.Rank() != 3 || w.Rank() != 2 || w.Dim(0) != x.Dim(1) {
		return nil, nil, fmt.Errorf("CausalConv1dDepthwise: want x (B, C, T) and w (C, K), got %v %v", x.Shape(), w.Shape())
	}
	B, C, K := x.Dim(0), x.Dim(1), w.Dim(1)
	if K < 1 {
		return nil, nil, fmt.Errorf("CausalConv1dDepthwise: kernel size must be ≥ 1")
	}
	if state != nil && (state.Rank() != 3 || state.Dim(0) != B || state.Dim(1) != C || state.Dim(2) != K-1) {
		return nil, nil, fmt.Errorf("CausalConv1dDepthwise: state must be (B, C, K-1) = (%d, %d, %d), got %v", B, C, K-1, state.Shape())
	}
	dev := x.Device()
	ins := []*tensor.Tensor{x, w}
	if state != nil {
		ins = append(ins, state)
	}
	for i, t := range ins {
		if t.DType() != dtype.F32 {
			return nil, nil, fmt.Errorf("CausalConv1dDepthwise: input %d is %v, want F32", i, t.DType())
		}
		if !t.Device().SameAs(dev) {
			return nil, nil, fmt.Errorf("CausalConv1dDepthwise: input %d on %v, want %v", i, t.Device(), dev)
		}
		if ins[i], err = ensureContig(t); err != nil {
			return nil, nil, err
		}
	}
	if state == nil {
		zero, err := tensor.Zeros(dtype.F32, []int{B, C, K - 1}, dev)
		if err != nil {
			return nil, nil, err
		}
		ins = append(ins, zero)
	}
	if out, err = tensor.Zeros(dtype.F32, x.Shape().Dims(), dev); err != nil {
		return nil, nil, err
	}
	if newState, err = tensor.Zeros(dtype.F32, []int{B, C, K - 1}, dev); err != nil {
		return nil, nil, err
	}
	kern, err := registry.Lookup(registry.Key{Op: "causal_conv1d_depthwise", Dev: dev.Kind(), DType: dtype.F32})
	if err != nil {
		return nil, nil, err
	}
	// Inputs: x, w, state, new state (written).
	if err := kern(&registry.Context{
		Inputs: append(ins, newState),
		Output: out,
		Attrs:  map[string]any{"silu": silu},
	}); err != nil {
		return nil, nil, err
	}
	return out, newState, nil
}
