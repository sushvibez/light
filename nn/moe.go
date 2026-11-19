package nn

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/quant"
	"light/tensor"
)

// SparseMoE is a routed mixture-of-experts feed-forward block in the
// Mixtral / Qwen3-MoE / DeepSeek shape: a router Linear picks TopK experts
// per token (softmax over all experts, optionally renormalised over the
// chosen k), each expert is a gated MLP down(act(gate(x)) ⊙ up(x)), and the
// expert outputs are summed with their routing weights. Expert weights are
// stacked so the whole layer is two grouped GEMMs (ops.MoEGemm) instead of
// a loop over experts:
//
//	GateUp: (E, 2·I, H)  — rows [0, I) are gate_proj, rows [I, 2I) are up_proj
//	Down:   (E, H, I)
//
// Act defaults to SiLU. GateUp / Down may also hold GGUF block-quantized
// experts (E, N, K); then activations must be F32 and ops.MoEGemmQ runs the
// fused dequant kernel. Inference only (no autograd through the routing).
type SparseMoE struct {
	Gate     *Linear // router (E, H), no bias
	GateUp   *tensor.Tensor
	Down     *tensor.Tensor
	TopK     int
	NormTopK bool
	Act      func(*tensor.Tensor) (*tensor.Tensor, error)
	// RouterF32: the router runs on an F32 copy of the input with an F32
	// Gate weight (DeepSeek computes its routing logits in F32); otherwise
	// at the activation dtype, cast to F32 for the softmax.
	RouterF32 bool
	// Router, when set, replaces the softmax + top-k selection: it receives
	// the (T, E) F32 router logits and returns the routing (DeepSeek's
	// group-limited and sigmoid-with-bias routers).
	Router func(logits *tensor.Tensor) (*ops.MoERouting, error)
	// GateUpQ4 / DownQ4 hold 4-bit experts (GPTQ / AWQ MoE checkpoints)
	// instead of GateUp / Down: per expert (2I, H), gate rows then up rows,
	// and (H, I). Forward expands only the routed experts (ops.MoEInt4).
	GateUpQ4, DownQ4 []quant.Int4
	// Stream, when set, holds the experts instead of GateUp / Down: only
	// its slots are resident (see ExpertStream).
	Stream *ExpertStream
	// StreamSlots asks the loader for a Stream with this many slots.
	StreamSlots int
	// Next is the following MoE layer: a streamed layer running a small
	// batch (decode) predicts Next's experts by applying Next's router to
	// its own input and prefetches them while it computes.
	Next *SparseMoE
}

// NewSparseMoEUninit allocates a block with zeroed expert weights at wdt;
// loaders fill GateUp / Down (or call StackExperts) afterwards.
func NewSparseMoEUninit(hidden, intermediate, numExperts, topk int, normTopK bool, wdt dtype.DType, dev device.Device) (*SparseMoE, error) {
	gate, err := NewLinearUninit(hidden, numExperts, false, wdt, dev)
	if err != nil {
		return nil, err
	}
	gu, err := tensor.Zeros(wdt, []int{numExperts, 2 * intermediate, hidden}, dev)
	if err != nil {
		return nil, err
	}
	dn, err := tensor.Zeros(wdt, []int{numExperts, hidden, intermediate}, dev)
	if err != nil {
		return nil, err
	}
	return &SparseMoE{Gate: gate, GateUp: gu, Down: dn, TopK: topk, NormTopK: normTopK}, nil
}

// NewSparseMoELazy allocates only the router placeholder; the loader
// installs GateUp / Down (dense or GGUF-quantized) afterwards. Used by the
// GGUF path so a 30B expert stack is never materialised at F32.
func NewSparseMoELazy(hidden, numExperts, topk int, normTopK bool, dev device.Device) (*SparseMoE, error) {
	gate, err := NewLinearPlaceholder(hidden, numExperts, false, dev)
	if err != nil {
		return nil, err
	}
	return &SparseMoE{Gate: gate, TopK: topk, NormTopK: normTopK}, nil
}

// NewSparseMoEStreamed allocates the router only; the loader attaches an
// ExpertStream with `slots` resident experts (StreamSlots) instead of
// GateUp / Down.
func NewSparseMoEStreamed(hidden, numExperts, topk int, normTopK bool, slots int, wdt dtype.DType, dev device.Device) (*SparseMoE, error) {
	gate, err := NewLinearUninit(hidden, numExperts, false, wdt, dev)
	if err != nil {
		return nil, err
	}
	return &SparseMoE{Gate: gate, TopK: topk, NormTopK: normTopK, StreamSlots: slots}, nil
}

// StackExperts builds GateUp / Down from per-expert (I, H) gate, (I, H) up
// and (H, I) down matrices (the HF checkpoint layout).
func (m *SparseMoE) StackExperts(gate, up, down []*tensor.Tensor) error {
	E := len(gate)
	if E == 0 || len(up) != E || len(down) != E {
		return fmt.Errorf("nn.SparseMoE: need equal non-empty gate/up/down lists")
	}
	gu := make([]*tensor.Tensor, E)
	for e := range E {
		cat, err := ops.Concat([]*tensor.Tensor{gate[e], up[e]}, 0)
		if err != nil {
			return err
		}
		gu[e] = cat
	}
	var err error
	if m.GateUp, err = ops.Stack(gu, 0); err != nil {
		return err
	}
	m.Down, err = ops.Stack(down, 0)
	return err
}

// Forward maps (…, H) → (…, H).
func (m *SparseMoE) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	H := m.Gate.InFeatures()
	if x.Dim(-1) != H {
		return nil, fmt.Errorf("nn.SparseMoE: last dim %d != hidden %d", x.Dim(-1), H)
	}
	dims := x.Shape().Dims()
	xc, err := x.Contiguous()
	if err != nil {
		return nil, err
	}
	flat, err := xc.Reshape(-1, H)
	if err != nil {
		return nil, err
	}
	T := flat.Dim(0)
	gateIn := flat
	if m.RouterF32 && flat.DType() != dtype.F32 {
		if gateIn, err = ops.Cast(flat, dtype.F32); err != nil {
			return nil, err
		}
	}
	if m.Stream != nil && m.Next != nil && m.Next.Stream != nil && T <= prefetchMaxTokens {
		m.Next.prefetchFor(gateIn)
	}
	route, err := m.route(gateIn)
	if err != nil {
		return nil, err
	}
	if m.GateUpQ4 != nil {
		act := m.Act
		if act == nil {
			act = ops.SiLU
		}
		y, err := ops.MoEInt4(flat, m.GateUpQ4, m.DownQ4, route, m.TopK, act)
		if err != nil {
			return nil, err
		}
		return y.Reshape(dims...)
	}
	var y *tensor.Tensor
	if m.Stream != nil {
		y, err = m.forwardStreamed(flat, route)
	} else {
		y, err = m.runExperts(flat, m.GateUp, m.Down, route)
	}
	if err != nil {
		return nil, err
	}
	_ = T
	return y.Reshape(dims...)
}

// route runs the router on gateIn (T, H): softmax + top-k, or Router.
func (m *SparseMoE) route(gateIn *tensor.Tensor) (*ops.MoERouting, error) {
	logits, err := m.Gate.Forward(gateIn)
	if err != nil {
		return nil, err
	}
	if logits.DType() != dtype.F32 {
		if logits, err = ops.Cast(logits, dtype.F32); err != nil {
			return nil, err
		}
	}
	if m.Router != nil {
		return m.Router(logits)
	}
	probs, err := ops.Softmax(logits, -1)
	if err != nil {
		return nil, err
	}
	return ops.MoERoute(probs, m.TopK, m.NormTopK)
}

// runExperts is the routed SwiGLU on flat (T, H) with stacked expert
// weights gateUp (E, 2I, H) / down (E, H, I): (T, H).
func (m *SparseMoE) runExperts(flat, gateUp, down *tensor.Tensor, route *ops.MoERouting) (*tensor.Tensor, error) {
	I := gateUp.Dim(1) / 2
	quantized := gateUp.DType().IsQuantized()
	if quantized && flat.DType() != dtype.F32 {
		return nil, fmt.Errorf("nn.SparseMoE: quantized experts (%v) need F32 activations, got %v", gateUp.DType(), flat.DType())
	}
	gemm := ops.MoEGemm
	if quantized {
		gemm = ops.MoEGemmQ
	}
	gu, err := gemm(flat, gateUp, route, nil, m.TopK) // (M, 2I)
	if err != nil {
		return nil, err
	}
	g, err := gu.Narrow(1, 0, I)
	if err != nil {
		return nil, err
	}
	u, err := gu.Narrow(1, I, I)
	if err != nil {
		return nil, err
	}
	act := m.Act
	if act == nil {
		act = ops.SiLU
	}
	if g, err = g.Contiguous(); err != nil {
		return nil, err
	}
	if g, err = act(g); err != nil {
		return nil, err
	}
	h, err := ops.Mul(g, u) // (M, I)
	if err != nil {
		return nil, err
	}
	pair, err := gemm(h, down, route, route.Weights, m.TopK) // (M, H), routing-weighted
	if err != nil {
		return nil, err
	}
	return ops.MoECombine(pair, m.TopK) // (T, H)
}
