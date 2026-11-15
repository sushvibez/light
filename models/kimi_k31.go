package models

import (
	"fmt"
	"math"
	"sort"

	"light/device"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// KimiK31 is Kimi K3.1's language model (moonshotai/Kimi-K3.1-Pro, model_type
// kimi_k31; text config kimi_linear), ported from vLLM's implementation
// (vllm/models/kimi_k3/nvidia, there being no modeling code in the repo or in
// transformers): Kimi Delta Attention layers with a full-rank output gate and
// a bounded forget gate, gated DeepSeek Sparse Attention layers (NoPE MLA
// whose keys are restricted to a ReLU indexer's top index_topk, times a
// sigmoid output gate), a dense first MLP then latent MoE layers (grouped
// sigmoid routing with a correction bias, experts run in a down-projected
// latent that is RMS-normed and projected back up, plus shared experts on the
// hidden state), SiTU-GLU activations, and Attention Residuals: each
// sub-layer's input is a softmax mix, per token, of the block snapshots taken
// every attn_res_block_size layers and the running in-block sum.
//
// Forward is a prefill (no KV / recurrent cache); the indexer's top-k is
// selected on the host.
type KimiK31 struct {
	Cfg     KimiK31Config
	Embed   *tensor.Tensor // (V, D)
	Layers  []*kimiK31Layer
	NormW   *tensor.Tensor
	OutResW *tensor.Tensor // output_attn_res: proj ⊙ norm weight
	LMHead  *nn.Linear
	aux     *GLM5 // shared helpers (RMSNorm, l2norm)
	dev     device.Device
}

type kimiK31MLP struct{ Gate, Up, Down *nn.Linear }

type kimiK31Layer struct {
	ID                  int
	InNormW, PostNormW  *tensor.Tensor
	AttnResW, MLPResW   *tensor.Tensor // AttnRes scorers: proj ⊙ norm weight
	KDA                 bool
	Q, K, V, O, B, FA   *nn.Linear
	FB, G               *nn.Linear
	ConvW               *tensor.Tensor // (3·H·hd, kernel)
	DtBias, ONormW      *tensor.Tensor
	ALog                []float32
	QA, QB, KVA, KVB    *nn.Linear
	MO, MG              *nn.Linear
	QANormW, KVANormW   *tensor.Tensor
	IWqB, IWk, IWeights *nn.Linear // DSA indexer (nil: dense causal MLA)
	IKNormW, IKNormB    *tensor.Tensor
	Dense, Shared       *kimiK31MLP
	Experts             []*kimiK31MLP
	RouterW             *tensor.Tensor // (E, D)
	RouterBias          []float32
	LatentDown          *nn.Linear
	LatentUp            *nn.Linear
	LatentNormW         *tensor.Tensor
}

func (m *KimiK31) rms(o *iop, x, w *tensor.Tensor) *tensor.Tensor { return m.aux.rms(o, x, w) }

// glu is down(act(gate·x) · up'(up·x)): SiTU-GLU (vLLM's SituAndMul,
// β·tanh(g/β)·σ(g) times β_lin·tanh(u/β_lin)) or SwiGLU.
func (m *KimiK31) glu(o *iop, e *kimiK31MLP, x *tensor.Tensor) *tensor.Tensor {
	g, u := o.lin(e.Gate, x), o.lin(e.Up, x)
	if m.Cfg.HiddenAct == "situ" {
		beta := 1.0
		if b := m.Cfg.SituBeta; b != nil && *b != 0 {
			beta = *b
		}
		sig := o.un(ops.Sigmoid, g)
		g = o.mul(o.affine(o.un(ops.Tanh, o.affine(g, 1/beta, 0)), beta, 0), sig)
		if lb := m.Cfg.SituLinearBeta; lb != nil && *lb != 0 {
			u = o.affine(o.un(ops.Tanh, o.affine(u, 1 / *lb, 0)), *lb, 0)
		}
	} else {
		g = o.un(ops.SiLU, g)
	}
	return o.lin(e.Down, o.mul(g, u))
}

// attnRes mixes the sources (each (B, S, D); the running sum last) with a
// per-token softmax over w·RMSNorm(source), then RMS-normalises the mix with
// outNorm when set (vLLM's attn_res op).
func (m *KimiK31) attnRes(o *iop, sources []*tensor.Tensor, w, outNorm *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	mixed := sources[len(sources)-1]
	if n := len(sources); n > 1 {
		B, S, D := mixed.Dim(0), mixed.Dim(1), mixed.Dim(2)
		parts := make([]*tensor.Tensor, n)
		for i, s := range sources {
			parts[i] = o.reshape(s, B, S, 1, D)
		}
		stack := o.cat(2, parts...) // (B, S, n, D)
		logits := o.sum(m.rms(o, stack, w), 3)
		p := o.reshape(o.do(func() (*tensor.Tensor, error) { return ops.Softmax(logits, 2) }), B, S, n, 1)
		mixed = o.sum(o.mul(stack, p), 2)
	}
	if outNorm != nil {
		mixed = m.rms(o, mixed, outNorm)
	}
	return mixed
}

// kda is Kimi Delta Attention on x (B, S, D): q / k / v through SiLU short
// convolutions, a per-channel forget gate lower_bound·σ(e^A·(f + dt_bias)),
// the delta rule, then RMSNorm(out)·σ(g_proj x) and o_proj.
func (m *KimiK31) kda(o *iop, ly *kimiK31Layer, x *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	lc := m.Cfg.LinearAttnConfig
	B, S := x.Dim(0), x.Dim(1)
	H, hd := lc.NumHeads, lc.HeadDim
	qkv := H * hd
	mixed := o.cat(2, o.lin(ly.Q, x), o.lin(ly.K, x), o.lin(ly.V, x)) // (B, S, 3qkv)
	conv := o.do(func() (*tensor.Tensor, error) {
		t, err := mixed.Permute(0, 2, 1)
		if err != nil {
			return nil, err
		}
		if t, err = t.Contiguous(); err != nil {
			return nil, err
		}
		out, _, err := ops.CausalConv1dDepthwise(t, ly.ConvW, nil, true)
		return out, err
	})
	conv = o.perm(conv, 0, 2, 1)
	part := func(i int) *tensor.Tensor { return o.reshape(o.narrow(conv, 2, i*qkv, qkv), B, S, H, hd) }
	q := o.affine(m.aux.l2norm(o, part(0), 1e-6), 1/math.Sqrt(float64(hd)), 0)
	k := m.aux.l2norm(o, part(1), 1e-6)
	v := part(2)
	f := o.reshape(o.add(o.lin(ly.FB, o.lin(ly.FA, x)), ly.DtBias), B, S, H, hd)
	a := o.do(func() (*tensor.Tensor, error) {
		e := make([]float32, H)
		for i, al := range ly.ALog {
			e[i] = float32(math.Exp(float64(al)))
		}
		return tensor.New(e, []int{H, 1}, m.dev)
	})
	var g *tensor.Tensor
	if lb := lc.GateLowerBound; lb != nil {
		g = o.affine(o.un(ops.Sigmoid, o.mul(f, a)), *lb, 0)
	} else {
		g = o.mul(o.un(ops.Neg, a), o.un(ops.Softplus, f))
	}
	beta := o.un(ops.Sigmoid, o.lin(ly.B, x))
	if o.err != nil {
		return nil
	}
	core, _, err := ops.KimiDeltaRule(q, k, v, g, beta, nil)
	if err != nil {
		o.err = err
		return nil
	}
	gate := o.reshape(o.lin(ly.G, x), B, S, H, hd)
	out := o.mul(o.do(func() (*tensor.Tensor, error) { return ops.RMSNorm(core, ly.ONormW, 1e-5) }), o.un(ops.Sigmoid, gate))
	return o.lin(ly.O, o.reshape(out, B, S, qkv))
}

// dsaSelect is the indexer's per-query key mask (B, 1, S, S): 0 on the
// index_topk best visible keys by Σ_h relu(q_h·k)·w_h (every visible key
// while there are no more than index_topk), -inf elsewhere; all visible keys
// when the layer has no indexer.
func (m *KimiK31) dsaSelect(o *iop, ly *kimiK31Layer, x, qr *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	B, S := x.Dim(0), x.Dim(1)
	ninf := float32(math.Inf(-1))
	mv := make([]float32, B*S*S)
	for i := range mv {
		if i%S > i/S%S {
			mv[i] = ninf
		}
	}
	if ly.IWqB != nil && S > m.Cfg.DSAConfig.IndexTopk {
		dc := m.Cfg.DSAConfig
		nh, dh, topk := dc.IndexNumHeads, dc.IndexHeadDim, dc.IndexTopk
		iq := o.reshape(o.lin(ly.IWqB, qr), B, S*nh, dh)
		ik := o.lin(ly.IWk, x)
		if ly.IKNormB != nil {
			ik = o.ln(ik, ly.IKNormW, ly.IKNormB, m.Cfg.RMSNormEps)
		} else {
			ik = m.rms(o, ik, ly.IKNormW)
		}
		ikT := o.do(func() (*tensor.Tensor, error) {
			t, err := ik.Transpose(1, 2)
			if err != nil {
				return nil, err
			}
			return t.Contiguous()
		})
		w := o.affine(o.lin(ly.IWeights, x), 1/math.Sqrt(float64(dh))/math.Sqrt(float64(nh)), 0)
		sc := o.reshape(o.un(ops.Relu, o.matmul(iq, ikT)), B, S, nh, S)
		sv := o.toHost(o.sum(o.mul(sc, o.reshape(w, B, S, nh, 1)), 2)) // (B, S, S)
		if o.err != nil {
			return nil
		}
		order := make([]int, S)
		for r := range B * S {
			t := r % S
			if t < topk {
				continue
			}
			row := sv[r*S : r*S+t+1]
			order = order[:t+1]
			for i := range order {
				order[i] = i
			}
			sort.SliceStable(order, func(a, b int) bool { return row[order[a]] > row[order[b]] })
			for _, j := range order[topk:] {
				mv[r*S+j] = ninf
			}
		}
	}
	return o.do(func() (*tensor.Tensor, error) { return tensor.New(mv, []int{B, 1, S, S}, m.dev) })
}

// mla is the gated (DSA) multi-head latent attention on x (B, S, D): NoPE,
// the qk_rope_head_dim key part shared by every head, unrotated.
func (m *KimiK31) mla(o *iop, ly *kimiK31Layer, x *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	cfg := m.Cfg
	B, S := x.Dim(0), x.Dim(1)
	H, nope, rope, dv := cfg.NumHeads, cfg.QKNopeHeadDim, cfg.QKRopeHeadDim, cfg.VHeadDim
	dk := nope + rope
	qr := m.rms(o, o.lin(ly.QA, x), ly.QANormW)
	q := o.perm(o.reshape(o.lin(ly.QB, qr), B, S, H, dk), 0, 2, 1, 3) // (B, H, S, dk)
	kva := o.lin(ly.KVA, x)
	kvc := m.rms(o, o.narrow(kva, 2, 0, cfg.KVLoraRank), ly.KVANormW)
	kv := o.reshape(o.lin(ly.KVB, kvc), B, S, H, nope+dv)
	k := o.narrow(kv, 3, 0, nope)
	if rope > 0 {
		pe := o.reshape(o.narrow(kva, 2, cfg.KVLoraRank, rope), B, S, 1, rope)
		copies := make([]*tensor.Tensor, H)
		for i := range copies {
			copies[i] = pe
		}
		k = o.cat(3, k, o.cat(2, copies...))
	}
	kT := o.do(func() (*tensor.Tensor, error) {
		t, err := k.Permute(0, 2, 3, 1) // (B, H, dk, S)
		if err != nil {
			return nil, err
		}
		return t.Contiguous()
	})
	v := o.do(func() (*tensor.Tensor, error) {
		t, err := kv.Narrow(3, nope, dv)
		if err == nil {
			t, err = t.Permute(0, 2, 1, 3)
		}
		if err != nil {
			return nil, err
		}
		return t.Contiguous()
	})
	mask := m.dsaSelect(o, ly, x, qr)
	sc := o.matmul(o.reshape(q, B*H, S, dk), o.reshape(kT, B*H, dk, S))
	sc = o.add(o.affine(o.reshape(sc, B, H, S, S), 1/math.Sqrt(float64(dk)), 0), mask)
	p := o.reshape(o.do(func() (*tensor.Tensor, error) { return ops.Softmax(sc, 3) }), B*H, S, S)
	att := o.reshape(o.matmul(p, o.reshape(v, B*H, S, dv)), B, H, S, dv)
	out := o.reshape(o.perm(att, 0, 2, 1, 3), B, S, H*dv)
	if ly.MG != nil {
		out = o.mul(out, o.un(ops.Sigmoid, o.lin(ly.MG, x)))
	}
	return o.lin(ly.MO, out)
}

// moe is the routed experts (in the latent when routed_expert_hidden_size is
// set) plus the shared experts on x (B, S, D). Routing runs on the host:
// grouped top-k over σ(logits) + correction bias (a group scored by its two
// best), weights from the unbiased scores, renormalised and scaled.
func (m *KimiK31) moe(o *iop, ly *kimiK31Layer, x *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	cfg := m.Cfg
	B, S, D := x.Dim(0), x.Dim(1), x.Dim(2)
	T := B * S
	flat := o.reshape(x, T, D)
	routedIn := flat
	if ly.LatentDown != nil {
		routedIn = o.lin(ly.LatentDown, flat)
	}
	logits := o.do(func() (*tensor.Tensor, error) {
		wt, err := ly.RouterW.Transpose(0, 1)
		if err != nil {
			return nil, err
		}
		return ops.MatMul(flat, wt)
	})
	lv := o.toHost(logits)
	if o.err != nil {
		return nil
	}
	E, K := cfg.NumExperts, cfg.ExpertsPerTok
	G, TG := 1, 1
	if cfg.UseGroupedTopk {
		G, TG = cfg.NumExpertGroup, cfg.TopkGroup
	}
	per := E / G
	type hit struct {
		tok int
		w   float32
	}
	byExpert := make([][]hit, E)
	sc, sel := make([]float64, E), make([]float64, E)
	idx, gidx, gs := make([]int, 0, E), make([]int, G), make([]float64, G)
	for t := range T {
		row := lv[t*E : (t+1)*E]
		if cfg.RouterAct == "softmax" {
			mx := math.Inf(-1)
			for _, l := range row {
				mx = max(mx, float64(l))
			}
			var s float64
			for e, l := range row {
				sc[e] = math.Exp(float64(l) - mx)
				s += sc[e]
			}
			for e := range sc {
				sc[e] /= s
			}
		} else {
			for e, l := range row {
				sc[e] = 1 / (1 + math.Exp(-float64(l)))
			}
		}
		for e := range sel {
			sel[e] = sc[e]
			if ly.RouterBias != nil {
				sel[e] += float64(ly.RouterBias[e])
			}
		}
		idx = idx[:0]
		if G > 1 {
			for g := range G {
				grp := append([]float64(nil), sel[g*per:(g+1)*per]...)
				sort.Sort(sort.Reverse(sort.Float64Slice(grp)))
				if ly.RouterBias != nil && per > 1 {
					gs[g] = grp[0] + grp[1]
				} else {
					gs[g] = grp[0]
				}
				gidx[g] = g
			}
			sort.SliceStable(gidx, func(a, b int) bool { return gs[gidx[a]] > gs[gidx[b]] })
			for _, g := range gidx[:TG] {
				for e := g * per; e < (g+1)*per; e++ {
					idx = append(idx, e)
				}
			}
		} else {
			for e := range E {
				idx = append(idx, e)
			}
		}
		sort.SliceStable(idx, func(a, b int) bool { return sel[idx[a]] > sel[idx[b]] })
		var sum float64
		for _, e := range idx[:K] {
			sum += sc[e]
		}
		for _, e := range idx[:K] {
			w := sc[e]
			if cfg.MoERenormalize {
				w /= sum
			}
			byExpert[e] = append(byExpert[e], hit{t, float32(w * cfg.RoutedScaling)})
		}
	}
	var routed *tensor.Tensor
	for e, hs := range byExpert {
		if len(hs) == 0 || o.err != nil {
			continue
		}
		tok := make([]int64, len(hs))
		wv := make([]float32, len(hs))
		for i, h := range hs {
			tok[i], wv[i] = int64(h.tok), h.w
		}
		ti := o.do(func() (*tensor.Tensor, error) { return tensor.New(tok, []int{len(tok)}, m.dev) })
		wt := o.do(func() (*tensor.Tensor, error) { return tensor.New(wv, []int{len(wv), 1}, m.dev) })
		xs := o.do(func() (*tensor.Tensor, error) { return ops.IndexSelect(routedIn, ti) })
		y := o.mul(m.glu(o, ly.Experts[e], xs), wt)
		// Scatter back with a one-hot (T, n) product.
		oh := o.do(func() (*tensor.Tensor, error) {
			v := make([]float32, T*len(tok))
			for i, t := range tok {
				v[int(t)*len(tok)+i] = 1
			}
			return tensor.New(v, []int{T, len(tok)}, m.dev)
		})
		y = o.matmul(oh, y)
		if routed == nil {
			routed = y
		} else {
			routed = o.add(routed, y)
		}
	}
	if o.err != nil {
		return nil
	}
	if routed == nil {
		o.err = fmt.Errorf("kimi_k31 layer %d: no expert routed", ly.ID)
		return nil
	}
	if ly.LatentNormW != nil {
		routed = m.rms(o, routed, ly.LatentNormW)
	}
	if ly.LatentUp != nil {
		routed = o.lin(ly.LatentUp, routed)
	}
	if ly.Shared != nil {
		routed = o.add(routed, m.glu(o, ly.Shared, flat))
	}
	return o.reshape(routed, B, S, D)
}

func (m *KimiK31) mixer(o *iop, ly *kimiK31Layer, x *tensor.Tensor) *tensor.Tensor {
	if ly.KDA {
		return m.kda(o, ly, x)
	}
	return m.mla(o, ly, x)
}

func (m *KimiK31) ffn(o *iop, ly *kimiK31Layer, x *tensor.Tensor) *tensor.Tensor {
	if ly.Dense != nil {
		return m.glu(o, ly.Dense, x)
	}
	return m.moe(o, ly, x)
}

// Forward returns the logits (B, S, V) of every position for tokens (B, S).
func (m *KimiK31) Forward(tokens *tensor.Tensor) (*tensor.Tensor, error) {
	o := &iop{}
	cfg := m.Cfg
	B, S := tokens.Dim(0), tokens.Dim(1)
	flatTok := o.do(func() (*tensor.Tensor, error) { return tokens.Reshape(B * S) })
	h := o.reshape(o.do(func() (*tensor.Tensor, error) { return ops.IndexSelect(m.Embed, flatTok) }), B, S, cfg.HiddenSize)
	bs := cfg.AttnResBlockSize
	if bs <= 0 { // plain pre-norm residual stream
		for _, ly := range m.Layers {
			h = o.add(h, m.mixer(o, ly, m.rms(o, h, ly.InNormW)))
			h = o.add(h, m.ffn(o, ly, m.rms(o, h, ly.PostNormW)))
		}
		logits := o.lin(m.LMHead, m.rms(o, h, m.NormW))
		return logits, o.err
	}
	// Attention Residuals: blocks[j] snapshots the stream at layer j·bs; prefix
	// is the sum of the sub-layer outputs since the last snapshot (the
	// embedding before the first one).
	prefix := h
	var delta *tensor.Tensor
	var blocks []*tensor.Tensor
	for i, ly := range m.Layers {
		if delta != nil {
			prefix = o.add(prefix, delta)
		}
		prev, write := (i+bs-1)/bs, i%bs == 0
		if write {
			blocks = append(blocks, prefix)
		}
		x := m.attnRes(o, append(append([]*tensor.Tensor(nil), blocks[:prev]...), prefix), ly.AttnResW, ly.InNormW)
		a := m.mixer(o, ly, x)
		if write {
			prefix = a
		} else {
			prefix = o.add(prefix, a)
		}
		nb := prev
		if write {
			nb++
		}
		x = m.attnRes(o, append(append([]*tensor.Tensor(nil), blocks[:nb]...), prefix), ly.MLPResW, ly.PostNormW)
		delta = m.ffn(o, ly, x)
	}
	if delta != nil {
		prefix = o.add(prefix, delta)
	}
	out := m.attnRes(o, append(append([]*tensor.Tensor(nil), blocks...), prefix), m.OutResW, nil)
	logits := o.lin(m.LMHead, m.rms(o, out, m.NormW))
	return logits, o.err
}
