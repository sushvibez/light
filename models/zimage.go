package models

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// ZImageConfig is diffusers' ZImageTransformer2DModel config (basic mode:
// one patch size).
type ZImageConfig struct {
	PatchSizes    []int   `json:"all_patch_size"`
	FPatchSizes   []int   `json:"all_f_patch_size"`
	InChannels    int     `json:"in_channels"`
	Dim           int     `json:"dim"`
	NumLayers     int     `json:"n_layers"`
	NumRefiner    int     `json:"n_refiner_layers"`
	NumHeads      int     `json:"n_heads"`
	NumKVHeads    int     `json:"n_kv_heads"`
	NormEps       float64 `json:"norm_eps"`
	QKNorm        bool    `json:"qk_norm"`
	CapFeatDim    int     `json:"cap_feat_dim"`
	SiglipFeatDim *int    `json:"siglip_feat_dim"`
	RopeTheta     float64 `json:"rope_theta"`
	TScale        float64 `json:"t_scale"`
	AxesDims      []int   `json:"axes_dims"`
	AxesLens      []int   `json:"axes_lens"`
}

const zImageAdaLNDim = 256 // ADALN_EMBED_DIM
const zImageSeqMultiple = 32

// ZImageTransformer is the Z-Image single-stream DiT (S3-DiT): image
// patches refined by modulated blocks, the caption (Qwen3 states, RMSNorm +
// linear) by unmodulated ones, then joint blocks over [image, caption] with
// 3-axis complex RoPE, tanh-gated scale-only AdaLN and RMSNorm before and
// after attention and the SwiGLU MLP; a scale-only AdaLN output layer.
// Both sequences are padded to a multiple of 32 with learned pad tokens.
type ZImageTransformer struct {
	Cfg          ZImageConfig
	XEmbed       *nn.Linear
	FinalMod     *nn.Linear // adaLN_modulation.1 (after SiLU)
	FinalLinear  *nn.Linear
	TimeL1       *nn.Linear
	TimeL2       *nn.Linear
	CapNorm      *nn.RMSNorm
	CapLinear    *nn.Linear
	XPad, CapPad *tensor.Tensor // (1, dim)
	NoiseRefiner []*zImageBlock
	CtxRefiner   []*zImageBlock
	Layers       []*zImageBlock
	ones, zeros  *tensor.Tensor
}

type zImageBlock struct {
	Attn                 fluxAttn
	Out                  *nn.Linear
	W1, W2, W3           *nn.Linear
	AttnNorm1, AttnNorm2 *nn.RMSNorm
	FFNNorm1, FFNNorm2   *nn.RMSNorm
	Mod                  *nn.Linear // nil for the context refiner
}

func (c ZImageConfig) headDim() int { return c.Dim / c.NumHeads }
func (c ZImageConfig) patch() int   { return c.PatchSizes[0] }

// NewZImageTransformer allocates an uninitialised F32 model.
func NewZImageTransformer(cfg ZImageConfig, dev device.Device) (*ZImageTransformer, error) {
	if len(cfg.PatchSizes) != 1 || len(cfg.FPatchSizes) != 1 || cfg.FPatchSizes[0] != 1 {
		return nil, fmt.Errorf("z-image: patch sizes %v / %v not supported (one spatial size, frame patch 1)", cfg.PatchSizes, cfg.FPatchSizes)
	}
	if cfg.SiglipFeatDim != nil {
		return nil, fmt.Errorf("z-image: the Omni (SigLIP) variant is not supported")
	}
	if cfg.NumKVHeads != cfg.NumHeads || !cfg.QKNorm {
		return nil, fmt.Errorf("z-image: kv heads %d / qk_norm %v not supported", cfg.NumKVHeads, cfg.QKNorm)
	}
	D, Dh, P := cfg.Dim, cfg.headDim(), cfg.patch()
	ad := min(D, zImageAdaLNDim)
	hidden := int(float64(D) / 3 * 8)
	var err error
	lin := func(in, out int, bias bool) *nn.Linear {
		if err != nil {
			return nil
		}
		var l *nn.Linear
		l, err = nn.NewLinearUninit(in, out, bias, dtype.F32, dev)
		return l
	}
	rms := func(n int, eps float64) *nn.RMSNorm {
		if err != nil {
			return nil
		}
		var r *nn.RMSNorm
		r, err = nn.NewRMSNormWithDType(n, eps, dtype.F32, dev)
		return r
	}
	block := func(mod bool) *zImageBlock {
		b := &zImageBlock{
			Attn: fluxAttn{Q: lin(D, D, false), K: lin(D, D, false), V: lin(D, D, false), NormQ: rms(Dh, 1e-5), NormK: rms(Dh, 1e-5)},
			Out:  lin(D, D, false), W1: lin(D, hidden, false), W2: lin(hidden, D, false), W3: lin(D, hidden, false),
			AttnNorm1: rms(D, cfg.NormEps), AttnNorm2: rms(D, cfg.NormEps), FFNNorm1: rms(D, cfg.NormEps), FFNNorm2: rms(D, cfg.NormEps),
		}
		if mod {
			b.Mod = lin(ad, 4*D, true)
		}
		return b
	}
	m := &ZImageTransformer{Cfg: cfg, XEmbed: lin(P*P*cfg.InChannels, D, true),
		FinalMod: lin(ad, D, true), FinalLinear: lin(D, P*P*cfg.InChannels, true),
		TimeL1: lin(256, 1024, true), TimeL2: lin(1024, ad, true),
		CapNorm: rms(cfg.CapFeatDim, cfg.NormEps), CapLinear: lin(cfg.CapFeatDim, D, true)}
	for range cfg.NumRefiner {
		m.NoiseRefiner = append(m.NoiseRefiner, block(true))
		m.CtxRefiner = append(m.CtxRefiner, block(false))
	}
	for range cfg.NumLayers {
		m.Layers = append(m.Layers, block(true))
	}
	if err != nil {
		return nil, err
	}
	for _, p := range []**tensor.Tensor{&m.XPad, &m.CapPad} {
		if *p, err = tensor.Zeros(dtype.F32, []int{1, D}, dev); err != nil {
			return nil, err
		}
	}
	if m.ones, err = tensor.Ones(dtype.F32, []int{1}, dev); err != nil {
		return nil, err
	}
	if m.zeros, err = tensor.Zeros(dtype.F32, []int{D}, dev); err != nil {
		return nil, err
	}
	return m, nil
}

// block runs one ZImageTransformerBlock on x (1, S, D); adaln is the
// timestep embedding (1, 256) or nil for the unmodulated context refiner.
func (m *ZImageTransformer) block(b *zImageBlock, x, adaln, cos, sin *tensor.Tensor) (*tensor.Tensor, error) {
	H, Dh := m.Cfg.NumHeads, m.Cfg.headDim()
	var scaleA, gateA, scaleM, gateM *tensor.Tensor
	var err error
	if b.Mod != nil {
		mod, err := modulation(b.Mod, adaln, 4) // scale_msa, gate_msa, scale_mlp, gate_mlp
		if err != nil {
			return nil, err
		}
		if scaleA, err = ops.Add(mod[0], m.ones); err != nil {
			return nil, err
		}
		if gateA, err = ops.Tanh(mod[1]); err != nil {
			return nil, err
		}
		if scaleM, err = ops.Add(mod[2], m.ones); err != nil {
			return nil, err
		}
		if gateM, err = ops.Tanh(mod[3]); err != nil {
			return nil, err
		}
	}
	scale := func(t, s *tensor.Tensor) (*tensor.Tensor, error) {
		if s == nil {
			return t, nil
		}
		return ops.Mul(t, s)
	}
	n, err := b.AttnNorm1.Forward(x)
	if err != nil {
		return nil, err
	}
	if n, err = scale(n, scaleA); err != nil {
		return nil, err
	}
	q, k, v, err := b.Attn.qkv(n, H, Dh)
	if err != nil {
		return nil, err
	}
	a, err := attend(q, k, v, cos, sin)
	if err != nil {
		return nil, err
	}
	if a, err = b.Out.Forward(a); err != nil {
		return nil, err
	}
	if a, err = b.AttnNorm2.Forward(a); err != nil {
		return nil, err
	}
	if a, err = scale(a, gateA); err != nil {
		return nil, err
	}
	if x, err = ops.Add(x, a); err != nil {
		return nil, err
	}
	if n, err = b.FFNNorm1.Forward(x); err != nil {
		return nil, err
	}
	if n, err = scale(n, scaleM); err != nil {
		return nil, err
	}
	g, err := b.W1.Forward(n)
	if err != nil {
		return nil, err
	}
	if g, err = ops.SiLU(g); err != nil {
		return nil, err
	}
	u, err := b.W3.Forward(n)
	if err != nil {
		return nil, err
	}
	if g, err = ops.Mul(g, u); err != nil {
		return nil, err
	}
	f, err := b.W2.Forward(g)
	if err != nil {
		return nil, err
	}
	if f, err = b.FFNNorm2.Forward(f); err != nil {
		return nil, err
	}
	if f, err = scale(f, gateM); err != nil {
		return nil, err
	}
	return ops.Add(x, f)
}

// padRows pads (n, D) rows to a multiple of 32 by repeating the last row;
// returns the padded tensor and the real row count.
func padRows(x *tensor.Tensor) (*tensor.Tensor, int, error) {
	n := x.Dim(0)
	pad := (zImageSeqMultiple - n%zImageSeqMultiple) % zImageSeqMultiple
	if pad == 0 {
		return x, n, nil
	}
	last, err := x.Narrow(0, n-1, 1)
	if err != nil {
		return nil, 0, err
	}
	rows := []*tensor.Tensor{x}
	for range pad {
		rows = append(rows, last)
	}
	out, err := ops.Concat(rows, 0)
	return out, n, err
}

// withPadToken replaces rows >= real of e (1, S, D) by tok (1, D).
func withPadToken(e *tensor.Tensor, real int, tok *tensor.Tensor) (*tensor.Tensor, error) {
	S, D := e.Dim(1), e.Dim(2)
	if real == S {
		return e, nil
	}
	head, err := e.Narrow(1, 0, real)
	if err != nil {
		return nil, err
	}
	t, err := tok.Reshape(1, 1, D)
	if err != nil {
		return nil, err
	}
	parts := []*tensor.Tensor{head}
	for range S - real {
		parts = append(parts, t)
	}
	return ops.Concat(parts, 1)
}

// Forward predicts the model output for latents (C, H, W) at timestep
// tModel (the pipeline's (1000 − t)/1000) and caption features (L, cap_dim)
// (only the real tokens), returning (C, H, W). The pipeline negates it
// before the Euler step.
func (m *ZImageTransformer) Forward(latents, caption *tensor.Tensor, tModel float32) (*tensor.Tensor, error) {
	cfg := m.Cfg
	dev := latents.Device()
	C, Hh, Ww, P := latents.Dim(0), latents.Dim(1), latents.Dim(2), cfg.patch()
	ht, wt := Hh/P, Ww/P
	// Timestep embedding: [cos | sin] of t·t_scale, MLP → (1, 256).
	pe, err := zImageTimestep(tModel*float32(cfg.TScale), dev)
	if err != nil {
		return nil, err
	}
	temb, err := mlpSiLU(m.TimeL1, m.TimeL2, pe)
	if err != nil {
		return nil, err
	}
	// Caption: pad to 32, RMSNorm + linear, pad rows → cap pad token.
	capP, capReal, err := padRows(caption)
	if err != nil {
		return nil, err
	}
	Lp := capP.Dim(0)
	ce, err := m.CapNorm.Forward(capP)
	if err != nil {
		return nil, err
	}
	if ce, err = m.CapLinear.Forward(ce); err != nil {
		return nil, err
	}
	if ce, err = ce.Reshape(1, Lp, cfg.Dim); err != nil {
		return nil, err
	}
	if ce, err = withPadToken(ce, capReal, m.CapPad); err != nil {
		return nil, err
	}
	// Image: (C, H, W) → patches (ht·wt, P·P·C) ordered (py, px, c).
	x, err := latents.Reshape(C, ht, P, wt, P)
	if err != nil {
		return nil, err
	}
	if x, err = x.Permute(1, 3, 2, 4, 0); err != nil {
		return nil, err
	}
	if x, err = x.Contiguous(); err != nil {
		return nil, err
	}
	if x, err = x.Reshape(ht*wt, P*P*C); err != nil {
		return nil, err
	}
	xP, xReal, err := padRows(x)
	if err != nil {
		return nil, err
	}
	Sp := xP.Dim(0)
	xe, err := m.XEmbed.Forward(xP)
	if err != nil {
		return nil, err
	}
	if xe, err = xe.Reshape(1, Sp, cfg.Dim); err != nil {
		return nil, err
	}
	if xe, err = withPadToken(xe, xReal, m.XPad); err != nil {
		return nil, err
	}
	// Position ids: caption (1+i, 0, 0); image (Lp+1, row, col), pads (0, 0, 0).
	capIDs := make([]float32, Lp*3)
	for i := range Lp {
		capIDs[i*3] = float32(1 + i)
	}
	xIDs := make([]float32, Sp*3)
	for i := range xReal {
		xIDs[i*3], xIDs[i*3+1], xIDs[i*3+2] = float32(Lp+1), float32(i/wt), float32(i%wt)
	}
	Dh := cfg.headDim()
	xc, xs, err := fluxRoPETables(cfg.AxesDims, cfg.RopeTheta, Dh, xIDs, Sp, dev)
	if err != nil {
		return nil, err
	}
	cc, cs, err := fluxRoPETables(cfg.AxesDims, cfg.RopeTheta, Dh, capIDs, Lp, dev)
	if err != nil {
		return nil, err
	}
	for i, b := range m.NoiseRefiner {
		if xe, err = m.block(b, xe, temb, xc, xs); err != nil {
			return nil, fmt.Errorf("z-image: noise refiner %d: %w", i, err)
		}
	}
	for i, b := range m.CtxRefiner {
		if ce, err = m.block(b, ce, nil, cc, cs); err != nil {
			return nil, fmt.Errorf("z-image: context refiner %d: %w", i, err)
		}
	}
	u, err := ops.Concat([]*tensor.Tensor{xe, ce}, 1)
	if err != nil {
		return nil, err
	}
	uc, err := ops.Concat([]*tensor.Tensor{xc, cc}, 1)
	if err != nil {
		return nil, err
	}
	us, err := ops.Concat([]*tensor.Tensor{xs, cs}, 1)
	if err != nil {
		return nil, err
	}
	for i, b := range m.Layers {
		if u, err = m.block(b, u, temb, uc, us); err != nil {
			return nil, fmt.Errorf("z-image: layer %d: %w", i, err)
		}
	}
	// Final layer: LayerNorm (no affine) · (1 + linear(silu(t))), linear.
	st, err := ops.SiLU(temb)
	if err != nil {
		return nil, err
	}
	sc, err := m.FinalMod.Forward(st)
	if err != nil {
		return nil, err
	}
	if sc, err = ops.Add(sc, m.ones); err != nil {
		return nil, err
	}
	if sc, err = sc.Reshape(1, 1, cfg.Dim); err != nil {
		return nil, err
	}
	ones, err := tensor.Ones(dtype.F32, []int{cfg.Dim}, dev)
	if err != nil {
		return nil, err
	}
	o, err := ops.LayerNorm(u, ones, m.zeros, 1e-6)
	if err != nil {
		return nil, err
	}
	if o, err = ops.Mul(o, sc); err != nil {
		return nil, err
	}
	if o, err = m.FinalLinear.Forward(o); err != nil {
		return nil, err
	}
	// Image tokens back to (C, H, W).
	if o, err = o.Narrow(1, 0, xReal); err != nil {
		return nil, err
	}
	if o, err = o.Contiguous(); err != nil {
		return nil, err
	}
	if o, err = o.Reshape(ht, wt, P, P, C); err != nil {
		return nil, err
	}
	if o, err = o.Permute(4, 0, 2, 1, 3); err != nil {
		return nil, err
	}
	if o, err = o.Contiguous(); err != nil {
		return nil, err
	}
	return o.Reshape(C, Hh, Ww)
}

// zImageTimestep is TimestepEmbedder.timestep_embedding(t, 256): [cos | sin]
// of t · exp(−ln(10000)·i/128), in F32.
func zImageTimestep(t float32, dev device.Device) (*tensor.Tensor, error) {
	return timestepEmbedding([]float32{t}, dev)
}

// LoadZImageTransformer loads a diffusers ZImageTransformer2DModel from a
// directory with config.json and its safetensors, at F32 on dev.
func LoadZImageTransformer(dir string, dev device.Device) (*ZImageTransformer, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	cfg := ZImageConfig{PatchSizes: []int{2}, FPatchSizes: []int{1}, InChannels: 16, NormEps: 1e-5, QKNorm: true,
		RopeTheta: 256, TScale: 1000}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("z-image config: %w", err)
	}
	hf, err := loadCheckpointTensors(dir)
	if err != nil {
		return nil, err
	}
	m, err := NewZImageTransformer(cfg, dev)
	if err != nil {
		return nil, err
	}
	var lerr error
	cp := func(dst *tensor.Tensor, name string) {
		if lerr == nil {
			lerr = copyExactCast(dst, hf, name)
		}
	}
	lin := func(l *nn.Linear, name string) {
		cp(l.W, name+".weight")
		if l.B != nil {
			cp(l.B, name+".bias")
		}
	}
	key := fmt.Sprintf("%d-%d", cfg.patch(), cfg.FPatchSizes[0])
	lin(m.XEmbed, "all_x_embedder."+key)
	lin(m.FinalLinear, "all_final_layer."+key+".linear")
	lin(m.FinalMod, "all_final_layer."+key+".adaLN_modulation.1")
	lin(m.TimeL1, "t_embedder.mlp.0")
	lin(m.TimeL2, "t_embedder.mlp.2")
	cp(m.CapNorm.Weight, "cap_embedder.0.weight")
	lin(m.CapLinear, "cap_embedder.1")
	cp(m.XPad, "x_pad_token")
	cp(m.CapPad, "cap_pad_token")
	blocks := func(bs []*zImageBlock, prefix string) {
		for i, b := range bs {
			p := fmt.Sprintf("%s.%d.", prefix, i)
			lin(b.Attn.Q, p+"attention.to_q")
			lin(b.Attn.K, p+"attention.to_k")
			lin(b.Attn.V, p+"attention.to_v")
			cp(b.Attn.NormQ.Weight, p+"attention.norm_q.weight")
			cp(b.Attn.NormK.Weight, p+"attention.norm_k.weight")
			lin(b.Out, p+"attention.to_out.0")
			lin(b.W1, p+"feed_forward.w1")
			lin(b.W2, p+"feed_forward.w2")
			lin(b.W3, p+"feed_forward.w3")
			cp(b.AttnNorm1.Weight, p+"attention_norm1.weight")
			cp(b.AttnNorm2.Weight, p+"attention_norm2.weight")
			cp(b.FFNNorm1.Weight, p+"ffn_norm1.weight")
			cp(b.FFNNorm2.Weight, p+"ffn_norm2.weight")
			if b.Mod != nil {
				lin(b.Mod, p+"adaLN_modulation.0")
			}
		}
	}
	blocks(m.NoiseRefiner, "noise_refiner")
	blocks(m.CtxRefiner, "context_refiner")
	blocks(m.Layers, "layers")
	if lerr != nil {
		return nil, fmt.Errorf("LoadZImageTransformer: %w", lerr)
	}
	return m, nil
}
