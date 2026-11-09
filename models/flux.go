package models

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// FluxConfig is diffusers' FluxTransformer2DModel config.
type FluxConfig struct {
	PatchSize       int   `json:"patch_size"`
	InChannels      int   `json:"in_channels"`
	OutChannels     int   `json:"out_channels"`
	NumLayers       int   `json:"num_layers"`        // double-stream blocks
	NumSingleLayers int   `json:"num_single_layers"` // single-stream blocks
	HeadDim         int   `json:"attention_head_dim"`
	NumHeads        int   `json:"num_attention_heads"`
	JointDim        int   `json:"joint_attention_dim"`   // text (T5) hidden size
	PooledDim       int   `json:"pooled_projection_dim"` // pooled CLIP size
	GuidanceEmbeds  bool  `json:"guidance_embeds"`       // FLUX.1-dev style distilled guidance
	AxesDims        []int `json:"axes_dims_rope"`        // RoPE dims per position axis
}

// FluxTransformer is the Flux MMDiT (diffusers' FluxTransformer2DModel):
// double-stream blocks with joint text / image attention and separate
// weights per stream, then single-stream blocks over the concatenated
// sequence (attention and MLP in parallel), modulated by AdaLN from the
// timestep (+ guidance) and pooled-text embedding, with 3-axis RoPE over
// the text and image position ids.
type FluxTransformer struct {
	Cfg                  FluxConfig
	XEmbed, ContextEmbed *nn.Linear
	TimeL1, TimeL2       *nn.Linear
	GuideL1, GuideL2     *nn.Linear // nil without guidance_embeds
	TextL1, TextL2       *nn.Linear
	Double               []*fluxDoubleBlock
	Single               []*fluxSingleBlock
	NormOut, ProjOut     *nn.Linear
	ones, zeros          *tensor.Tensor // LayerNorm without affine
}

type fluxAttn struct {
	Q, K, V      *nn.Linear
	NormQ, NormK *nn.RMSNorm
}

type fluxDoubleBlock struct {
	Norm1, Norm1Ctx          *nn.Linear // AdaLN-Zero: 6·D modulation
	Img, Ctx                 fluxAttn
	Out, CtxOut              *nn.Linear
	FF1, FF2, CtxFF1, CtxFF2 *nn.Linear
}

type fluxSingleBlock struct {
	Norm             *nn.Linear // 3·D modulation
	Attn             fluxAttn
	ProjMLP, ProjOut *nn.Linear
}

func (c FluxConfig) inner() int { return c.NumHeads * c.HeadDim }

// NewFluxTransformer allocates an uninitialised F32 model.
func NewFluxTransformer(cfg FluxConfig, dev device.Device) (*FluxTransformer, error) {
	if cfg.PatchSize == 0 {
		cfg.PatchSize = 1
	}
	if cfg.OutChannels == 0 {
		cfg.OutChannels = cfg.InChannels
	}
	sum := 0
	for _, d := range cfg.AxesDims {
		sum += d
	}
	if sum != cfg.HeadDim {
		return nil, fmt.Errorf("flux: axes_dims_rope %v sum to %d, head dim is %d", cfg.AxesDims, sum, cfg.HeadDim)
	}
	D := cfg.inner()
	var err error
	lin := func(in, out int) *nn.Linear {
		if err != nil {
			return nil
		}
		var l *nn.Linear
		l, err = nn.NewLinearUninit(in, out, true, dtype.F32, dev)
		return l
	}
	rms := func() *nn.RMSNorm {
		if err != nil {
			return nil
		}
		var n *nn.RMSNorm
		n, err = nn.NewRMSNormWithDType(cfg.HeadDim, 1e-6, dtype.F32, dev)
		return n
	}
	attn := func() fluxAttn {
		return fluxAttn{Q: lin(D, D), K: lin(D, D), V: lin(D, D), NormQ: rms(), NormK: rms()}
	}
	m := &FluxTransformer{Cfg: cfg,
		XEmbed: lin(cfg.InChannels*cfg.PatchSize*cfg.PatchSize, D), ContextEmbed: lin(cfg.JointDim, D),
		TimeL1: lin(256, D), TimeL2: lin(D, D), TextL1: lin(cfg.PooledDim, D), TextL2: lin(D, D),
		NormOut: lin(D, 2*D), ProjOut: lin(D, cfg.PatchSize*cfg.PatchSize*cfg.OutChannels)}
	if cfg.GuidanceEmbeds {
		m.GuideL1, m.GuideL2 = lin(256, D), lin(D, D)
	}
	for range cfg.NumLayers {
		m.Double = append(m.Double, &fluxDoubleBlock{
			Norm1: lin(D, 6*D), Norm1Ctx: lin(D, 6*D), Img: attn(), Ctx: attn(),
			Out: lin(D, D), CtxOut: lin(D, D),
			FF1: lin(D, 4*D), FF2: lin(4*D, D), CtxFF1: lin(D, 4*D), CtxFF2: lin(4*D, D)})
	}
	for range cfg.NumSingleLayers {
		m.Single = append(m.Single, &fluxSingleBlock{Norm: lin(D, 3*D), Attn: attn(),
			ProjMLP: lin(D, 4*D), ProjOut: lin(5*D, D)})
	}
	if err != nil {
		return nil, err
	}
	if m.ones, err = tensor.Ones(dtype.F32, []int{D}, dev); err != nil {
		return nil, err
	}
	if m.zeros, err = tensor.Zeros(dtype.F32, []int{D}, dev); err != nil {
		return nil, err
	}
	return m, nil
}

// timestepEmbedding is diffusers' get_timestep_embedding(t, 256,
// flip_sin_to_cos=True, downscale_freq_shift=0) in F32: [cos | sin].
func timestepEmbedding(ts []float32, dev device.Device) (*tensor.Tensor, error) {
	const half = 128
	out := make([]float32, len(ts)*2*half)
	for b, t := range ts {
		for i := range half {
			e := float32(-math.Log(10000)) * float32(i) / float32(half)
			a := t * float32(math.Exp(float64(e)))
			out[b*2*half+i] = float32(math.Cos(float64(a)))
			out[b*2*half+half+i] = float32(math.Sin(float64(a)))
		}
	}
	return tensor.New(out, []int{len(ts), 2 * half}, dev)
}

// fluxRoPE is FluxPosEmbed: for each position axis, cos / sin of
// pos · θ^(−2j/d) (θ = 10000, F64 then F32), each repeated for the pair it
// rotates, concatenated over the axes. ids is (S, len(AxesDims)) row-major.
// Returns cos, sin as (1, S, 1, HeadDim).
func (m *FluxTransformer) fluxRoPE(ids []float32, S int, dev device.Device) (*tensor.Tensor, *tensor.Tensor, error) {
	return fluxRoPETables(m.Cfg.AxesDims, 10000, m.Cfg.HeadDim, ids, S, dev)
}

// fluxRoPETables builds FluxPosEmbed / Flux2PosEmbed tables for S
// positions with len(axes) coordinates each (ids row-major): per axis,
// cos / sin of pos · θ^(−2j/d) in F64, rounded to F32, repeated for the
// pair it rotates, concatenated over the axes; (1, S, 1, headDim).
func fluxRoPETables(axes []int, theta float64, headDim int, ids []float32, S int, dev device.Device) (*tensor.Tensor, *tensor.Tensor, error) {
	na, Dh := len(axes), headDim
	cs, sn := make([]float32, S*Dh), make([]float32, S*Dh)
	for s := range S {
		off := 0
		for a, d := range axes {
			pos := float64(ids[s*na+a])
			for j := range d / 2 {
				f := 1 / math.Pow(theta, float64(2*j)/float64(d))
				ang := pos * f
				c, si := float32(math.Cos(ang)), float32(math.Sin(ang))
				cs[s*Dh+off+2*j], cs[s*Dh+off+2*j+1] = c, c
				sn[s*Dh+off+2*j], sn[s*Dh+off+2*j+1] = si, si
			}
			off += d
		}
	}
	c, err := tensor.New(cs, []int{1, S, 1, Dh}, dev)
	if err != nil {
		return nil, nil, err
	}
	s, err := tensor.New(sn, []int{1, S, 1, Dh}, dev)
	return c, s, err
}

// applyFluxRoPE rotates x (B, S, H, Dh) in interleaved pairs:
// x·cos + rot(x)·sin with rot(x)[2i] = −x[2i+1], rot(x)[2i+1] = x[2i].
func applyFluxRoPE(x, cos, sin *tensor.Tensor) (*tensor.Tensor, error) {
	B, S, H, Dh := x.Dim(0), x.Dim(1), x.Dim(2), x.Dim(3)
	xc, err := x.Contiguous()
	if err != nil {
		return nil, err
	}
	p, err := xc.Reshape(B, S, H, Dh/2, 2)
	if err != nil {
		return nil, err
	}
	even, err := p.Narrow(4, 0, 1)
	if err != nil {
		return nil, err
	}
	odd, err := p.Narrow(4, 1, 1)
	if err != nil {
		return nil, err
	}
	nodd, err := ops.Neg(odd)
	if err != nil {
		return nil, err
	}
	rot, err := ops.Concat([]*tensor.Tensor{nodd, even}, 4)
	if err != nil {
		return nil, err
	}
	if rot, err = rot.Reshape(B, S, H, Dh); err != nil {
		return nil, err
	}
	a, err := ops.Mul(xc, cos)
	if err != nil {
		return nil, err
	}
	b, err := ops.Mul(rot, sin)
	if err != nil {
		return nil, err
	}
	return ops.Add(a, b)
}

// qkv projects x (B, S, D) to RMS-normed q, k and v (B, S, H, Dh).
func (a *fluxAttn) qkv(x *tensor.Tensor, H, Dh int) (q, k, v *tensor.Tensor, err error) {
	B, S := x.Dim(0), x.Dim(1)
	proj := func(l *nn.Linear, n *nn.RMSNorm) (*tensor.Tensor, error) {
		y, err := l.Forward(x)
		if err != nil {
			return nil, err
		}
		if y, err = y.Reshape(B, S, H, Dh); err != nil {
			return nil, err
		}
		if n != nil {
			return n.Forward(y)
		}
		return y, nil
	}
	if q, err = proj(a.Q, a.NormQ); err != nil {
		return
	}
	if k, err = proj(a.K, a.NormK); err != nil {
		return
	}
	v, err = proj(a.V, nil)
	return
}

// attend runs RoPE + softmax(q·kᵀ/√Dh)·v over (B, S, H, Dh) inputs and
// returns (B, S, H·Dh).
func attend(q, k, v, cos, sin *tensor.Tensor) (*tensor.Tensor, error) {
	B, S, H, Dh := q.Dim(0), q.Dim(1), q.Dim(2), q.Dim(3)
	var err error
	if q, err = applyFluxRoPE(q, cos, sin); err != nil {
		return nil, err
	}
	if k, err = applyFluxRoPE(k, cos, sin); err != nil {
		return nil, err
	}
	perm := func(t *tensor.Tensor) (*tensor.Tensor, error) { return autograd.Permute(t, 0, 2, 1, 3) }
	if q, err = perm(q); err != nil {
		return nil, err
	}
	if k, err = perm(k); err != nil {
		return nil, err
	}
	if v, err = perm(v); err != nil {
		return nil, err
	}
	o, err := nn.SDPA(q, k, v, nil)
	if err != nil {
		return nil, err
	}
	if o, err = perm(o); err != nil {
		return nil, err
	}
	if o, err = o.Contiguous(); err != nil {
		return nil, err
	}
	return o.Reshape(B, S, H*Dh)
}

// modulation is linear(silu(temb)) split into n (B, 1, D) chunks.
func modulation(l *nn.Linear, stemb *tensor.Tensor, n int) ([]*tensor.Tensor, error) {
	e, err := l.Forward(stemb)
	if err != nil {
		return nil, err
	}
	B, D := e.Dim(0), e.Dim(1)/n
	out := make([]*tensor.Tensor, n)
	for i := range n {
		c, err := e.Narrow(1, i*D, D)
		if err != nil {
			return nil, err
		}
		if c, err = c.Contiguous(); err != nil {
			return nil, err
		}
		if out[i], err = c.Reshape(B, 1, D); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// modulate is LayerNorm(x) (no affine, eps 1e-6) · (1 + scale) + shift.
func (m *FluxTransformer) modulate(x, shift, scale *tensor.Tensor) (*tensor.Tensor, error) {
	n, err := ops.LayerNorm(x, m.ones, m.zeros, 1e-6)
	if err != nil {
		return nil, err
	}
	s1, err := ops.Add(scale, m.ones)
	if err != nil {
		return nil, err
	}
	if n, err = ops.Mul(n, s1); err != nil {
		return nil, err
	}
	return ops.Add(n, shift)
}

func gateAdd(x, gate, y *tensor.Tensor) (*tensor.Tensor, error) {
	g, err := ops.Mul(gate, y)
	if err != nil {
		return nil, err
	}
	return ops.Add(x, g)
}

func mlp(l1, l2 *nn.Linear, x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := l1.Forward(x)
	if err != nil {
		return nil, err
	}
	if h, err = ops.Gelu(h); err != nil { // tanh approximation
		return nil, err
	}
	return l2.Forward(h)
}

// FluxInputs are one transformer call's inputs.
type FluxInputs struct {
	Image    *tensor.Tensor // (B, S_img, in_channels·patch²) packed latents
	Text     *tensor.Tensor // (B, S_txt, joint_attention_dim) T5 states
	Pooled   *tensor.Tensor // (B, pooled_projection_dim) CLIP pooled
	Timestep []float32      // B values in [0, 1] (scaled by 1000 inside)
	Guidance []float32      // B values, nil without guidance_embeds
	ImgIDs   []float32      // (S_img, axes) position ids
	TxtIDs   []float32      // (S_txt, axes)
}

// Forward predicts the flow-matching velocity (B, S_img, out_channels·patch²).
func (m *FluxTransformer) Forward(in FluxInputs) (*tensor.Tensor, error) {
	dev := in.Image.Device()

	St := in.Text.Dim(1)
	x, err := m.XEmbed.Forward(in.Image)
	if err != nil {
		return nil, err
	}
	ctx, err := m.ContextEmbed.Forward(in.Text)
	if err != nil {
		return nil, err
	}
	// temb = time(t·1000) [+ guidance(g·1000)] + text(pooled)
	embed := func(vals []float32, l1, l2 *nn.Linear) (*tensor.Tensor, error) {
		s := make([]float32, len(vals))
		for i, v := range vals {
			s[i] = v * 1000
		}
		p, err := timestepEmbedding(s, dev)
		if err != nil {
			return nil, err
		}
		return mlpSiLU(l1, l2, p)
	}
	temb, err := embed(in.Timestep, m.TimeL1, m.TimeL2)
	if err != nil {
		return nil, err
	}
	if m.GuideL1 != nil {
		if in.Guidance == nil {
			return nil, fmt.Errorf("flux: the model embeds guidance; FluxInputs.Guidance is nil")
		}
		g, err := embed(in.Guidance, m.GuideL1, m.GuideL2)
		if err != nil {
			return nil, err
		}
		if temb, err = ops.Add(temb, g); err != nil {
			return nil, err
		}
	}
	pt, err := mlpSiLU(m.TextL1, m.TextL2, in.Pooled)
	if err != nil {
		return nil, err
	}
	if temb, err = ops.Add(temb, pt); err != nil {
		return nil, err
	}
	stemb, err := ops.SiLU(temb)
	if err != nil {
		return nil, err
	}
	ids := append(append([]float32{}, in.TxtIDs...), in.ImgIDs...)
	cos, sin, err := m.fluxRoPE(ids, St+x.Dim(1), dev)
	if err != nil {
		return nil, err
	}

	for i, b := range m.Double {
		if x, ctx, err = m.doubleBlock(b, x, ctx, stemb, cos, sin); err != nil {
			return nil, fmt.Errorf("flux: double block %d: %w", i, err)
		}
	}
	h, err := ops.Concat([]*tensor.Tensor{ctx, x}, 1)
	if err != nil {
		return nil, err
	}
	for i, b := range m.Single {
		if h, err = m.singleBlock(b, h, stemb, cos, sin); err != nil {
			return nil, fmt.Errorf("flux: single block %d: %w", i, err)
		}
	}
	if x, err = h.Narrow(1, St, h.Dim(1)-St); err != nil {
		return nil, err
	}
	mod, err := modulation(m.NormOut, stemb, 2) // scale, shift (in that order)
	if err != nil {
		return nil, err
	}
	if x, err = m.modulate(x, mod[1], mod[0]); err != nil {
		return nil, err
	}
	return m.ProjOut.Forward(x)
}

func mlpSiLU(l1, l2 *nn.Linear, x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := l1.Forward(x)
	if err != nil {
		return nil, err
	}
	if h, err = ops.SiLU(h); err != nil {
		return nil, err
	}
	return l2.Forward(h)
}

func (m *FluxTransformer) doubleBlock(b *fluxDoubleBlock, x, ctx, stemb, cos, sin *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	H, Dh := m.Cfg.NumHeads, m.Cfg.HeadDim
	St := ctx.Dim(1)
	mi, err := modulation(b.Norm1, stemb, 6) // shift, scale, gate (attn); shift, scale, gate (mlp)
	if err != nil {
		return nil, nil, err
	}
	mc, err := modulation(b.Norm1Ctx, stemb, 6)
	if err != nil {
		return nil, nil, err
	}
	nx, err := m.modulate(x, mi[0], mi[1])
	if err != nil {
		return nil, nil, err
	}
	nc, err := m.modulate(ctx, mc[0], mc[1])
	if err != nil {
		return nil, nil, err
	}
	qi, ki, vi, err := b.Img.qkv(nx, H, Dh)
	if err != nil {
		return nil, nil, err
	}
	qc, kc, vc, err := b.Ctx.qkv(nc, H, Dh)
	if err != nil {
		return nil, nil, err
	}
	cat := func(c, i *tensor.Tensor) (*tensor.Tensor, error) { return ops.Concat([]*tensor.Tensor{c, i}, 1) }
	q, err := cat(qc, qi)
	if err != nil {
		return nil, nil, err
	}
	k, err := cat(kc, ki)
	if err != nil {
		return nil, nil, err
	}
	v, err := cat(vc, vi)
	if err != nil {
		return nil, nil, err
	}
	o, err := attend(q, k, v, cos, sin)
	if err != nil {
		return nil, nil, err
	}
	oc, err := o.Narrow(1, 0, St)
	if err != nil {
		return nil, nil, err
	}
	oi, err := o.Narrow(1, St, o.Dim(1)-St)
	if err != nil {
		return nil, nil, err
	}
	if oi, err = b.Out.Forward(oi); err != nil {
		return nil, nil, err
	}
	if oc, err = b.CtxOut.Forward(oc); err != nil {
		return nil, nil, err
	}
	stream := func(h, o *tensor.Tensor, md []*tensor.Tensor, f1, f2 *nn.Linear) (*tensor.Tensor, error) {
		h, err := gateAdd(h, md[2], o)
		if err != nil {
			return nil, err
		}
		n, err := m.modulate(h, md[3], md[4])
		if err != nil {
			return nil, err
		}
		f, err := mlp(f1, f2, n)
		if err != nil {
			return nil, err
		}
		return gateAdd(h, md[5], f)
	}
	if x, err = stream(x, oi, mi, b.FF1, b.FF2); err != nil {
		return nil, nil, err
	}
	if ctx, err = stream(ctx, oc, mc, b.CtxFF1, b.CtxFF2); err != nil {
		return nil, nil, err
	}
	return x, ctx, nil
}

func (m *FluxTransformer) singleBlock(b *fluxSingleBlock, h, stemb, cos, sin *tensor.Tensor) (*tensor.Tensor, error) {
	md, err := modulation(b.Norm, stemb, 3) // shift, scale, gate
	if err != nil {
		return nil, err
	}
	n, err := m.modulate(h, md[0], md[1])
	if err != nil {
		return nil, err
	}
	mlpH, err := b.ProjMLP.Forward(n)
	if err != nil {
		return nil, err
	}
	if mlpH, err = ops.Gelu(mlpH); err != nil {
		return nil, err
	}
	q, k, v, err := b.Attn.qkv(n, m.Cfg.NumHeads, m.Cfg.HeadDim)
	if err != nil {
		return nil, err
	}
	a, err := attend(q, k, v, cos, sin)
	if err != nil {
		return nil, err
	}
	cat, err := ops.Concat([]*tensor.Tensor{a, mlpH}, 2)
	if err != nil {
		return nil, err
	}
	o, err := b.ProjOut.Forward(cat)
	if err != nil {
		return nil, err
	}
	return gateAdd(h, md[2], o)
}

// LoadFluxTransformer loads a diffusers FluxTransformer2DModel from a
// directory holding config.json and its safetensors (a pipeline's
// `transformer/` folder) at F32 on dev.
func LoadFluxTransformer(dir string, dev device.Device) (*FluxTransformer, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var cfg FluxConfig
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("flux config: %w", err)
	}
	hf, err := loadCheckpointTensors(dir)
	if err != nil {
		return nil, err
	}
	m, err := NewFluxTransformer(cfg, dev)
	if err != nil {
		return nil, err
	}
	var lerr error
	lin := func(l *nn.Linear, name string) {
		if lerr == nil {
			lerr = copyExactCast(l.W, hf, name+".weight")
		}
		if lerr == nil {
			lerr = copyExactCast(l.B, hf, name+".bias")
		}
	}
	norm := func(n *nn.RMSNorm, name string) {
		if lerr == nil {
			lerr = copyExactCast(n.Weight, hf, name+".weight")
		}
	}
	lin(m.XEmbed, "x_embedder")
	lin(m.ContextEmbed, "context_embedder")
	lin(m.TimeL1, "time_text_embed.timestep_embedder.linear_1")
	lin(m.TimeL2, "time_text_embed.timestep_embedder.linear_2")
	if m.GuideL1 != nil {
		lin(m.GuideL1, "time_text_embed.guidance_embedder.linear_1")
		lin(m.GuideL2, "time_text_embed.guidance_embedder.linear_2")
	}
	lin(m.TextL1, "time_text_embed.text_embedder.linear_1")
	lin(m.TextL2, "time_text_embed.text_embedder.linear_2")
	for i, b := range m.Double {
		p := fmt.Sprintf("transformer_blocks.%d.", i)
		lin(b.Norm1, p+"norm1.linear")
		lin(b.Norm1Ctx, p+"norm1_context.linear")
		lin(b.Img.Q, p+"attn.to_q")
		lin(b.Img.K, p+"attn.to_k")
		lin(b.Img.V, p+"attn.to_v")
		norm(b.Img.NormQ, p+"attn.norm_q")
		norm(b.Img.NormK, p+"attn.norm_k")
		lin(b.Ctx.Q, p+"attn.add_q_proj")
		lin(b.Ctx.K, p+"attn.add_k_proj")
		lin(b.Ctx.V, p+"attn.add_v_proj")
		norm(b.Ctx.NormQ, p+"attn.norm_added_q")
		norm(b.Ctx.NormK, p+"attn.norm_added_k")
		lin(b.Out, p+"attn.to_out.0")
		lin(b.CtxOut, p+"attn.to_add_out")
		lin(b.FF1, p+"ff.net.0.proj")
		lin(b.FF2, p+"ff.net.2")
		lin(b.CtxFF1, p+"ff_context.net.0.proj")
		lin(b.CtxFF2, p+"ff_context.net.2")
	}
	for i, b := range m.Single {
		p := fmt.Sprintf("single_transformer_blocks.%d.", i)
		lin(b.Norm, p+"norm.linear")
		lin(b.Attn.Q, p+"attn.to_q")
		lin(b.Attn.K, p+"attn.to_k")
		lin(b.Attn.V, p+"attn.to_v")
		norm(b.Attn.NormQ, p+"attn.norm_q")
		norm(b.Attn.NormK, p+"attn.norm_k")
		lin(b.ProjMLP, p+"proj_mlp")
		lin(b.ProjOut, p+"proj_out")
	}
	lin(m.NormOut, "norm_out.linear")
	lin(m.ProjOut, "proj_out")
	if lerr != nil {
		return nil, fmt.Errorf("LoadFluxTransformer: %w", lerr)
	}
	return m, nil
}
