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

// Flux2Config is diffusers' Flux2Transformer2DModel config (FLUX.2 dev /
// klein).
type Flux2Config struct {
	PatchSize        int     `json:"patch_size"`
	InChannels       int     `json:"in_channels"`
	OutChannels      int     `json:"out_channels"`
	NumLayers        int     `json:"num_layers"`
	NumSingleLayers  int     `json:"num_single_layers"`
	HeadDim          int     `json:"attention_head_dim"`
	NumHeads         int     `json:"num_attention_heads"`
	JointDim         int     `json:"joint_attention_dim"`
	TimestepChannels int     `json:"timestep_guidance_channels"`
	MLPRatio         float64 `json:"mlp_ratio"`
	AxesDims         []int   `json:"axes_dims_rope"`
	RopeTheta        float64 `json:"rope_theta"`
	Eps              float64 `json:"eps"`
	GuidanceEmbeds   bool    `json:"guidance_embeds"`
}

func (c Flux2Config) inner() int { return c.NumHeads * c.HeadDim }
func (c Flux2Config) mlp() int   { return int(float64(c.inner()) * c.MLPRatio) }

// Flux2Transformer is the FLUX.2 MMDiT: like Flux but bias-free, with
// modulations shared by all blocks (one image and one text set for the
// double-stream blocks, one for the single-stream ones) driven by the
// timestep (+ guidance) alone, SwiGLU MLPs, single-stream blocks with one
// fused qkv + MLP projection, and RoPE over four position axes.
type Flux2Transformer struct {
	Cfg                       Flux2Config
	XEmbed, ContextEmbed      *nn.Linear
	TimeL1, TimeL2            *nn.Linear
	GuideL1, GuideL2          *nn.Linear
	ModImg, ModTxt, ModSingle *nn.Linear
	Double                    []*flux2DoubleBlock
	Single                    []*flux2SingleBlock
	NormOut, ProjOut          *nn.Linear
	ones, zeros               *tensor.Tensor
}

type flux2DoubleBlock struct {
	Img, Ctx          fluxAttn
	Out, CtxOut       *nn.Linear
	FFIn, FFOut       *nn.Linear
	CtxFFIn, CtxFFOut *nn.Linear
}

type flux2SingleBlock struct {
	QKVMLP       *nn.Linear // D → 3D + 2·mlp
	NormQ, NormK *nn.RMSNorm
	Out          *nn.Linear // D + mlp → D
}

// NewFlux2Transformer allocates an uninitialised F32 model.
func NewFlux2Transformer(cfg Flux2Config, dev device.Device) (*Flux2Transformer, error) {
	if cfg.PatchSize == 0 {
		cfg.PatchSize = 1
	}
	if cfg.OutChannels == 0 {
		cfg.OutChannels = cfg.InChannels
	}
	if cfg.TimestepChannels == 0 {
		cfg.TimestepChannels = 256
	}
	if cfg.Eps == 0 {
		cfg.Eps = 1e-6
	}
	if cfg.TimestepChannels != 256 {
		return nil, fmt.Errorf("flux2: timestep_guidance_channels %d not supported (256)", cfg.TimestepChannels)
	}
	sum := 0
	for _, d := range cfg.AxesDims {
		sum += d
	}
	if sum != cfg.HeadDim {
		return nil, fmt.Errorf("flux2: axes_dims_rope %v sum to %d, head dim is %d", cfg.AxesDims, sum, cfg.HeadDim)
	}
	D, M := cfg.inner(), cfg.mlp()
	var err error
	lin := func(in, out int) *nn.Linear {
		if err != nil {
			return nil
		}
		var l *nn.Linear
		l, err = nn.NewLinearUninit(in, out, false, dtype.F32, dev)
		return l
	}
	rms := func() *nn.RMSNorm {
		if err != nil {
			return nil
		}
		var n *nn.RMSNorm
		n, err = nn.NewRMSNormWithDType(cfg.HeadDim, cfg.Eps, dtype.F32, dev)
		return n
	}
	attn := func() fluxAttn { return fluxAttn{Q: lin(D, D), K: lin(D, D), V: lin(D, D), NormQ: rms(), NormK: rms()} }
	m := &Flux2Transformer{Cfg: cfg,
		XEmbed: lin(cfg.InChannels*cfg.PatchSize*cfg.PatchSize, D), ContextEmbed: lin(cfg.JointDim, D),
		TimeL1: lin(256, D), TimeL2: lin(D, D),
		ModImg: lin(D, 6*D), ModTxt: lin(D, 6*D), ModSingle: lin(D, 3*D),
		NormOut: lin(D, 2*D), ProjOut: lin(D, cfg.PatchSize*cfg.PatchSize*cfg.OutChannels)}
	if cfg.GuidanceEmbeds {
		m.GuideL1, m.GuideL2 = lin(256, D), lin(D, D)
	}
	for range cfg.NumLayers {
		m.Double = append(m.Double, &flux2DoubleBlock{Img: attn(), Ctx: attn(), Out: lin(D, D), CtxOut: lin(D, D),
			FFIn: lin(D, 2*M), FFOut: lin(M, D), CtxFFIn: lin(D, 2*M), CtxFFOut: lin(M, D)})
	}
	for range cfg.NumSingleLayers {
		m.Single = append(m.Single, &flux2SingleBlock{QKVMLP: lin(D, 3*D+2*M), NormQ: rms(), NormK: rms(), Out: lin(D+M, D)})
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

// swiglu is silu(x[..., :half]) · x[..., half:].
func swiglu(x *tensor.Tensor) (*tensor.Tensor, error) {
	ax := x.Rank() - 1
	half := x.Dim(ax) / 2
	a, err := x.Narrow(ax, 0, half)
	if err != nil {
		return nil, err
	}
	b, err := x.Narrow(ax, half, half)
	if err != nil {
		return nil, err
	}
	if a, err = ops.SiLU(a); err != nil {
		return nil, err
	}
	return ops.Mul(a, b)
}

func (m *Flux2Transformer) modulate(x, shift, scale *tensor.Tensor) (*tensor.Tensor, error) {
	n, err := ops.LayerNorm(x, m.ones, m.zeros, m.Cfg.Eps)
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

// Flux2Inputs are one transformer call's inputs.
type Flux2Inputs struct {
	Image    *tensor.Tensor // (B, S_img, in_channels) packed latents
	Text     *tensor.Tensor // (B, S_txt, joint_attention_dim)
	Timestep []float32      // B values in [0, 1]
	Guidance []float32      // B values; nil without guidance_embeds
	ImgIDs   []float32      // (S_img, len(axes))
	TxtIDs   []float32      // (S_txt, len(axes))
}

// Forward predicts the flow-matching velocity (B, S_img, out_channels).
func (m *Flux2Transformer) Forward(in Flux2Inputs) (*tensor.Tensor, error) {
	dev := in.Image.Device()
	cfg := m.Cfg
	H, Dh := cfg.NumHeads, cfg.HeadDim
	St := in.Text.Dim(1)
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
	if m.GuideL1 != nil && in.Guidance != nil {
		g, err := embed(in.Guidance, m.GuideL1, m.GuideL2)
		if err != nil {
			return nil, err
		}
		if temb, err = ops.Add(temb, g); err != nil {
			return nil, err
		}
	}
	stemb, err := ops.SiLU(temb)
	if err != nil {
		return nil, err
	}
	modImg, err := modulation(m.ModImg, stemb, 6)
	if err != nil {
		return nil, err
	}
	modTxt, err := modulation(m.ModTxt, stemb, 6)
	if err != nil {
		return nil, err
	}
	modS, err := modulation(m.ModSingle, stemb, 3)
	if err != nil {
		return nil, err
	}
	x, err := m.XEmbed.Forward(in.Image)
	if err != nil {
		return nil, err
	}
	ctx, err := m.ContextEmbed.Forward(in.Text)
	if err != nil {
		return nil, err
	}
	ids := append(append([]float32{}, in.TxtIDs...), in.ImgIDs...)
	cos, sin, err := fluxRoPETables(cfg.AxesDims, cfg.RopeTheta, Dh, ids, St+x.Dim(1), dev)
	if err != nil {
		return nil, err
	}
	for i, b := range m.Double {
		if x, ctx, err = m.doubleBlock(b, x, ctx, modImg, modTxt, cos, sin); err != nil {
			return nil, fmt.Errorf("flux2: double block %d: %w", i, err)
		}
	}
	h, err := ops.Concat([]*tensor.Tensor{ctx, x}, 1)
	if err != nil {
		return nil, err
	}
	for i, b := range m.Single {
		if h, err = m.singleBlock(b, h, modS, cos, sin, H, Dh); err != nil {
			return nil, fmt.Errorf("flux2: single block %d: %w", i, err)
		}
	}
	if x, err = h.Narrow(1, St, h.Dim(1)-St); err != nil {
		return nil, err
	}
	mod, err := modulation(m.NormOut, stemb, 2) // scale, shift
	if err != nil {
		return nil, err
	}
	if x, err = m.modulate(x, mod[1], mod[0]); err != nil {
		return nil, err
	}
	return m.ProjOut.Forward(x)
}

func (m *Flux2Transformer) doubleBlock(b *flux2DoubleBlock, x, ctx *tensor.Tensor, mi, mc []*tensor.Tensor, cos, sin *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	H, Dh := m.Cfg.NumHeads, m.Cfg.HeadDim
	St := ctx.Dim(1)
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
	stream := func(h, o *tensor.Tensor, md []*tensor.Tensor, fin, fout *nn.Linear) (*tensor.Tensor, error) {
		h, err := gateAdd(h, md[2], o)
		if err != nil {
			return nil, err
		}
		n, err := m.modulate(h, md[3], md[4])
		if err != nil {
			return nil, err
		}
		f, err := fin.Forward(n)
		if err != nil {
			return nil, err
		}
		if f, err = swiglu(f); err != nil {
			return nil, err
		}
		if f, err = fout.Forward(f); err != nil {
			return nil, err
		}
		return gateAdd(h, md[5], f)
	}
	if x, err = stream(x, oi, mi, b.FFIn, b.FFOut); err != nil {
		return nil, nil, err
	}
	if ctx, err = stream(ctx, oc, mc, b.CtxFFIn, b.CtxFFOut); err != nil {
		return nil, nil, err
	}
	return x, ctx, nil
}

func (m *Flux2Transformer) singleBlock(b *flux2SingleBlock, h *tensor.Tensor, md []*tensor.Tensor, cos, sin *tensor.Tensor, H, Dh int) (*tensor.Tensor, error) {
	B, S, D, M := h.Dim(0), h.Dim(1), m.Cfg.inner(), m.Cfg.mlp()
	n, err := m.modulate(h, md[0], md[1])
	if err != nil {
		return nil, err
	}
	p, err := b.QKVMLP.Forward(n)
	if err != nil {
		return nil, err
	}
	part := func(off, width int) (*tensor.Tensor, error) {
		t, err := p.Narrow(2, off, width)
		if err != nil {
			return nil, err
		}
		return t.Contiguous()
	}
	head := func(off int, norm *nn.RMSNorm) (*tensor.Tensor, error) {
		t, err := part(off, D)
		if err != nil {
			return nil, err
		}
		if t, err = t.Reshape(B, S, H, Dh); err != nil {
			return nil, err
		}
		if norm != nil {
			return norm.Forward(t)
		}
		return t, nil
	}
	q, err := head(0, b.NormQ)
	if err != nil {
		return nil, err
	}
	k, err := head(D, b.NormK)
	if err != nil {
		return nil, err
	}
	v, err := head(2*D, nil)
	if err != nil {
		return nil, err
	}
	a, err := attend(q, k, v, cos, sin)
	if err != nil {
		return nil, err
	}
	mlpH, err := part(3*D, 2*M)
	if err != nil {
		return nil, err
	}
	if mlpH, err = swiglu(mlpH); err != nil {
		return nil, err
	}
	cat, err := ops.Concat([]*tensor.Tensor{a, mlpH}, 2)
	if err != nil {
		return nil, err
	}
	o, err := b.Out.Forward(cat)
	if err != nil {
		return nil, err
	}
	return gateAdd(h, md[2], o)
}

// LoadFlux2Transformer loads a diffusers Flux2Transformer2DModel from a
// directory with config.json and its safetensors, at F32 on dev.
func LoadFlux2Transformer(dir string, dev device.Device) (*Flux2Transformer, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	cfg := Flux2Config{MLPRatio: 3, RopeTheta: 2000, Eps: 1e-6, TimestepChannels: 256}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("flux2 config: %w", err)
	}
	hf, err := loadCheckpointTensors(dir)
	if err != nil {
		return nil, err
	}
	m, err := NewFlux2Transformer(cfg, dev)
	if err != nil {
		return nil, err
	}
	var lerr error
	cp := func(dst *tensor.Tensor, name string) {
		if lerr == nil {
			lerr = copyExactCast(dst, hf, name)
		}
	}
	lin := func(l *nn.Linear, name string) { cp(l.W, name+".weight") }
	cp2 := func(n *nn.RMSNorm, name string) { cp(n.Weight, name+".weight") }
	lin(m.XEmbed, "x_embedder")
	lin(m.ContextEmbed, "context_embedder")
	lin(m.TimeL1, "time_guidance_embed.timestep_embedder.linear_1")
	lin(m.TimeL2, "time_guidance_embed.timestep_embedder.linear_2")
	if m.GuideL1 != nil {
		lin(m.GuideL1, "time_guidance_embed.guidance_embedder.linear_1")
		lin(m.GuideL2, "time_guidance_embed.guidance_embedder.linear_2")
	}
	lin(m.ModImg, "double_stream_modulation_img.linear")
	lin(m.ModTxt, "double_stream_modulation_txt.linear")
	lin(m.ModSingle, "single_stream_modulation.linear")
	for i, b := range m.Double {
		p := fmt.Sprintf("transformer_blocks.%d.", i)
		lin(b.Img.Q, p+"attn.to_q")
		lin(b.Img.K, p+"attn.to_k")
		lin(b.Img.V, p+"attn.to_v")
		cp2(b.Img.NormQ, p+"attn.norm_q")
		cp2(b.Img.NormK, p+"attn.norm_k")
		lin(b.Ctx.Q, p+"attn.add_q_proj")
		lin(b.Ctx.K, p+"attn.add_k_proj")
		lin(b.Ctx.V, p+"attn.add_v_proj")
		cp2(b.Ctx.NormQ, p+"attn.norm_added_q")
		cp2(b.Ctx.NormK, p+"attn.norm_added_k")
		lin(b.Out, p+"attn.to_out.0")
		lin(b.CtxOut, p+"attn.to_add_out")
		lin(b.FFIn, p+"ff.linear_in")
		lin(b.FFOut, p+"ff.linear_out")
		lin(b.CtxFFIn, p+"ff_context.linear_in")
		lin(b.CtxFFOut, p+"ff_context.linear_out")
	}
	for i, b := range m.Single {
		p := fmt.Sprintf("single_transformer_blocks.%d.attn.", i)
		lin(b.QKVMLP, p+"to_qkv_mlp_proj")
		cp2(b.NormQ, p+"norm_q")
		cp2(b.NormK, p+"norm_k")
		lin(b.Out, p+"to_out")
	}
	lin(m.NormOut, "norm_out.linear")
	lin(m.ProjOut, "proj_out")
	if lerr != nil {
		return nil, fmt.Errorf("LoadFlux2Transformer: %w", lerr)
	}
	return m, nil
}
