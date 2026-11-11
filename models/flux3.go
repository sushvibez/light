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

// Flux3Config is diffusers' Flux3Transformer2DModel config (FLUX 3 Image
// Klein / Klein Base).
type Flux3Config struct {
	InChannels       int     `json:"in_channels"`
	CondInChannels   int     `json:"cond_in_channels"`
	NumLayers        int     `json:"num_layers"`
	NumSingleLayers  int     `json:"num_single_layers"`
	HeadDim          int     `json:"attention_head_dim"`
	NumHeads         int     `json:"num_attention_heads"`
	JointDim         int     `json:"joint_attention_dim"`
	TimestepChannels int     `json:"timestep_channels"`
	MLPRatio         float64 `json:"mlp_ratio"`
	AxesDims         []int   `json:"axes_dims_rope"`
	RopeTheta        float64 `json:"rope_theta"`
	Eps              float64 `json:"eps"`
	FlowMap          bool    `json:"flow_map"`
}

func (c Flux3Config) inner() int { return c.NumHeads * c.HeadDim }
func (c Flux3Config) mlp() int   { return int(float64(c.inner()) * c.MLPRatio) }

// flux3Block is one Flux3 transformer block: parallel attention + SwiGLU
// feed-forward from the same modulated input, gated into the residual. The
// per-mode blocks and the joint single blocks share this layout.
type flux3Block struct {
	Attn        fluxAttn
	Out         *nn.Linear
	FFIn, FFOut *nn.Linear
}

// Flux3 token modes, in the order of the modulation heads.
const (
	flux3Image = iota
	flux3Cond
	flux3Txt
)

// Flux3Transformer is the FLUX 3 DiT: the noisy image tokens, the optional
// reference image tokens and the text context each run through their own
// per-mode blocks (self-attention within the mode), then join for the
// single blocks ([txt, image, refs]); only the image tokens reach the
// output. Modulation is per mode: the image at the flow timestep, the text
// and references at 0; a flow-map (step-distilled) model adds the embedding
// of the step's target timestep to every mode.
type Flux3Transformer struct {
	Cfg                 Flux3Config
	TimeL1, TimeL2      *nn.Linear
	TargetL1, TargetL2  *nn.Linear // flow map only
	EarlyMod, SingleMod [3]*nn.Linear
	XEmbed, CondEmbed   *nn.Linear
	ContextEmbed        *nn.Linear
	ModeImg, ModeCond   []*flux3Block
	ModeTxt, Single     []*flux3Block
	NormOut, ProjOut    *nn.Linear
	ones, zeros         *tensor.Tensor
}

// NewFlux3Transformer allocates an uninitialised F32 model.
func NewFlux3Transformer(cfg Flux3Config, dev device.Device) (*Flux3Transformer, error) {
	if cfg.TimestepChannels == 0 {
		cfg.TimestepChannels = 256
	}
	if cfg.TimestepChannels != 256 {
		return nil, fmt.Errorf("flux3: timestep_channels %d not supported (256)", cfg.TimestepChannels)
	}
	if cfg.CondInChannels == 0 {
		cfg.CondInChannels = cfg.InChannels
	}
	sum := 0
	for _, d := range cfg.AxesDims {
		sum += d
	}
	if sum != cfg.HeadDim {
		return nil, fmt.Errorf("flux3: axes_dims_rope %v sum to %d, head dim is %d", cfg.AxesDims, sum, cfg.HeadDim)
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
	block := func() *flux3Block {
		return &flux3Block{Attn: fluxAttn{Q: lin(D, D), K: lin(D, D), V: lin(D, D), NormQ: rms(), NormK: rms()},
			Out: lin(D, D), FFIn: lin(D, 2*M), FFOut: lin(M, D)}
	}
	m := &Flux3Transformer{Cfg: cfg,
		TimeL1: lin(256, D), TimeL2: lin(D, D),
		XEmbed: lin(cfg.InChannels, D), CondEmbed: lin(cfg.CondInChannels, D), ContextEmbed: lin(cfg.JointDim, D),
		NormOut: lin(D, 2*D), ProjOut: lin(D, cfg.InChannels)}
	if cfg.FlowMap {
		m.TargetL1, m.TargetL2 = lin(256, D), lin(D, D)
	}
	for i := range 3 {
		m.EarlyMod[i], m.SingleMod[i] = lin(D, 3*D), lin(D, 3*D)
	}
	for range cfg.NumLayers {
		m.ModeImg = append(m.ModeImg, block())
		m.ModeCond = append(m.ModeCond, block())
		m.ModeTxt = append(m.ModeTxt, block())
	}
	for range cfg.NumSingleLayers {
		m.Single = append(m.Single, block())
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

// Flux3Inputs are one transformer call's inputs (batch 1 positions shared
// by the batch, as in diffusers).
type Flux3Inputs struct {
	Image          *tensor.Tensor // (B, S_img, in_channels) packed noisy latents
	Text           *tensor.Tensor // (B, S_txt, joint_attention_dim)
	Cond           *tensor.Tensor // (B, S_cond, cond_in_channels) reference tokens, or nil
	Timestep       []float32      // B values in [0, 1]
	TargetTimestep []float32      // B values; flow-map models only
	ImgIDs         []float32      // (S_img, len(axes))
	TxtIDs         []float32      // (S_txt, len(axes))
	CondIDs        []float32      // (S_cond, len(axes))
}

// modulate is LayerNorm(x) (no affine) · (1 + scale) + shift.
func (m *Flux3Transformer) modulate(x, shift, scale *tensor.Tensor) (*tensor.Tensor, error) {
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

// mixer is attention(x) + ff(x) of an already modulated x (B, S, D).
func (m *Flux3Transformer) mixer(b *flux3Block, x, cos, sin *tensor.Tensor) (*tensor.Tensor, error) {
	q, k, v, err := b.Attn.qkv(x, m.Cfg.NumHeads, m.Cfg.HeadDim)
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
	f, err := b.FFIn.Forward(x)
	if err != nil {
		return nil, err
	}
	if f, err = swiglu(f); err != nil {
		return nil, err
	}
	if f, err = b.FFOut.Forward(f); err != nil {
		return nil, err
	}
	return ops.Add(a, f)
}

// modeBlock: x + gate · mixer(modulate(x)), attention within the mode.
func (m *Flux3Transformer) modeBlock(b *flux3Block, x *tensor.Tensor, mod []*tensor.Tensor, cos, sin *tensor.Tensor) (*tensor.Tensor, error) {
	n, err := m.modulate(x, mod[0], mod[1])
	if err != nil {
		return nil, err
	}
	o, err := m.mixer(b, n, cos, sin)
	if err != nil {
		return nil, err
	}
	return gateAdd(x, mod[2], o)
}

// singleBlock runs the joint block over [ctx, content] with per-part
// modulation; returns the updated ctx and content.
func (m *Flux3Transformer) singleBlock(b *flux3Block, ctx, content *tensor.Tensor, cmod, mod []*tensor.Tensor, cos, sin *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	nc, err := m.modulate(ctx, cmod[0], cmod[1])
	if err != nil {
		return nil, nil, err
	}
	nx, err := m.modulate(content, mod[0], mod[1])
	if err != nil {
		return nil, nil, err
	}
	joined, err := ops.Concat([]*tensor.Tensor{nc, nx}, 1)
	if err != nil {
		return nil, nil, err
	}
	o, err := m.mixer(b, joined, cos, sin)
	if err != nil {
		return nil, nil, err
	}
	St := ctx.Dim(1)
	oc, err := o.Narrow(1, 0, St)
	if err != nil {
		return nil, nil, err
	}
	ox, err := o.Narrow(1, St, content.Dim(1))
	if err != nil {
		return nil, nil, err
	}
	if ctx, err = gateAdd(ctx, cmod[2], oc); err != nil {
		return nil, nil, err
	}
	if content, err = gateAdd(content, mod[2], ox); err != nil {
		return nil, nil, err
	}
	return ctx, content, nil
}

// Forward predicts the velocity of the image tokens (B, S_img, in_channels).
func (m *Flux3Transformer) Forward(in Flux3Inputs) (*tensor.Tensor, error) {
	dev := in.Image.Device()
	cfg := m.Cfg
	B, Si, St := in.Image.Dim(0), in.Image.Dim(1), in.Text.Dim(1)
	if (in.TargetTimestep != nil) != cfg.FlowMap {
		return nil, fmt.Errorf("flux3: target timestep is required by, and only by, a flow-map model")
	}
	if (in.Cond == nil) != (in.CondIDs == nil) {
		return nil, fmt.Errorf("flux3: reference tokens and their ids go together")
	}
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
	imgT, err := embed(in.Timestep, m.TimeL1, m.TimeL2)
	if err != nil {
		return nil, err
	}
	zeroT, err := embed(make([]float32, B), m.TimeL1, m.TimeL2)
	if err != nil {
		return nil, err
	}
	if cfg.FlowMap {
		tgt, err := embed(in.TargetTimestep, m.TargetL1, m.TargetL2)
		if err != nil {
			return nil, err
		}
		if imgT, err = ops.Add(imgT, tgt); err != nil {
			return nil, err
		}
		if zeroT, err = ops.Add(zeroT, tgt); err != nil {
			return nil, err
		}
	}
	imgAct, err := ops.SiLU(imgT)
	if err != nil {
		return nil, err
	}
	zeroAct, err := ops.SiLU(zeroT)
	if err != nil {
		return nil, err
	}
	mods := func(mode int, act *tensor.Tensor) (early, single []*tensor.Tensor, err error) {
		if early, err = modulation(m.EarlyMod[mode], act, 3); err != nil {
			return
		}
		single, err = modulation(m.SingleMod[mode], act, 3)
		return
	}
	txtEarly, txtSingle, err := mods(flux3Txt, zeroAct)
	if err != nil {
		return nil, err
	}
	imgEarly, imgSingle, err := mods(flux3Image, imgAct)
	if err != nil {
		return nil, err
	}
	rope := func(ids []float32, S int) (*tensor.Tensor, *tensor.Tensor, error) {
		return fluxRoPETables(cfg.AxesDims, cfg.RopeTheta, cfg.HeadDim, ids, S, dev)
	}
	ctx, err := m.ContextEmbed.Forward(in.Text)
	if err != nil {
		return nil, err
	}
	tCos, tSin, err := rope(in.TxtIDs, St)
	if err != nil {
		return nil, err
	}
	x, err := m.XEmbed.Forward(in.Image)
	if err != nil {
		return nil, err
	}
	iCos, iSin, err := rope(in.ImgIDs, Si)
	if err != nil {
		return nil, err
	}
	for i, b := range m.ModeTxt {
		if ctx, err = m.modeBlock(b, ctx, txtEarly, tCos, tSin); err != nil {
			return nil, fmt.Errorf("flux3: text mode block %d: %w", i, err)
		}
	}
	for i, b := range m.ModeImg {
		if x, err = m.modeBlock(b, x, imgEarly, iCos, iSin); err != nil {
			return nil, fmt.Errorf("flux3: image mode block %d: %w", i, err)
		}
	}
	content, contentMod := x, imgSingle
	jCosParts, jSinParts := []*tensor.Tensor{tCos, iCos}, []*tensor.Tensor{tSin, iSin}
	if in.Cond != nil {
		Sc := in.Cond.Dim(1)
		condEarly, condSingle, err := mods(flux3Cond, zeroAct)
		if err != nil {
			return nil, err
		}
		c, err := m.CondEmbed.Forward(in.Cond)
		if err != nil {
			return nil, err
		}
		cCos, cSin, err := rope(in.CondIDs, Sc)
		if err != nil {
			return nil, err
		}
		for i, b := range m.ModeCond {
			if c, err = m.modeBlock(b, c, condEarly, cCos, cSin); err != nil {
				return nil, fmt.Errorf("flux3: reference mode block %d: %w", i, err)
			}
		}
		if content, err = ops.Concat([]*tensor.Tensor{x, c}, 1); err != nil {
			return nil, err
		}
		// Per-token modulation: the image's for its tokens, the references' for theirs.
		D := cfg.inner()
		contentMod = make([]*tensor.Tensor, 3)
		for j := range 3 {
			a, err := imgSingle[j].Expand(B, Si, D)
			if err != nil {
				return nil, err
			}
			bb, err := condSingle[j].Expand(B, Sc, D)
			if err != nil {
				return nil, err
			}
			if contentMod[j], err = ops.Concat([]*tensor.Tensor{a, bb}, 1); err != nil {
				return nil, err
			}
		}
		jCosParts, jSinParts = append(jCosParts, cCos), append(jSinParts, cSin)
	}
	jCos, err := ops.Concat(jCosParts, 1)
	if err != nil {
		return nil, err
	}
	jSin, err := ops.Concat(jSinParts, 1)
	if err != nil {
		return nil, err
	}
	for i, b := range m.Single {
		if ctx, content, err = m.singleBlock(b, ctx, content, txtSingle, contentMod, jCos, jSin); err != nil {
			return nil, fmt.Errorf("flux3: single block %d: %w", i, err)
		}
	}
	if x, err = content.Narrow(1, 0, Si); err != nil {
		return nil, err
	}
	out, err := modulation(m.NormOut, imgAct, 2) // (scale, shift)
	if err != nil {
		return nil, err
	}
	if x, err = m.modulate(x, out[1], out[0]); err != nil {
		return nil, err
	}
	return m.ProjOut.Forward(x)
}

// LoadFlux3Transformer loads a Flux3Transformer2DModel from a directory with
// config.json and its safetensors (a pipeline's transformer folder).
func LoadFlux3Transformer(dir string, dev device.Device) (*Flux3Transformer, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	cfg := Flux3Config{InChannels: 128, CondInChannels: 128, MLPRatio: 3, RopeTheta: 10000, Eps: 1e-6, TimestepChannels: 256}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("flux3 config: %w", err)
	}
	hf, err := loadCheckpointTensors(dir)
	if err != nil {
		return nil, err
	}
	m, err := NewFlux3Transformer(cfg, dev)
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
	block := func(b *flux3Block, p string) {
		lin(b.Attn.Q, p+"attn.to_q")
		lin(b.Attn.K, p+"attn.to_k")
		lin(b.Attn.V, p+"attn.to_v")
		cp(b.Attn.NormQ.Weight, p+"attn.norm_q.weight")
		cp(b.Attn.NormK.Weight, p+"attn.norm_k.weight")
		lin(b.Out, p+"attn.to_out.0")
		lin(b.FFIn, p+"ff.linear_in")
		lin(b.FFOut, p+"ff.linear_out")
	}
	lin(m.TimeL1, "timestep_embedder.linear_1")
	lin(m.TimeL2, "timestep_embedder.linear_2")
	if m.TargetL1 != nil {
		lin(m.TargetL1, "target_timestep_embedder.linear_1")
		lin(m.TargetL2, "target_timestep_embedder.linear_2")
	}
	for i, name := range []string{"image", "image_cond", "txt"} {
		lin(m.EarlyMod[i], "early_stream_modulations."+name+".linear")
		lin(m.SingleMod[i], "single_stream_modulations."+name+".linear")
	}
	lin(m.XEmbed, "x_embedders.image")
	lin(m.CondEmbed, "x_embedders.image_cond")
	lin(m.ContextEmbed, "context_embedder")
	for i := range cfg.NumLayers {
		block(m.ModeImg[i], fmt.Sprintf("mode_blocks.image.%d.", i))
		block(m.ModeCond[i], fmt.Sprintf("mode_blocks.image_cond.%d.", i))
		block(m.ModeTxt[i], fmt.Sprintf("txt_mode_blocks.%d.", i))
	}
	for i, b := range m.Single {
		block(b, fmt.Sprintf("single_transformer_blocks.%d.", i))
	}
	lin(m.NormOut, "norm_out.linear")
	lin(m.ProjOut, "proj_out")
	if lerr != nil {
		return nil, fmt.Errorf("LoadFlux3Transformer: %w", lerr)
	}
	return m, nil
}
