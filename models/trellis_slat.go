package models

import (
	"fmt"
	"math"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/splat"
	"light/tensor"
)

// TRELLIS stage 2: the structured latent (SLat), 8 channels on each occupied
// voxel of stage 1's 64³ grid, and its Gaussian decoder.
//
//	TrellisSLatFlow     SLatFlowModel: sparse 3³-conv residual blocks
//	                    around a downsample by 2, then adaLN cross-attention
//	                    transformer blocks with full self attention over the
//	                    coarse voxels (APE of their coords), then upsample
//	                    and residual blocks with the skips
//	TrellisSLatSample   FlowEulerGuidanceIntervalSampler over the voxel rows
//	TrellisSLatDecoder  SLatGaussianDecoder: a sparse transformer with
//	                    shifted-window (swin) attention, 32 Gaussians per
//	                    voxel (to_representation in Gaussians)

// TrellisSLatFlowArgs are SLatFlowModel's constructor arguments.
type TrellisSLatFlowArgs struct {
	Resolution      int     `json:"resolution"`
	InChannels      int     `json:"in_channels"`
	OutChannels     int     `json:"out_channels"`
	ModelChannels   int     `json:"model_channels"`
	CondChannels    int     `json:"cond_channels"`
	NumBlocks       int     `json:"num_blocks"`
	NumHeads        int     `json:"num_heads"`
	NumHeadChannels int     `json:"num_head_channels"`
	MLPRatio        float64 `json:"mlp_ratio"`
	PatchSize       int     `json:"patch_size"`
	NumIORes        int     `json:"num_io_res_blocks"`
	IOChannels      []int   `json:"io_block_channels"`
	PEMode          string  `json:"pe_mode"`
	UseSkip         *bool   `json:"use_skip_connection"`
	ShareMod        bool    `json:"share_mod"`
	QKRMSNorm       bool    `json:"qk_rms_norm"`
	QKRMSNormCross  bool    `json:"qk_rms_norm_cross"`
	UseFP16         bool    `json:"use_fp16"`
}

// TrellisRepresentationConfig is SLatGaussianDecoder's representation_config.
type TrellisRepresentationConfig struct {
	LR                map[string]float64 `json:"lr"`
	PerturbOffset     bool               `json:"perturb_offset"`
	VoxelSize         float64            `json:"voxel_size"`
	NumGaussians      int                `json:"num_gaussians"`
	FilterKernel2D    float64            `json:"2d_filter_kernel_size"`
	FilterKernel3D    float64            `json:"3d_filter_kernel_size"`
	ScalingBias       float64            `json:"scaling_bias"`
	OpacityBias       float64            `json:"opacity_bias"`
	ScalingActivation string             `json:"scaling_activation"`
}

// TrellisSLatDecoderArgs are SLatGaussianDecoder's constructor arguments.
type TrellisSLatDecoderArgs struct {
	Resolution      int                         `json:"resolution"`
	ModelChannels   int                         `json:"model_channels"`
	LatentChannels  int                         `json:"latent_channels"`
	NumBlocks       int                         `json:"num_blocks"`
	NumHeads        int                         `json:"num_heads"`
	NumHeadChannels int                         `json:"num_head_channels"`
	MLPRatio        float64                     `json:"mlp_ratio"`
	AttnMode        string                      `json:"attn_mode"`
	WindowSize      int                         `json:"window_size"`
	PEMode          string                      `json:"pe_mode"`
	QKRMSNorm       bool                        `json:"qk_rms_norm"`
	UseFP16         bool                        `json:"use_fp16"`
	Rep             TrellisRepresentationConfig `json:"representation_config"`
}

// --- flow model ------------------------------------------------------------------

// trellisSRes is SparseResBlock3d: (down / up by 2,) affine norm, SiLU,
// 3³ conv, norm modulated by the time (scale, shift), SiLU, conv, plus the
// (linear) skip of the resampled input.
type trellisSRes struct {
	Norm1        idolNorm
	Conv1, Conv2 *trellisSparseConv
	Emb          *nn.Linear // after SiLU: 2·out (scale, shift)
	Skip         *nn.Linear // nil: identity
	Down, Up     bool
	out          int
}

func (r *trellisSRes) forward(o *iop, c *idolCommon, g *TrellisGrid, x, temb *tensor.Tensor) (*tensor.Tensor, *TrellisGrid) {
	if o.err != nil {
		return nil, nil
	}
	emb := o.lin(r.Emb, o.un(ops.SiLU, temb))
	scale, shift := o.reshape(o.narrow(emb, 1, 0, r.out), 1, r.out), o.reshape(o.narrow(emb, 1, r.out, r.out), 1, r.out)
	if r.Down {
		x, g = g.downsample(o, x)
	} else if r.Up {
		x, g = g.upsample(o, x)
	}
	h := r.Conv1.forward(o, g, o.un(ops.SiLU, o.ln(x, r.Norm1.W, r.Norm1.B, 1e-6)))
	if o.err != nil {
		return nil, nil
	}
	h = idolModulate(o, c.normN(o, h, 1e-6), shift, scale)
	h = r.Conv2.forward(o, g, o.un(ops.SiLU, h))
	skip := x
	if r.Skip != nil {
		skip = o.lin(r.Skip, x)
	}
	return o.add(h, skip), g
}

// normN is a LayerNorm without affine over the last dim of any width.
func (c *idolCommon) normN(o *iop, x *tensor.Tensor, eps float64) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	C := x.Dim(-1)
	if c.normCache == nil {
		c.normCache = map[int][2]*tensor.Tensor{}
	}
	p, ok := c.normCache[C]
	if !ok {
		p[0] = o.do(func() (*tensor.Tensor, error) { return tensor.Ones(dtype.F32, []int{C}, c.dev) })
		p[1] = o.do(func() (*tensor.Tensor, error) { return tensor.Zeros(dtype.F32, []int{C}, c.dev) })
		if o.err != nil {
			return nil
		}
		c.normCache[C] = p
	}
	return o.ln(x, p[0], p[1], eps)
}

// TrellisSLatFlow is SLatFlowModel.
type TrellisSLatFlow struct {
	Args      TrellisSLatFlowArgs
	TEmbed    idolTimeEmbed
	AdaLN     *nn.Linear // share_mod only
	Input     *nn.Linear
	InBlocks  []*trellisSRes
	Blocks    []*trellisBlock
	OutBlocks []*trellisSRes
	Out       *nn.Linear
	c         *idolCommon
}

func (a *TrellisSLatFlowArgs) skip() bool { return a.UseSkip == nil || *a.UseSkip }

// LoadTrellisSLatFlow loads ckpts/<name> (base path without extension:
// .json and .safetensors) onto dev. Under nn.WithWeightStorage16 the
// linears (and the sparse convs' per-offset linears) keep 16-bit weights.
func LoadTrellisSLatFlow(base string, dev device.Device) (*TrellisSLatFlow, error) {
	a := TrellisSLatFlowArgs{NumHeadChannels: 64, MLPRatio: 4, PatchSize: 2, NumIORes: 2, PEMode: "ape"}
	if err := trellisConfig(base+".json", "SLatFlowModel", &a); err != nil {
		return nil, err
	}
	if a.NumHeads == 0 {
		a.NumHeads = a.ModelChannels / a.NumHeadChannels
	}
	if a.PEMode != "ape" {
		return nil, fmt.Errorf("trellis slat flow: pe_mode %q (only ape is supported)", a.PEMode)
	}
	if a.IOChannels != nil && 1<<len(a.IOChannels) != a.PatchSize {
		return nil, fmt.Errorf("trellis slat flow: patch size %d with %d io stages", a.PatchSize, len(a.IOChannels))
	}
	l, err := newIdolLoader(base + ".safetensors")
	if err != nil {
		return nil, err
	}
	l.dev = dev
	c, err := newIdolCommon(a.ModelChannels, a.NumHeads, dev)
	if err != nil {
		return nil, err
	}
	m := &TrellisSLatFlow{Args: a, c: c,
		TEmbed: idolTimeEmbed{L1: l.linear("t_embedder.mlp.0"), L2: l.linear("t_embedder.mlp.2")},
		Input:  l.linear("input_layer"), Out: l.linear("out_layer"),
	}
	if a.ShareMod {
		m.AdaLN = l.linear("adaLN_modulation.1")
	}
	res := func(p string, in, out int, down, up bool) *trellisSRes {
		r := &trellisSRes{Norm1: l.norm(p + ".norm1"), Conv1: l.sparseConv(p + ".conv1"), Conv2: l.sparseConv(p + ".conv2"),
			Emb: l.linear(p + ".emb_layers.1"), Down: down, Up: up, out: out}
		if in != out {
			r.Skip = l.linear(p + ".skip_connection")
		}
		return r
	}
	io := a.IOChannels
	n := 0
	for i, ch := range io {
		next := a.ModelChannels
		if i+1 < len(io) {
			next = io[i+1]
		}
		for range a.NumIORes - 1 {
			m.InBlocks = append(m.InBlocks, res(fmt.Sprintf("input_blocks.%d", n), ch, ch, false, false))
			n++
		}
		m.InBlocks = append(m.InBlocks, res(fmt.Sprintf("input_blocks.%d", n), ch, next, true, false))
		n++
	}
	m.Blocks = l.trellisBlocks("blocks", a.NumBlocks, a.ShareMod, a.QKRMSNorm, a.QKRMSNormCross)
	n = 0
	mul := 1
	if a.skip() {
		mul = 2
	}
	for i := len(io) - 1; i >= 0; i-- {
		ch, prev := io[i], a.ModelChannels
		if i+1 < len(io) {
			prev = io[i+1]
		}
		m.OutBlocks = append(m.OutBlocks, res(fmt.Sprintf("out_blocks.%d", n), prev*mul, ch, false, true))
		n++
		for range a.NumIORes - 1 {
			m.OutBlocks = append(m.OutBlocks, res(fmt.Sprintf("out_blocks.%d", n), ch*mul, ch, false, false))
			n++
		}
	}
	if l.err != nil {
		return nil, fmt.Errorf("trellis slat flow: %w", l.err)
	}
	return m, nil
}

// trellisBlocks loads ModulatedTransformerCrossBlocks (dense or sparse:
// the same layers) <prefix>.0 ... <prefix>.<n-1>.
func (l *idolLoader) trellisBlocks(prefix string, n int, shareMod, qk, qkCross bool) []*trellisBlock {
	mha := func(name string, cross, norm bool) *idolMHA {
		h := &idolMHA{Out: l.linear(name + ".to_out")}
		if cross {
			h.Q, h.KV = l.linear(name+".to_q"), l.linear(name+".to_kv")
		} else {
			h.QKV = l.linear(name + ".to_qkv")
		}
		if norm {
			h.QNorm, h.KNorm = l.param(name+".q_rms_norm.gamma"), l.param(name+".k_rms_norm.gamma")
		}
		return h
	}
	var out []*trellisBlock
	for i := range n {
		p := fmt.Sprintf("%s.%d", prefix, i)
		b := &trellisBlock{
			Self: mha(p+".self_attn", false, qk), Cross: mha(p+".cross_attn", true, qkCross),
			Norm2: l.norm(p + ".norm2"), FFN: l.ffn(p + ".mlp"),
		}
		if !shareMod {
			b.AdaLN = l.linear(p + ".adaLN_modulation.1")
		}
		out = append(out, b)
	}
	return out
}

// Forward returns the velocity (N, out) of the latent x (N, in) on the
// voxels of g at time t (one value, in [0, 1000]) given the image tokens
// cond (1, L, cond_channels).
func (m *TrellisSLatFlow) Forward(g *TrellisGrid, x *tensor.Tensor, t []float32, cond *tensor.Tensor) (*tensor.Tensor, error) {
	a := m.Args
	if x.Rank() != 2 || x.Dim(0) != g.N() || x.Dim(1) != a.InChannels {
		return nil, fmt.Errorf("trellis slat flow: input %v, want (%d, %d)", x.Shape(), g.N(), a.InChannels)
	}
	if len(t) != 1 || cond.Dim(0) != 1 {
		return nil, fmt.Errorf("trellis slat flow: one object at a time (t %v, cond %v)", t, cond.Shape())
	}
	o := &iop{}
	c := m.c
	h := o.lin(m.Input, x)
	freq := o.do(func() (*tensor.Tensor, error) { return idolSinusoidal(t, 10000, false, c.dev) })
	temb := m.TEmbed.forward(o, freq)
	if m.AdaLN != nil {
		temb = o.lin(m.AdaLN, o.un(ops.SiLU, temb))
	}
	lv := g
	skips := make([]*tensor.Tensor, 0, len(m.InBlocks))
	for _, b := range m.InBlocks {
		h, lv = b.forward(o, c, lv, h, temb)
		skips = append(skips, h)
	}
	if o.err != nil {
		return nil, o.err
	}
	n := lv.N()
	h = o.reshape(o.add(h, lv.posEmbed(o, a.ModelChannels)), 1, n, a.ModelChannels)
	for _, b := range m.Blocks {
		h = b.forward(o, c, h, temb, cond)
	}
	h = o.reshape(h, n, a.ModelChannels)
	for i, b := range m.OutBlocks {
		if a.skip() {
			h = o.cat(1, h, skips[len(skips)-1-i])
		}
		h, lv = b.forward(o, c, lv, h, temb)
	}
	out := o.lin(m.Out, c.normN(o, h, 1e-5))
	return out, o.err
}

// trellisEulerSample is FlowEulerGuidanceIntervalSampler.sample over any
// velocity function v(x, t, cond) (see TrellisSample).
func trellisEulerSample(v func(x *tensor.Tensor, t []float32, cond *tensor.Tensor) (*tensor.Tensor, error),
	batch int, noise, cond, negCond *tensor.Tensor, o TrellisSampleOptions) (*tensor.Tensor, error) {
	o.defaults()
	var err error
	if negCond == nil && o.CFG > 0 {
		if negCond, err = tensor.Zeros(cond.DType(), cond.Shape().Dims(), cond.Device()); err != nil {
			return nil, err
		}
	}
	grid := make([]float64, o.Steps+1)
	for i := range grid {
		t := float64(i)*(-1/float64(o.Steps)) + 1 // np.linspace(1, 0, steps + 1)
		if i == o.Steps {
			t = 0
		}
		grid[i] = o.RescaleT * t / (1 + (o.RescaleT-1)*t)
	}
	x := noise
	for i := range o.Steps {
		t, tPrev := grid[i], grid[i+1]
		ts := make([]float32, batch)
		for b := range ts {
			ts[b] = float32(1000 * t)
		}
		vc, err := v(x, ts, cond)
		if err != nil {
			return nil, err
		}
		if o.CFG > 0 && o.Interval[0] <= t && t <= o.Interval[1] {
			u, err := v(x, ts, negCond)
			if err != nil {
				return nil, err
			}
			if vc, err = idolGuide(vc, u, o.CFG); err != nil {
				return nil, err
			}
		}
		if x, err = idolEuler(x, vc, t-tPrev); err != nil {
			return nil, err
		}
		if o.Step != nil {
			o.Step(i, o.Steps)
		}
	}
	return x, nil
}

// TrellisSLatSample is sample_slat's sampler run: FlowEulerGuidanceIntervalSampler
// from noise (N, in) on the voxels of g, before the slat_normalization
// (TrellisSLatUnnormalize). negCond nil means zeros.
func TrellisSLatSample(m *TrellisSLatFlow, g *TrellisGrid, noise, cond, negCond *tensor.Tensor, o TrellisSampleOptions) (*tensor.Tensor, error) {
	return trellisEulerSample(func(x *tensor.Tensor, t []float32, c *tensor.Tensor) (*tensor.Tensor, error) {
		return m.Forward(g, x, t, c)
	}, 1, noise, cond, negCond, o)
}

// TrellisSLatUnnormalize is slat · std + mean per channel (pipeline.json's
// slat_normalization) for x (N, C).
func TrellisSLatUnnormalize(x *tensor.Tensor, mean, std []float64) (*tensor.Tensor, error) {
	C := x.Dim(1)
	if len(mean) != C || len(std) != C {
		return nil, fmt.Errorf("trellis: slat_normalization has %d / %d values for %d channels", len(mean), len(std), C)
	}
	m, s := make([]float32, C), make([]float32, C)
	for i := range C {
		m[i], s[i] = float32(mean[i]), float32(std[i])
	}
	mt, err := tensor.New(m, []int{1, C}, x.Device())
	if err != nil {
		return nil, err
	}
	st, err := tensor.New(s, []int{1, C}, x.Device())
	if err != nil {
		return nil, err
	}
	y, err := ops.Mul(x, st)
	if err != nil {
		return nil, err
	}
	return ops.Add(y, mt)
}

// --- Gaussian decoder ------------------------------------------------------------

// trellisSwinBlock is SparseTransformerBlock with attn_mode "windowed":
// norm (no affine), windowed self attention, norm, FFN.
type trellisSwinBlock struct {
	QKV, Out     *nn.Linear
	QNorm, KNorm *tensor.Tensor // qk_rms_norm only
	FFN          idolFFN
	Shift        int
}

// trellisSwinTorso is SparseTransformerBase with attn_mode "swin" and APE,
// shared by the SLat decoders: input layer plus the coords' position
// embedding, then the blocks (window W, odd blocks shifted by W / 2).
type trellisSwinTorso struct {
	Input  *nn.Linear
	Blocks []*trellisSwinBlock
	window int
	c      *idolCommon
}

// swinTorso loads input_layer and blocks.0 ... blocks.<n-1>.
func (l *idolLoader) swinTorso(C, H, n, window int, qk bool, dev device.Device) (trellisSwinTorso, error) {
	c, err := newIdolCommon(C, H, dev)
	if err != nil {
		return trellisSwinTorso{}, err
	}
	t := trellisSwinTorso{Input: l.linear("input_layer"), window: window, c: c}
	for i := range n {
		p := fmt.Sprintf("blocks.%d", i)
		b := &trellisSwinBlock{QKV: l.linear(p + ".attn.to_qkv"), Out: l.linear(p + ".attn.to_out"),
			FFN: l.ffn(p + ".mlp"), Shift: window / 2 * (i % 2)}
		if qk {
			b.QNorm, b.KNorm = l.param(p+".attn.q_rms_norm.gamma"), l.param(p+".attn.k_rms_norm.gamma")
		}
		t.Blocks = append(t.Blocks, b)
	}
	return t, nil
}

// forward runs the torso on x (N, in) on the voxels of g: (N, C).
func (t *trellisSwinTorso) forward(o *iop, g *TrellisGrid, x *tensor.Tensor) *tensor.Tensor {
	c := t.c
	N, C, H := g.N(), c.C, c.H
	h := o.add(o.lin(t.Input, x), g.posEmbed(o, C))
	for _, b := range t.Blocks {
		qkv := o.lin(b.QKV, c.norm(o, idolNorm{}, h))
		if b.QNorm != nil {
			s := o.reshape(qkv, N, 3, H, C/H)
			q := c.mhRMS(o, o.reshape(o.narrow(s, 1, 0, 1), N, 1, H, C/H), b.QNorm)
			k := c.mhRMS(o, o.reshape(o.narrow(s, 1, 1, 1), N, 1, H, C/H), b.KNorm)
			qkv = o.reshape(o.cat(1, q, k, o.narrow(s, 1, 2, 1)), N, 3*C)
		}
		h = o.add(h, o.lin(b.Out, g.windowedAttend(o, qkv, H, t.window, b.Shift)))
		h = o.add(h, b.FFN.forward(o, c.norm(o, idolNorm{}, h)))
	}
	return h
}

// TrellisSLatDecoder is SLatGaussianDecoder.
type TrellisSLatDecoder struct {
	Args TrellisSLatDecoderArgs
	trellisSwinTorso
	Out          *nn.Linear
	Perturbation []float32 // (G, 3) offset_perturbation
}

// trellisHammersley is hammersley_sequence(3, n, num): (n / num,
// radical_inverse(2, n), radical_inverse(3, n)).
func trellisHammersley(n, num int) [3]float64 {
	ri := func(base, n int) float64 {
		v, inv := 0.0, 1/float64(base)
		f := inv
		for n > 0 {
			v += float64(n%base) * f
			n /= base
			f *= inv
		}
		return v
	}
	return [3]float64{float64(n) / float64(num), ri(2, n), ri(3, n)}
}

// LoadTrellisSLatDecoder loads ckpts/<name> (base path without extension)
// onto dev.
func LoadTrellisSLatDecoder(base string, dev device.Device) (*TrellisSLatDecoder, error) {
	a := TrellisSLatDecoderArgs{NumHeadChannels: 64, MLPRatio: 4, AttnMode: "swin", WindowSize: 8, PEMode: "ape"}
	if err := trellisConfig(base+".json", "SLatGaussianDecoder", &a); err != nil {
		return nil, err
	}
	if a.NumHeads == 0 {
		a.NumHeads = a.ModelChannels / a.NumHeadChannels
	}
	if a.AttnMode != "swin" {
		return nil, fmt.Errorf("trellis gaussian decoder: attn_mode %q (only swin is supported)", a.AttnMode)
	}
	if a.PEMode != "ape" {
		return nil, fmt.Errorf("trellis gaussian decoder: pe_mode %q (only ape is supported)", a.PEMode)
	}
	r := a.Rep
	if r.NumGaussians <= 0 || r.ScalingActivation != "softplus" && r.ScalingActivation != "exp" {
		return nil, fmt.Errorf("trellis gaussian decoder: representation_config %+v", r)
	}
	l, err := newIdolLoader(base + ".safetensors")
	if err != nil {
		return nil, err
	}
	l.dev = dev
	torso, err := l.swinTorso(a.ModelChannels, a.NumHeads, a.NumBlocks, a.WindowSize, a.QKRMSNorm, dev)
	if err != nil {
		return nil, err
	}
	m := &TrellisSLatDecoder{Args: a, trellisSwinTorso: torso, Out: l.linear("out_layer")}
	G := r.NumGaussians
	if _, ok := l.hf["offset_perturbation"]; ok {
		m.Perturbation = l.host("offset_perturbation")
	} else {
		m.Perturbation = make([]float32, 3*G)
		for i := range G {
			h := trellisHammersley(i, G)
			for k := range 3 {
				m.Perturbation[3*i+k] = float32(math.Atanh(float64(float32(float32(h[k])*2-1) / float32(r.VoxelSize))))
			}
		}
	}
	if l.err != nil {
		return nil, fmt.Errorf("trellis gaussian decoder: %w", l.err)
	}
	if len(m.Perturbation) != 3*G {
		return nil, fmt.Errorf("trellis gaussian decoder: offset_perturbation has %d values, want %d", len(m.Perturbation), 3*G)
	}
	if m.Out.W.Dim(0) != 14*G {
		return nil, fmt.Errorf("trellis gaussian decoder: out_layer has %d outputs, want %d", m.Out.W.Dim(0), 14*G)
	}
	return m, nil
}

// Forward returns the per-voxel Gaussian parameters (N, 14·G) of the
// (un-normalised) structured latent x (N, latent) on the voxels of g.
func (m *TrellisSLatDecoder) Forward(g *TrellisGrid, x *tensor.Tensor) (*tensor.Tensor, error) {
	a := m.Args
	if x.Rank() != 2 || x.Dim(0) != g.N() || x.Dim(1) != a.LatentChannels {
		return nil, fmt.Errorf("trellis gaussian decoder: input %v, want (%d, %d)", x.Shape(), g.N(), a.LatentChannels)
	}
	o := &iop{}
	h := m.trellisSwinTorso.forward(o, g, x)
	out := o.lin(m.Out, m.c.norm(o, idolFinalNorm, h))
	return out, o.err
}

// Gaussians is to_representation for one object: the decoder's outputs
// (N, 14·G) on the voxels of g to activated Gaussians (N·G, voxel-major)
// in TRELLIS's z-up frame inside [-0.5, 0.5]³ (Gaussian's aabb): positions
// (voxel centre + tanh(offset·lr + perturbation)/res·voxel_size/2 − 0.5),
// degree-0 SH colours, scales sqrt(act(s·lr + bias)² + filter²) with
// act softplus (or exp), unit rotations normalize(r·lr + (1, 0, 0, 0))
// and opacities sigmoid(o·lr + logit(opacity_bias)).
func (m *TrellisSLatDecoder) Gaussians(g *TrellisGrid, feats *tensor.Tensor) (*splat.Gaussians, error) {
	f, err := feats.To(device.CPU())
	if err != nil {
		return nil, err
	}
	if f.DType() != dtype.F32 {
		if f, err = ops.Cast(f, dtype.F32); err != nil {
			return nil, err
		}
	}
	if f, err = f.Contiguous(); err != nil {
		return nil, err
	}
	fv := tensor.MustSlice[float32](f)
	r := m.Args.Rep
	G := r.NumGaussians
	W := 14 * G
	N := g.N()
	if len(fv) != N*W {
		return nil, fmt.Errorf("trellis gaussians: %d values for %d voxels × %d", len(fv), N, W)
	}
	lr := func(k string) float32 {
		if v, ok := r.LR[k]; ok {
			return float32(v)
		}
		return 1
	}
	lrX, lrF, lrS, lrR, lrO := lr("_xyz"), lr("_features_dc"), lr("_scaling"), lr("_rotation"), lr("_opacity")
	// The biases as Gaussian.setup_functions computes them (F32 tensors).
	sb := float32(r.ScalingBias)
	var scaleBias float32
	act := func(x float64) float64 { return math.Exp(x) }
	if r.ScalingActivation == "softplus" {
		scaleBias = float32(float64(sb) + math.Log(-math.Expm1(-float64(sb))))
		act = func(x float64) float64 { // F.softplus: beta 1, threshold 20
			if x > 20 {
				return x
			}
			return math.Log1p(math.Exp(x))
		}
	} else {
		scaleBias = float32(math.Log(float64(sb)))
	}
	ob := float32(r.OpacityBias)
	opBias := float32(math.Log(float64(ob / (1 - ob))))
	k2 := r.FilterKernel3D * r.FilterKernel3D
	res := float32(m.Args.Resolution)
	half := float32(0.5 * r.VoxelSize)
	out := &splat.Gaussians{XYZ: make([]float32, 3*N*G), FeaturesDC: make([]float32, 3*N*G), Scaling: make([]float32, 3*N*G),
		Rotation: make([]float32, 4*N*G), Opacity: make([]float32, N*G)}
	for v := range N {
		row := fv[v*W : (v+1)*W]
		cen := g.Coords[v]
		for j := range G {
			i := v*G + j
			for k := range 3 {
				off := row[3*j+k] * lrX
				if r.PerturbOffset {
					off += m.Perturbation[3*j+k]
				}
				xyz := (float32(cen[k])+0.5)/res + float32(math.Tanh(float64(off)))/res*half
				out.XYZ[3*i+k] = xyz - 0.5
				out.FeaturesDC[3*i+k] = row[3*G+3*j+k] * lrF
				s := act(float64(row[6*G+3*j+k]*lrS + scaleBias))
				out.Scaling[3*i+k] = float32(math.Sqrt(s*s + k2))
			}
			var q [4]float64
			var qn float64
			for k := range 4 {
				q[k] = float64(row[9*G+4*j+k] * lrR)
				if k == 0 {
					q[k]++
				}
				qn += q[k] * q[k]
			}
			qn = max(math.Sqrt(qn), 1e-12)
			for k := range 4 {
				out.Rotation[4*i+k] = float32(q[k] / qn)
			}
			out.Opacity[i] = float32(1 / (1 + math.Exp(-float64(row[13*G+j]*lrO+opBias))))
		}
	}
	return out, nil
}
