package models

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/safetensors"
	"light/tensor"
)

// Idol (speridlabs/idol, idol-sdk): a single image to 3D Gaussians. Three
// networks share one config.json:
//
//	IdolDenoiser       rectified-flow transformer over 8192 latent tokens
//	                   (16 ch) and one camera token (5 ch), conditioned on
//	                   DINOv3 + FLUX.2 VAE image tokens; learned ("RePo")
//	                   rotary positions predicted from the hidden states.
//	IdolOctreeDecoder  per-cell child logits, descended level by level to
//	                   place anchor points (IdolSampleAnchors).
//	IdolGaussianDecoder each anchor -> 32 Gaussians (offset, colour, scale,
//	                   rotation, opacity).
//
// Layer names follow idol-sdk's state dicts (idol/models/*.py).

// IdolConfig is speridlabs/idol's config.json.
type IdolConfig struct {
	Denoiser struct {
		Args IdolDenoiserArgs `json:"args"`
	} `json:"denoiser"`
	Decoder struct {
		Args IdolDecoderArgs `json:"args"`
	} `json:"decoder"`
	DecoderGS struct {
		Args IdolDecoderArgs `json:"args"`
	} `json:"decoder_gs"`
	ImageSize     int               `json:"image_size"`
	MaxVoxelLevel int               `json:"max_voxel_level"`
	ImageEncoders map[string]string `json:"image_encoders"`
}

// IdolDenoiserArgs are LatentSeqMMFlowModel's constructor arguments.
type IdolDenoiserArgs struct {
	QTokenLength     int     `json:"q_token_length"`
	InChannels       int     `json:"in_channels"`
	CamChannels      int     `json:"cam_channels"`
	ModelChannels    int     `json:"model_channels"`
	CondChannels     int     `json:"cond_channels"`
	Cond2Channels    int     `json:"cond2_channels"`
	NumBlocks        int     `json:"num_blocks"`
	NumRefinerBlocks int     `json:"num_refiner_blocks"`
	NumHeads         int     `json:"num_heads"`
	MLPRatio         float64 `json:"mlp_ratio"`
}

// IdolDecoderArgs are the two decoders' constructor arguments.
type IdolDecoderArgs struct {
	InChannels     int     `json:"in_channels"`
	ModelChannels  int     `json:"model_channels"`
	CondChannels   int     `json:"cond_channels"`
	NumBlocks      int     `json:"num_blocks"`
	NumHeads       int     `json:"num_heads"`
	MLPRatio       float64 `json:"mlp_ratio"`
	Representation struct {
		NumGaussians     int                `json:"num_gaussians"`
		LR               map[string]float64 `json:"lr"`
		PerturbeSize     float64            `json:"perturbe_size"`
		FilterKernelSize float64            `json:"3d_filter_kernel_size"`
		ScalingBias      float64            `json:"scaling_bias"`
		OpacityBias      float64            `json:"opacity_bias"`
	} `json:"representation_config"`
}

const idolHeadDim = 64 // RePo's head dim (num_head_channels), = channels / heads in every released config

// LoadIdolConfig reads <dir>/config.json.
func LoadIdolConfig(dir string) (*IdolConfig, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var c IdolConfig
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("idol config: %w", err)
	}
	return &c, nil
}

// --- op chaining -------------------------------------------------------------

// iop chains tensor ops, keeping the first error: every method is a no-op
// returning nil once one failed, and callers check o.err at the end.
type iop struct{ err error }

func (o *iop) do(f func() (*tensor.Tensor, error)) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	t, err := f()
	o.err = err
	return t
}

func (o *iop) lin(l *nn.Linear, x *tensor.Tensor) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) { return l.Forward(x) })
}
func (o *iop) add(a, b *tensor.Tensor) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) { return ops.Add(a, b) })
}
func (o *iop) sub(a, b *tensor.Tensor) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) { return ops.Sub(a, b) })
}
func (o *iop) mul(a, b *tensor.Tensor) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) { return ops.Mul(a, b) })
}
func (o *iop) affine(a *tensor.Tensor, m, c float64) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) { return ops.Affine(a, m, c) })
}
func (o *iop) un(f func(*tensor.Tensor) (*tensor.Tensor, error), a *tensor.Tensor) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) { return f(a) })
}
func (o *iop) reshape(a *tensor.Tensor, dims ...int) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) {
		c, err := a.Contiguous()
		if err != nil {
			return nil, err
		}
		return c.Reshape(dims...)
	})
}
func (o *iop) narrow(a *tensor.Tensor, dim, start, n int) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) { return a.Narrow(dim, start, n) })
}
func (o *iop) perm(a *tensor.Tensor, p ...int) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) { return a.Permute(p...) })
}
func (o *iop) cat(axis int, ts ...*tensor.Tensor) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) { return ops.Concat(ts, axis) })
}
func (o *iop) ln(x, w, b *tensor.Tensor, eps float64) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) { return ops.LayerNorm(x, w, b, eps) })
}

// --- shared layers -----------------------------------------------------------

// idolNorm is a LayerNorm: affine when W is set, else the model's ones /
// zeros. Eps 0 means the blocks' 1e-6; the final F.layer_norm calls and
// RePo's LayerNorm32 use PyTorch's default 1e-5 (idolFinalNorm).
type idolNorm struct {
	W, B *tensor.Tensor
	Eps  float64
}

var idolFinalNorm = idolNorm{Eps: 1e-5}

type idolCommon struct {
	C, H        int
	ones, zeros *tensor.Tensor // (C,)
	onesD       *tensor.Tensor // (head dim,)
	dev         device.Device
	normCache   map[int][2]*tensor.Tensor // normN's ones / zeros per width
}

func newIdolCommon(C, H int, dev device.Device) (*idolCommon, error) {
	c := &idolCommon{C: C, H: H, dev: dev}
	var err error
	if c.ones, err = tensor.Ones(dtype.F32, []int{C}, dev); err != nil {
		return nil, err
	}
	if c.zeros, err = tensor.Zeros(dtype.F32, []int{C}, dev); err != nil {
		return nil, err
	}
	c.onesD, err = tensor.Ones(dtype.F32, []int{C / H}, dev)
	return c, err
}

func (c *idolCommon) norm(o *iop, n idolNorm, x *tensor.Tensor) *tensor.Tensor {
	eps := n.Eps
	if eps == 0 {
		eps = 1e-6
	}
	if n.W != nil {
		return o.ln(x, n.W, n.B, eps)
	}
	return o.ln(x, c.ones, c.zeros, eps)
}

// mhRMS is MultiHeadRMSNorm: x / ‖x‖ · γ · √D per head (x (B, L, H, D),
// γ (H, D)).
func (c *idolCommon) mhRMS(o *iop, x, gamma *tensor.Tensor) *tensor.Tensor {
	n := o.do(func() (*tensor.Tensor, error) { return ops.RMSNorm(x, c.onesD, 1e-30) })
	return o.mul(n, gamma)
}

// ffn is FeedForwardNet: Linear, tanh-GELU, Linear.
type idolFFN struct{ L1, L2 *nn.Linear }

func (f idolFFN) forward(o *iop, x *tensor.Tensor) *tensor.Tensor {
	return o.lin(f.L2, o.un(ops.Gelu, o.lin(f.L1, x)))
}

// chunks splits m (B, n·C) into n (B, 1, C) views.
func idolChunks(o *iop, m *tensor.Tensor, n int) []*tensor.Tensor {
	if o.err != nil {
		return make([]*tensor.Tensor, n)
	}
	B, C := m.Dim(0), m.Dim(1)/n
	out := make([]*tensor.Tensor, n)
	for i := range out {
		out[i] = o.reshape(o.narrow(m, 1, i*C, C), B, 1, C)
	}
	return out
}

// modulate is h·(1 + scale) + shift.
func idolModulate(o *iop, h, shift, scale *tensor.Tensor) *tensor.Tensor {
	return o.add(o.mul(h, o.affine(scale, 1, 1)), shift)
}

// attend runs softmax(q·kᵀ/√D)·v over (B, L, H, D) q and (B, S, H, D) k,
// v, returning (B, L, H·D).
func idolAttend(o *iop, q, k, v *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	B, L, H, D := q.Dim(0), q.Dim(1), q.Dim(2), q.Dim(3)
	a := o.do(func() (*tensor.Tensor, error) {
		return nn.SDPA(o.perm(q, 0, 2, 1, 3), o.perm(k, 0, 2, 1, 3), o.perm(v, 0, 2, 1, 3), nil)
	})
	return o.reshape(o.perm(a, 0, 2, 1, 3), B, L, H*D)
}

// idolSelfAttend is idolAttend for self-attention: on BF16 tensor cores
// (attendBF16) when the projection computes at 16 bits on CUDA (upstream
// runs these blocks in FP16), else the F32 path.
func idolSelfAttend(o *iop, proj *nn.Linear, q, k, v *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	if proj != nil && proj.Mixed16 && device.IsCUDA(q.Device()) {
		out, ok, err := attendBF16(q, k, v)
		if err != nil {
			o.err = err
			return nil
		}
		if ok {
			return out
		}
	}
	return idolAttend(o, q, k, v)
}

// sinusoidal is SinusoidalEmbedder's frequency part: [cos | sin] of
// t · exp(−ln(maxPeriod)·i/half) (· 2π when twoPi), 256 wide, (B, 256).
func idolSinusoidal(ts []float32, maxPeriod float64, twoPi bool, dev device.Device) (*tensor.Tensor, error) {
	const half = 128
	v := make([]float32, len(ts)*2*half)
	for b, t := range ts {
		for i := range half {
			a := float64(float32(t) * float32(math.Exp(-math.Log(maxPeriod)*float64(i)/half)))
			if twoPi {
				a = float64(float32(a) * float32(2*math.Pi))
			}
			v[b*2*half+i] = float32(math.Cos(a))
			v[b*2*half+half+i] = float32(math.Sin(a))
		}
	}
	return tensor.New(v, []int{len(ts), 2 * half}, dev)
}

type idolTimeEmbed struct{ L1, L2 *nn.Linear } // Linear, SiLU, Linear

func (e idolTimeEmbed) forward(o *iop, freq *tensor.Tensor) *tensor.Tensor {
	return o.lin(e.L2, o.un(ops.SiLU, o.lin(e.L1, freq)))
}

// --- denoiser ----------------------------------------------------------------

// idolRePo is RePo3DRotaryEmbedding: a gated MLP of the normalised hidden
// states predicts a 3-D offset per head; the angles are offset × learned
// frequencies × π per axis (10 / 10 / 12 pairs for head dim 64).
type idolRePo struct {
	Norm                 idolNorm
	Gate, Content, Final *nn.Linear
	Freqs                [3]*tensor.Tensor // learned, F32
	freqsPi              [3]*tensor.Tensor // × π, built on first use
}

// angles returns cos and sin (B, L, H, D/2) for hidden states h (B, L, C).
func (r *idolRePo) angles(o *iop, c *idolCommon, h *tensor.Tensor) (cos, sin *tensor.Tensor) {
	if o.err != nil {
		return nil, nil
	}
	B, L := h.Dim(0), h.Dim(1)
	n := c.norm(o, r.Norm, h)
	d := o.lin(r.Final, o.mul(o.un(ops.SiLU, o.lin(r.Gate, n)), o.lin(r.Content, n)))
	d = o.reshape(d, B, L, c.H, 3)
	parts := make([]*tensor.Tensor, 3)
	for k := range parts {
		if r.freqsPi[k] == nil {
			r.freqsPi[k] = o.affine(r.Freqs[k], math.Pi, 0)
		}
		parts[k] = o.mul(o.narrow(d, 3, k, 1), r.freqsPi[k]) // (B, L, H, n_k)
	}
	ang := o.cat(3, parts...)
	return o.un(ops.Cos, ang), o.un(ops.Sin, ang)
}

// idolRotate rotates interleaved pairs of x (B, L, H, D) by (cos, sin)
// (B, L, H, D/2): apply_rotary_emb.
func idolRotate(o *iop, x, cos, sin *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	B, L, H, D := x.Dim(0), x.Dim(1), x.Dim(2), x.Dim(3)
	p := o.reshape(x, B, L, H, D/2, 2)
	x0, x1 := o.narrow(p, 4, 0, 1), o.narrow(p, 4, 1, 1)
	c, s := o.reshape(cos, B, L, H, D/2, 1), o.reshape(sin, B, L, H, D/2, 1)
	y0 := o.sub(o.mul(x0, c), o.mul(x1, s))
	y1 := o.add(o.mul(x0, s), o.mul(x1, c))
	return o.reshape(o.cat(4, y0, y1), B, L, H, D)
}

// idolUnifiedBlock is UnifiedTransformerBlock: RePo self attention + FFN,
// adaLN-modulated by the time (plus its own shift table) unless it is a
// condition-refiner block (affine LayerNorms, no modulation).
type idolUnifiedBlock struct {
	Norm1, Norm2 idolNorm
	QKV, Out     *nn.Linear
	QNorm, KNorm *tensor.Tensor // (H, D)
	FFN          idolFFN
	ShiftTable   *tensor.Tensor // (1, 6C); nil without modulation
	RePo         *idolRePo
}

func (b *idolUnifiedBlock) attn(o *iop, c *idolCommon, h, cos, sin *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	B, L := h.Dim(0), h.Dim(1)
	D := c.C / c.H
	qkv := o.reshape(o.lin(b.QKV, h), B, L, 3, c.H, D)
	q := o.reshape(o.narrow(qkv, 2, 0, 1), B, L, c.H, D)
	k := o.reshape(o.narrow(qkv, 2, 1, 1), B, L, c.H, D)
	v := o.reshape(o.narrow(qkv, 2, 2, 1), B, L, c.H, D)
	q = c.mhRMS(o, idolRotate(o, q, cos, sin), b.QNorm)
	k = c.mhRMS(o, idolRotate(o, k, cos, sin), b.KNorm)
	return o.lin(b.Out, idolSelfAttend(o, b.QKV, q, k, v))
}

func (b *idolUnifiedBlock) forward(o *iop, c *idolCommon, x, tmod *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	cos, sin := b.RePo.angles(o, c, x)
	if b.ShiftTable == nil {
		x = o.add(x, b.attn(o, c, c.norm(o, b.Norm1, x), cos, sin))
		return o.add(x, b.FFN.forward(o, c.norm(o, b.Norm2, x)))
	}
	m := idolChunks(o, o.add(tmod, b.ShiftTable), 6) // shift, scale, gate (msa), shift, scale, gate (mlp)
	h := idolModulate(o, c.norm(o, b.Norm1, x), m[0], m[1])
	x = o.add(x, o.mul(b.attn(o, c, h, cos, sin), m[2]))
	h = idolModulate(o, c.norm(o, b.Norm2, x), m[3], m[4])
	return o.add(x, o.mul(b.FFN.forward(o, h), m[5]))
}

// IdolDenoiser is LatentSeqMMFlowModel.
type IdolDenoiser struct {
	Args                             IdolDenoiserArgs
	TEmbed                           idolTimeEmbed
	AdaLN                            *nn.Linear // after SiLU, 6C
	Input, Cond, Cond2               *nn.Linear
	NoiseRefiner, CtxRefiner, Blocks []*idolUnifiedBlock
	CamRefiner                       []*nn.Linear   // Linear (GELU Linear)...
	ShiftTable                       *tensor.Tensor // (1, 2, C)
	Out, CamOut                      *nn.Linear
	posEmb                           *tensor.Tensor // (1, N, C), from pos_pe
	c                                *idolCommon
}

// IdolInputs are one denoiser evaluation's inputs.
type IdolInputs struct {
	Latent   *tensor.Tensor // (B, N, in)
	Camera   *tensor.Tensor // (B, 1, cam)
	T        []float32      // per batch element, in [0, 1000]
	Feature1 *tensor.Tensor // (B, Lc, cond) DINOv3 tokens
	Feature2 *tensor.Tensor // (B, Lc, cond2) FLUX.2 VAE tokens
}

// Forward returns the velocity of the latent and the camera.
func (m *IdolDenoiser) Forward(in IdolInputs) (vLatent, vCamera *tensor.Tensor, err error) {
	o := &iop{}
	c := m.c
	N := in.Latent.Dim(1)
	hx := o.add(o.lin(m.Input, in.Latent), m.posEmb)
	hc := o.add(o.lin(m.Cond, in.Feature1), o.lin(m.Cond2, in.Feature2))
	freq := o.do(func() (*tensor.Tensor, error) { return idolSinusoidal(in.T, 10000, false, c.dev) })
	temb := m.TEmbed.forward(o, freq)
	tmod := o.lin(m.AdaLN, o.un(ops.SiLU, temb))
	for _, b := range m.NoiseRefiner {
		hx = b.forward(o, c, hx, tmod)
	}
	for _, b := range m.CtxRefiner {
		hc = b.forward(o, c, hc, nil)
	}
	hcam := in.Camera
	for i, l := range m.CamRefiner {
		if i > 0 {
			hcam = o.un(ops.Gelu, hcam)
		}
		hcam = o.lin(l, hcam)
	}
	h := o.cat(1, hx, hc, hcam)
	for _, b := range m.Blocks {
		h = b.forward(o, c, h, tmod)
	}
	if o.err != nil {
		return nil, nil, o.err
	}
	B, L := h.Dim(0), h.Dim(1)
	st := o.add(m.ShiftTable, o.reshape(temb, B, 1, c.C)) // (B, 2, C)
	shift, scale := o.narrow(st, 1, 0, 1), o.narrow(st, 1, 1, 1)
	head := func(x *tensor.Tensor, out *nn.Linear) *tensor.Tensor {
		return o.lin(out, idolModulate(o, c.norm(o, idolFinalNorm, x), shift, scale))
	}
	vLatent = head(o.narrow(h, 1, 0, N), m.Out)
	vCamera = head(o.narrow(h, 1, L-1, 1), m.CamOut)
	return vLatent, vCamera, o.err
}

// idolPosEmbed is PcdAbsolutePositionEmbedder over (N, 3) positions:
// per coordinate [sin | cos] of 2π·x·f with f = 2^[0..15, then 16 steps
// spread up to 16], padded with zeros to C. Host-side (pos_pe is fixed).
func idolPosEmbed(pos []float32, n, C int) []float32 {
	fd := C / 3 / 2
	const maxRes = 16
	freqs := make([]float32, 0, fd)
	for i := 0; i < maxRes && len(freqs) < fd; i++ {
		freqs = append(freqs, float32(math.Pow(2, float64(i))))
	}
	resDim := max(0, fd-maxRes)
	for i := 0; len(freqs) < fd; i++ {
		e := float32(i) / float32(resDim) * maxRes
		freqs = append(freqs, float32(math.Pow(2, float64(e))))
	}
	out := make([]float32, n*C)
	for t := range n {
		for k := range 3 {
			x := pos[t*3+k]
			base := t*C + k*2*fd
			for i, f := range freqs {
				a := float64(x * f * float32(2*math.Pi))
				out[base+i] = float32(math.Sin(a))
				out[base+fd+i] = float32(math.Cos(a))
			}
		}
	}
	return out
}

// --- decoders ----------------------------------------------------------------

// idolMHA is MultiHeadAttention: self (QKV) or cross (Q, KV) attention with
// per-head RMS-normalised queries and keys (when QNorm is set).
type idolMHA struct {
	QKV, Q, KV, Out *nn.Linear
	QNorm, KNorm    *tensor.Tensor
}

func (a *idolMHA) forward(o *iop, c *idolCommon, x, ctx *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	B, L := x.Dim(0), x.Dim(1)
	D := c.C / c.H
	var q, k, v *tensor.Tensor
	if a.QKV != nil {
		qkv := o.reshape(o.lin(a.QKV, x), B, L, 3, c.H, D)
		q = o.reshape(o.narrow(qkv, 2, 0, 1), B, L, c.H, D)
		k = o.reshape(o.narrow(qkv, 2, 1, 1), B, L, c.H, D)
		v = o.reshape(o.narrow(qkv, 2, 2, 1), B, L, c.H, D)
	} else {
		S := ctx.Dim(1)
		q = o.reshape(o.lin(a.Q, x), B, L, c.H, D)
		kv := o.reshape(o.lin(a.KV, ctx), B, S, 2, c.H, D)
		k = o.reshape(o.narrow(kv, 2, 0, 1), B, S, c.H, D)
		v = o.reshape(o.narrow(kv, 2, 1, 1), B, S, c.H, D)
	}
	if a.QNorm != nil { // TRELLIS's cross attention has no q / k norm
		q, k = c.mhRMS(o, q, a.QNorm), c.mhRMS(o, k, a.KNorm)
	}
	if a.QKV != nil {
		return o.lin(a.Out, idolSelfAttend(o, a.QKV, q, k, v))
	}
	return o.lin(a.Out, idolAttend(o, q, k, v))
}

// idolCrossBlock is TransformerCrossBlock (SelfAttn set: self + cross +
// FFN, norm2 affine) or ModulatedTransformerCrossOnlyBlock (cross + FFN,
// adaLN-modulated by the octree level).
type idolCrossBlock struct {
	SelfAttn, CrossAttn *idolMHA
	Norm2               idolNorm
	FFN                 idolFFN
}

func (b *idolCrossBlock) forward(o *iop, c *idolCommon, x, ctx, mod *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	if mod != nil {
		m := idolChunks(o, mod, 6)
		h := idolModulate(o, c.norm(o, idolNorm{}, x), m[0], m[1])
		x = o.add(x, o.mul(b.CrossAttn.forward(o, c, h, ctx), m[2]))
		h = idolModulate(o, c.norm(o, idolNorm{}, x), m[3], m[4])
		return o.add(x, o.mul(b.FFN.forward(o, h), m[5]))
	}
	x = o.add(x, b.SelfAttn.forward(o, c, c.norm(o, idolNorm{}, x), nil))
	x = o.add(x, b.CrossAttn.forward(o, c, c.norm(o, b.Norm2, x), ctx))
	return o.add(x, b.FFN.forward(o, c.norm(o, idolNorm{}, x)))
}

// idolPointEmbed is in_proj(x) + PcdAbsolutePositionEmbedderv2(x), then
// input_layer: per coordinate [sin | cos] of π·x·2^linspace(0, 10).
type idolPointEmbed struct {
	InProj, Input *nn.Linear
	freqs         *tensor.Tensor // (F,) π·2^linspace(0, 10, F)
	pad           int
}

func (e *idolPointEmbed) forward(o *iop, c *idolCommon, pts *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	B, L := pts.Dim(0), pts.Dim(1)
	if e.freqs == nil {
		fd := c.C / 3 / 2
		f := make([]float32, fd)
		for i := range f {
			l := float32(0)
			if fd > 1 {
				l = float32(10 * float64(i) / float64(fd-1))
			}
			f[i] = float32(math.Pow(2, float64(l))) * float32(math.Pi)
		}
		e.freqs = o.do(func() (*tensor.Tensor, error) { return tensor.New(f, []int{fd}, c.dev) })
		e.pad = c.C - 6*fd
	}
	ang := o.mul(o.reshape(pts, B*L, 3, 1), e.freqs) // (N, 3, F)
	emb := o.reshape(o.cat(2, o.un(ops.Sin, ang), o.un(ops.Cos, ang)), B, L, 3*2*e.freqs.Dim(0))
	if e.pad > 0 {
		z := o.do(func() (*tensor.Tensor, error) { return tensor.Zeros(dtype.F32, []int{B, L, e.pad}, c.dev) })
		emb = o.cat(2, emb, z)
	}
	return o.lin(e.Input, o.add(o.lin(e.InProj, pts), emb))
}

// IdolOctreeDecoder is OctreeProbabilityFixedlenDecoder.
type IdolOctreeDecoder struct {
	Args   IdolDecoderArgs
	Embed  idolPointEmbed
	LEmbed idolTimeEmbed
	AdaLN  *nn.Linear
	Blocks []*idolCrossBlock
	Out    *nn.Linear
	c      *idolCommon
}

// Forward returns the logits (B, L, 8) of how the points of the cells
// centred at x (B, L, 3) split among their children at octree resolution
// `level`, given the latent cond (B, Lc, 16).
func (m *IdolOctreeDecoder) Forward(x *tensor.Tensor, level int, cond *tensor.Tensor) (*tensor.Tensor, error) {
	o := &iop{}
	c := m.c
	h := m.Embed.forward(o, c, x)
	ts := make([]float32, x.Dim(0))
	for i := range ts {
		ts[i] = float32(level)
	}
	freq := o.do(func() (*tensor.Tensor, error) { return idolSinusoidal(ts, 1024, true, c.dev) })
	mod := o.lin(m.AdaLN, o.un(ops.SiLU, m.LEmbed.forward(o, freq)))
	for _, b := range m.Blocks {
		h = b.forward(o, c, h, cond, mod)
	}
	out := o.lin(m.Out, c.norm(o, idolFinalNorm, h))
	return out, o.err
}

// IdolGaussianDecoder is GaussianFixedlenDecoder.
type IdolGaussianDecoder struct {
	Args         IdolDecoderArgs
	Embed        idolPointEmbed
	Blocks       []*idolCrossBlock
	Out          *nn.Linear
	Perturbation []float32 // (G, 3) points_offset_perturbation
	BaseOffset   float32   // base_offset_scale
	c            *idolCommon
}

// Forward returns the per-anchor outputs (B, L, G·15) for anchors (B, L, 3).
func (m *IdolGaussianDecoder) Forward(pts, cond *tensor.Tensor) (*tensor.Tensor, error) {
	o := &iop{}
	c := m.c
	h := m.Embed.forward(o, c, pts)
	for _, b := range m.Blocks {
		h = b.forward(o, c, h, cond, nil)
	}
	out := o.lin(m.Out, c.norm(o, idolFinalNorm, h))
	return out, o.err
}

// --- loading -----------------------------------------------------------------

// Idol is the three networks.
type Idol struct {
	Cfg      *IdolConfig
	Denoiser *IdolDenoiser
	Octree   *IdolOctreeDecoder
	Gaussian *IdolGaussianDecoder
}

// idolLoader builds layers from one safetensors file's tensors.
type idolLoader struct {
	hf  map[string]*tensor.Tensor
	dev device.Device
	err error
	// fp8Block, when set, keeps F8E4M3 weights that have a
	// <name>.weight_scale_inv as block-scaled FP8 (nn.Linear.WBlockScale).
	fp8Block [2]int
}

func newIdolLoader(path string) (*idolLoader, error) {
	hf, _, err := safetensors.LoadFiles([]string{path}, device.CPU())
	if err != nil {
		return nil, err
	}
	return &idolLoader{hf: hf}, nil
}

func (l *idolLoader) linear(name string) *nn.Linear {
	if l.err != nil {
		return nil
	}
	w, ok := l.hf[name+".weight"]
	if !ok {
		l.err = fmt.Errorf("idol: missing %s.weight", name)
		return nil
	}
	_, hasB := l.hf[name+".bias"]
	if sc, ok := l.hf[name+".weight_scale_inv"]; ok && l.fp8Block[0] > 0 && w.DType() == dtype.F8E4M3 {
		bs := l.fp8Block
		ragged := w.Dim(0)%bs[0] != 0 || w.Dim(1)%bs[1] != 0
		if ragged || hasB || !(device.IsCPU(l.dev) || device.IsCUDA(l.dev)) {
			// Block-scaled FP8 runs on CPU and CUDA for whole blocks;
			// otherwise widen on the host now (W = fp8 · scale_inv per block).
			d, err := dequantFP8BlockHost(w, sc, bs)
			if err != nil {
				l.err = err
				return nil
			}
			l.hf[name+".weight"] = d
			delete(l.hf, name+".weight_scale_inv")
			return l.linear(name)
		}
	}
	if sc, ok := l.hf[name+".weight_scale_inv"]; ok && l.fp8Block[0] > 0 && w.DType() == dtype.F8E4M3 && !hasB {
		lin, err := nn.NewLinearPlaceholder(w.Dim(1), w.Dim(0), false, l.dev)
		if err != nil {
			l.err = err
			return nil
		}
		wd, err := w.To(l.dev)
		if err != nil {
			l.err = err
			return nil
		}
		if sc, err = ops.Cast(sc, dtype.F32); err != nil {
			l.err = err
			return nil
		}
		if sc, err = sc.To(l.dev); err != nil {
			l.err = err
			return nil
		}
		if l.err = lin.SetWeight(wd); l.err != nil {
			return nil
		}
		lin.WBlockScale, lin.WBlock = sc, l.fp8Block
		delete(l.hf, name+".weight")
		delete(l.hf, name+".weight_scale_inv")
		return lin
	}
	lin, err := nn.NewLinearUninit(w.Dim(1), w.Dim(0), hasB, dtype.F32, l.dev)
	if err != nil {
		l.err = err
		return nil
	}
	if l.err = copyExactCast(lin.W, l.hf, name+".weight"); l.err == nil && hasB {
		l.err = copyExactCast(lin.B, l.hf, name+".bias")
	}
	return lin
}

// dequantFP8BlockHost widens an F8E4M3 weight (N, K) with its per-block
// scale_inv (⌈N/b0⌉, ⌈K/b1⌉) to F32 on the host, edge blocks included.
func dequantFP8BlockHost(w, scale *tensor.Tensor, bs [2]int) (*tensor.Tensor, error) {
	wf, err := ops.Cast(w, dtype.F32)
	if err != nil {
		return nil, err
	}
	sf, err := ops.Cast(scale, dtype.F32)
	if err != nil {
		return nil, err
	}
	N, K := w.Dim(0), w.Dim(1)
	nb, kb := (N+bs[0]-1)/bs[0], (K+bs[1]-1)/bs[1]
	if sf.NumElements() != nb*kb {
		return nil, fmt.Errorf("fp8 weight %v: scale has %d elements, want %d x %d", w.Shape(), sf.NumElements(), nb, kb)
	}
	wv, sv := tensor.MustSlice[float32](wf), tensor.MustSlice[float32](sf)
	out := make([]float32, N*K)
	for i := range N {
		for j := range K {
			out[i*K+j] = wv[i*K+j] * sv[(i/bs[0])*kb+j/bs[1]]
		}
	}
	return tensor.New(out, []int{N, K}, device.CPU())
}

// param uploads a tensor at F32.
func (l *idolLoader) param(name string) *tensor.Tensor {
	if l.err != nil {
		return nil
	}
	t, ok := l.hf[name]
	if !ok {
		l.err = fmt.Errorf("idol: missing %s", name)
		return nil
	}
	if t.DType() != dtype.F32 {
		if t, l.err = ops.Cast(t, dtype.F32); l.err != nil {
			return nil
		}
	}
	var out *tensor.Tensor
	out, l.err = t.To(l.dev)
	return out
}

// host returns a tensor's values at F32 on the host.
func (l *idolLoader) host(name string) []float32 {
	if l.err != nil {
		return nil
	}
	t, ok := l.hf[name]
	if !ok {
		l.err = fmt.Errorf("idol: missing %s", name)
		return nil
	}
	if t.DType() != dtype.F32 {
		if t, l.err = ops.Cast(t, dtype.F32); l.err != nil {
			return nil
		}
	}
	return append([]float32(nil), tensor.MustSlice[float32](t)...)
}

func (l *idolLoader) norm(name string) idolNorm {
	if _, ok := l.hf[name+".weight"]; !ok {
		return idolNorm{}
	}
	return idolNorm{W: l.param(name + ".weight"), B: l.param(name + ".bias")}
}

func (l *idolLoader) ffn(name string) idolFFN {
	return idolFFN{L1: l.linear(name + ".mlp.0"), L2: l.linear(name + ".mlp.2")}
}

func (l *idolLoader) repo(name string) *idolRePo {
	n := l.norm(name + ".norm")
	n.Eps = 1e-5 // LayerNorm32(model_channels): PyTorch's default eps
	return &idolRePo{
		Norm: n, Gate: l.linear(name + ".gate_map"), Content: l.linear(name + ".content_map"),
		Final: l.linear(name + ".final_map"),
		Freqs: [3]*tensor.Tensor{l.param(name + ".freqs_0"), l.param(name + ".freqs_1"), l.param(name + ".freqs_2")},
	}
}

func (l *idolLoader) unified(name, repo string, mod bool) *idolUnifiedBlock {
	b := &idolUnifiedBlock{
		Norm1: l.norm(name + ".norm1"), Norm2: l.norm(name + ".norm2"),
		QKV: l.linear(name + ".attn.qkv"), Out: l.linear(name + ".attn.out"),
		QNorm: l.param(name + ".attn.q_norm.gamma"), KNorm: l.param(name + ".attn.k_norm.gamma"),
		FFN: l.ffn(name + ".mlp"), RePo: l.repo(repo),
	}
	if mod {
		b.ShiftTable = l.param(name + ".shift_table")
	}
	return b
}

func (l *idolLoader) mha(name string, cross bool) *idolMHA {
	a := &idolMHA{Out: l.linear(name + ".to_out"), QNorm: l.param(name + ".q_rms_norm.gamma"),
		KNorm: l.param(name + ".k_rms_norm.gamma")}
	if cross {
		a.Q, a.KV = l.linear(name+".to_q"), l.linear(name+".to_kv")
	} else {
		a.QKV = l.linear(name + ".to_qkv")
	}
	return a
}

// LoadIdol loads speridlabs/idol's three networks from dir (config.json,
// denoiser / decoder / decoder_gs .safetensors) onto dev. Under
// nn.WithWeightStorage16 the linears keep 16-bit weights.
func LoadIdol(dir string, dev device.Device) (*Idol, error) {
	cfg, err := LoadIdolConfig(dir)
	if err != nil {
		return nil, err
	}
	m := &Idol{Cfg: cfg}
	if m.Denoiser, err = loadIdolDenoiser(filepath.Join(dir, "denoiser.safetensors"), cfg.Denoiser.Args, dev); err != nil {
		return nil, err
	}
	if m.Octree, err = loadIdolOctree(filepath.Join(dir, "decoder.safetensors"), cfg.Decoder.Args, dev); err != nil {
		return nil, err
	}
	if m.Gaussian, err = loadIdolGaussian(filepath.Join(dir, "decoder_gs.safetensors"), cfg.DecoderGS.Args, dev); err != nil {
		return nil, err
	}
	return m, nil
}

func loadIdolDenoiser(path string, a IdolDenoiserArgs, dev device.Device) (*IdolDenoiser, error) {
	l, err := newIdolLoader(path)
	if err != nil {
		return nil, err
	}
	l.dev = dev
	if a.ModelChannels/a.NumHeads != idolHeadDim {
		return nil, fmt.Errorf("idol denoiser: head dim %d, RePo needs %d", a.ModelChannels/a.NumHeads, idolHeadDim)
	}
	c, err := newIdolCommon(a.ModelChannels, a.NumHeads, dev)
	if err != nil {
		return nil, err
	}
	m := &IdolDenoiser{Args: a, c: c,
		TEmbed: idolTimeEmbed{L1: l.linear("t_embedder.mlp.0"), L2: l.linear("t_embedder.mlp.2")},
		AdaLN:  l.linear("adaLN_modulation.1"), Input: l.linear("input_layer"),
		Cond: l.linear("cond_embedder"), Cond2: l.linear("cond_embedder2"),
		ShiftTable: l.param("shift_table"), Out: l.linear("out_layer"), CamOut: l.linear("cam_out_layer"),
	}
	for i := range a.NumRefinerBlocks {
		m.NoiseRefiner = append(m.NoiseRefiner, l.unified(fmt.Sprintf("noise_refiner.%d", i), fmt.Sprintf("noise_repo_layers.%d", i), true))
		m.CtxRefiner = append(m.CtxRefiner, l.unified(fmt.Sprintf("context_refiner.%d", i), fmt.Sprintf("context_repo_layers.%d", i), false))
		m.CamRefiner = append(m.CamRefiner, l.linear(fmt.Sprintf("cam_refiner.mlp.%d", 2*i)))
	}
	for i := range a.NumBlocks {
		m.Blocks = append(m.Blocks, l.unified(fmt.Sprintf("blocks.%d", i), fmt.Sprintf("repo_layers.%d", i), true))
	}
	pos := l.host("pos_pe")
	if l.err != nil {
		return nil, fmt.Errorf("idol denoiser: %w", l.err)
	}
	N := len(pos) / 3
	if m.posEmb, err = tensor.New(idolPosEmbed(pos, N, a.ModelChannels), []int{1, N, a.ModelChannels}, dev); err != nil {
		return nil, err
	}
	return m, nil
}

func loadIdolOctree(path string, a IdolDecoderArgs, dev device.Device) (*IdolOctreeDecoder, error) {
	l, err := newIdolLoader(path)
	if err != nil {
		return nil, err
	}
	l.dev = dev
	c, err := newIdolCommon(a.ModelChannels, a.NumHeads, dev)
	if err != nil {
		return nil, err
	}
	m := &IdolOctreeDecoder{Args: a, c: c,
		Embed:  idolPointEmbed{InProj: l.linear("in_proj"), Input: l.linear("input_layer")},
		LEmbed: idolTimeEmbed{L1: l.linear("l_embedder.mlp.0"), L2: l.linear("l_embedder.mlp.2")},
		AdaLN:  l.linear("adaLN_modulation.1"), Out: l.linear("out_proj"),
	}
	for i := range a.NumBlocks {
		p := fmt.Sprintf("blocks.%d", i)
		m.Blocks = append(m.Blocks, &idolCrossBlock{CrossAttn: l.mha(p+".cross_attn", true), FFN: l.ffn(p + ".mlp")})
	}
	if l.err != nil {
		return nil, fmt.Errorf("idol decoder: %w", l.err)
	}
	return m, nil
}

func loadIdolGaussian(path string, a IdolDecoderArgs, dev device.Device) (*IdolGaussianDecoder, error) {
	l, err := newIdolLoader(path)
	if err != nil {
		return nil, err
	}
	l.dev = dev
	c, err := newIdolCommon(a.ModelChannels, a.NumHeads, dev)
	if err != nil {
		return nil, err
	}
	m := &IdolGaussianDecoder{Args: a, c: c,
		Embed: idolPointEmbed{InProj: l.linear("in_proj"), Input: l.linear("input_layer")},
		Out:   l.linear("out_proj"), Perturbation: l.host("points_offset_perturbation"),
	}
	if b := l.host("base_offset_scale"); len(b) == 1 {
		m.BaseOffset = b[0]
	}
	for i := range a.NumBlocks {
		p := fmt.Sprintf("blocks.%d", i)
		m.Blocks = append(m.Blocks, &idolCrossBlock{SelfAttn: l.mha(p+".self_attn", false),
			CrossAttn: l.mha(p+".cross_attn", true), Norm2: l.norm(p + ".norm2"), FFN: l.ffn(p + ".mlp")})
	}
	if l.err != nil {
		return nil, fmt.Errorf("idol decoder_gs: %w", l.err)
	}
	return m, nil
}

// IdolImageTokens is idol's DINOv3 conditioning on last_hidden_state: a
// LayerNorm without affine (PyTorch's default eps).
func IdolImageTokens(hidden *tensor.Tensor) (*tensor.Tensor, error) {
	D := hidden.Dim(-1)
	ones, err := tensor.Ones(dtype.F32, []int{D}, hidden.Device())
	if err != nil {
		return nil, err
	}
	zeros, err := tensor.Zeros(dtype.F32, []int{D}, hidden.Device())
	if err != nil {
		return nil, err
	}
	return ops.LayerNorm(hidden, ones, zeros, 1e-5)
}
