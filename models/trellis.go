package models

import (
	"encoding/json"
	"fmt"
	"math"
	"os"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// TRELLIS (microsoft/TRELLIS-image-large) stage 1: the dense half of
// TrellisImageTo3DPipeline, from DINOv2 image tokens to the sparse
// structure (the occupied voxels of a 64³ grid).
//
//	TrellisFlow       SparseStructureFlowModel: a DiT over the 16³ x 8
//	                  occupancy latent (adaLN-modulated self attention with
//	                  per-head RMS-normalised q / k, cross attention to the
//	                  image tokens, absolute sin / cos positions)
//	TrellisSample     FlowEulerGuidanceIntervalSampler
//	TrellisSSDecoder  SparseStructureDecoder: 3-D conv res-blocks and
//	                  pixel-shuffle upsampling to 64³ occupancy logits
//
// Each checkpoint is ckpts/<name>.json ({"name", "args"}) next to
// ckpts/<name>.safetensors; layer names follow TRELLIS's modules
// (trellis/models/sparse_structure_{flow,vae}.py).

// TrellisFlowArgs are SparseStructureFlowModel's constructor arguments.
type TrellisFlowArgs struct {
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
	PEMode          string  `json:"pe_mode"`
	ShareMod        bool    `json:"share_mod"`
	QKRMSNorm       bool    `json:"qk_rms_norm"`
	QKRMSNormCross  bool    `json:"qk_rms_norm_cross"`
	UseFP16         bool    `json:"use_fp16"`
}

// TrellisDecoderArgs are SparseStructureDecoder's constructor arguments.
type TrellisDecoderArgs struct {
	OutChannels        int    `json:"out_channels"`
	LatentChannels     int    `json:"latent_channels"`
	NumResBlocks       int    `json:"num_res_blocks"`
	NumResBlocksMiddle int    `json:"num_res_blocks_middle"`
	Channels           []int  `json:"channels"`
	NormType           string `json:"norm_type"`
	UseFP16            bool   `json:"use_fp16"`
}

// trellisConfig reads a TRELLIS checkpoint config ({"name", "args"}) into
// args (pre-filled with the constructor defaults), checking the name.
func trellisConfig(path, want string, args any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var c struct {
		Name string          `json:"name"`
		Args json.RawMessage `json:"args"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	if c.Name != want {
		return fmt.Errorf("%s: model %q, want %s", path, c.Name, want)
	}
	if err := json.Unmarshal(c.Args, args); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}

// --- flow model --------------------------------------------------------------

// trellisBlock is ModulatedTransformerCrossBlock: adaLN-modulated self
// attention, cross attention to the image tokens (affine norm2) and an FFN.
type trellisBlock struct {
	AdaLN       *nn.Linear // after SiLU, 6C; nil with share_mod
	Self, Cross *idolMHA
	Norm2       idolNorm
	FFN         idolFFN
}

func (b *trellisBlock) forward(o *iop, c *idolCommon, x, temb, ctx *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	mod := temb
	if b.AdaLN != nil {
		mod = o.lin(b.AdaLN, o.un(ops.SiLU, temb))
	}
	m := idolChunks(o, mod, 6) // shift, scale, gate (msa), shift, scale, gate (mlp)
	h := idolModulate(o, c.norm(o, idolNorm{}, x), m[0], m[1])
	x = o.add(x, o.mul(b.Self.forward(o, c, h, nil), m[2]))
	x = o.add(x, b.Cross.forward(o, c, c.norm(o, b.Norm2, x), ctx))
	h = idolModulate(o, c.norm(o, idolNorm{}, x), m[3], m[4])
	return o.add(x, o.mul(b.FFN.forward(o, h), m[5]))
}

// TrellisFlow is SparseStructureFlowModel.
type TrellisFlow struct {
	Args   TrellisFlowArgs
	TEmbed idolTimeEmbed
	AdaLN  *nn.Linear // share_mod only
	Input  *nn.Linear
	Blocks []*trellisBlock
	Out    *nn.Linear
	PosEmb *tensor.Tensor // (1, N, C): the pos_emb buffer
	c      *idolCommon
}

// LoadTrellisFlow loads ckpts/<name> (base path without extension: .json
// and .safetensors) onto dev. Under nn.WithWeightStorage16 the linears keep
// 16-bit weights.
func LoadTrellisFlow(base string, dev device.Device) (*TrellisFlow, error) {
	a := TrellisFlowArgs{NumHeadChannels: 64, MLPRatio: 4, PatchSize: 2, PEMode: "ape"}
	if err := trellisConfig(base+".json", "SparseStructureFlowModel", &a); err != nil {
		return nil, err
	}
	if a.NumHeads == 0 {
		a.NumHeads = a.ModelChannels / a.NumHeadChannels
	}
	if a.PatchSize != 1 {
		return nil, fmt.Errorf("trellis flow: patch size %d (only 1, as the released models, is supported)", a.PatchSize)
	}
	if a.PEMode != "ape" {
		return nil, fmt.Errorf("trellis flow: pe_mode %q (only ape is supported)", a.PEMode)
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
	m := &TrellisFlow{Args: a, c: c,
		TEmbed: idolTimeEmbed{L1: l.linear("t_embedder.mlp.0"), L2: l.linear("t_embedder.mlp.2")},
		Input:  l.linear("input_layer"), Out: l.linear("out_layer"),
	}
	if a.ShareMod {
		m.AdaLN = l.linear("adaLN_modulation.1")
	}
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
	for i := range a.NumBlocks {
		p := fmt.Sprintf("blocks.%d", i)
		b := &trellisBlock{
			Self: mha(p+".self_attn", false, a.QKRMSNorm), Cross: mha(p+".cross_attn", true, a.QKRMSNormCross),
			Norm2: l.norm(p + ".norm2"), FFN: l.ffn(p + ".mlp"),
		}
		if !a.ShareMod {
			b.AdaLN = l.linear(p + ".adaLN_modulation.1")
		}
		m.Blocks = append(m.Blocks, b)
	}
	if _, ok := l.hf["pos_emb"]; ok {
		m.PosEmb = l.param("pos_emb")
	} else if l.err == nil {
		n := a.Resolution / a.PatchSize
		m.PosEmb, err = tensor.New(TrellisPosEmbed(n, a.ModelChannels), []int{n * n * n, a.ModelChannels}, dev)
		if err != nil {
			return nil, err
		}
	}
	if l.err != nil {
		return nil, fmt.Errorf("trellis flow: %w", l.err)
	}
	if m.PosEmb, err = m.PosEmb.Reshape(1, m.PosEmb.Dim(0), m.PosEmb.Dim(1)); err != nil {
		return nil, err
	}
	return m, nil
}

// TrellisPosEmbed is AbsolutePositionEmbedder(C, 3) over the n³ grid
// coordinates (meshgrid "ij"): per axis [sin | cos] of x·10000^(−i/F), F
// = C / 3 / 2, zero-padded to C. Row-major (n³, C).
func TrellisPosEmbed(n, C int) []float32 {
	fd := C / 3 / 2
	freqs := make([]float32, fd)
	for i := range freqs {
		freqs[i] = float32(1 / math.Pow(10000, float64(float32(i)/float32(fd))))
	}
	out := make([]float32, n*n*n*C)
	for t := range n * n * n {
		xyz := [3]int{t / (n * n), t / n % n, t % n}
		for k, x := range xyz {
			base := t*C + k*2*fd
			for i, f := range freqs {
				a := float64(float32(x) * f)
				out[base+i] = float32(math.Sin(a))
				out[base+fd+i] = float32(math.Cos(a))
			}
		}
	}
	return out
}

// Forward returns the velocity (B, out, R, R, R) for the latent x (B, in,
// R, R, R) at times t (per batch element, in [0, 1000]) given the image
// tokens cond (B, L, cond_channels).
func (m *TrellisFlow) Forward(x *tensor.Tensor, t []float32, cond *tensor.Tensor) (*tensor.Tensor, error) {
	a := m.Args
	R := a.Resolution
	if x.Rank() != 5 || x.Dim(1) != a.InChannels || x.Dim(2) != R || x.Dim(3) != R || x.Dim(4) != R {
		return nil, fmt.Errorf("trellis flow: input %v, want (B, %d, %d, %d, %d)", x.Shape(), a.InChannels, R, R, R)
	}
	o := &iop{}
	c := m.c
	B, N := x.Dim(0), R*R*R
	h := o.perm(o.reshape(x, B, a.InChannels, N), 0, 2, 1) // patch size 1: tokens are the voxels
	h = o.add(o.lin(m.Input, h), m.PosEmb)
	freq := o.do(func() (*tensor.Tensor, error) { return idolSinusoidal(t, 10000, false, c.dev) })
	temb := m.TEmbed.forward(o, freq)
	if m.AdaLN != nil {
		temb = o.lin(m.AdaLN, o.un(ops.SiLU, temb))
	}
	for _, b := range m.Blocks {
		h = b.forward(o, c, h, temb, cond)
	}
	h = o.lin(m.Out, c.norm(o, idolFinalNorm, h))
	out := o.reshape(o.perm(h, 0, 2, 1), B, a.OutChannels, R, R, R)
	return out, o.err
}

// TrellisSampleOptions are FlowEulerGuidanceIntervalSampler's parameters
// (pipeline.json's sparse_structure_sampler.params).
type TrellisSampleOptions struct {
	Steps    int        // default 25
	CFG      float64    // cfg_strength, default 5 (negative disables)
	Interval [2]float64 // cfg_interval, default [0.5, 1]
	RescaleT float64    // rescale_t, default 3
	// Step, when set, is called after each step (progress).
	Step func(i, n int)
}

func (o *TrellisSampleOptions) defaults() {
	if o.Steps == 0 {
		o.Steps = 25
	}
	if o.CFG == 0 {
		o.CFG = 5
	} else if o.CFG < 0 {
		o.CFG = 0
	}
	if o.Interval == [2]float64{} {
		o.Interval = [2]float64{0.5, 1}
	}
	if o.RescaleT == 0 {
		o.RescaleT = 3
	}
}

// TrellisSample is FlowEulerGuidanceIntervalSampler.sample: Euler steps x
// ← x − (t − t')·v from noise (t = 1) to t = 0 over linspace(1, 0) rescaled
// by t ↦ s·t / (1 + (s − 1)·t), with v = (1 + w)·v(cond) − w·v(negCond)
// while t is inside the guidance interval and v(cond) outside. negCond nil
// means zeros (TRELLIS's get_cond: torch.zeros_like(cond)). sigma_min
// only enters the sampler's x_0 predictions, not the samples.
func TrellisSample(m *TrellisFlow, noise, cond, negCond *tensor.Tensor, o TrellisSampleOptions) (*tensor.Tensor, error) {
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
	B := noise.Dim(0)
	x := noise
	for i := range o.Steps {
		t, tPrev := grid[i], grid[i+1]
		ts := make([]float32, B)
		for b := range ts {
			ts[b] = float32(1000 * t)
		}
		v, err := m.Forward(x, ts, cond)
		if err != nil {
			return nil, err
		}
		if o.CFG > 0 && o.Interval[0] <= t && t <= o.Interval[1] {
			u, err := m.Forward(x, ts, negCond)
			if err != nil {
				return nil, err
			}
			if v, err = idolGuide(v, u, o.CFG); err != nil {
				return nil, err
			}
		}
		if x, err = idolEuler(x, v, t-tPrev); err != nil {
			return nil, err
		}
		if o.Step != nil {
			o.Step(i, o.Steps)
		}
	}
	return x, nil
}

// TrellisImageTokens is TRELLIS's encode_image on top of DINOv2: the
// x_prenorm tokens (CLS, registers and patches; DINOv2.Hidden) through a
// LayerNorm without affine (PyTorch's default eps).
func TrellisImageTokens(hidden *tensor.Tensor) (*tensor.Tensor, error) {
	return IdolImageTokens(hidden)
}

// --- decoder -----------------------------------------------------------------

// The decoder runs channel-last, (B, X, Y, Z, C): ChannelLayerNorm32 is
// then a plain LayerNorm and each conv a linear over gathered neighbours.

// trellisConv is a Conv3d (stride 1, padding k/2) as a linear over the k³
// neighbours: W (Cout, k³·Cin) in (kx, ky, kz, cin) order.
type trellisConv struct {
	Lin *nn.Linear
	K   int
}

// trellisConvChunkBytes caps the gathered neighbour tensor of one conv
// call; larger inputs run in slabs along the first spatial axis.
var trellisConvChunkBytes = 64 << 20

func (cv *trellisConv) forward(o *iop, x *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	B, X, Y, Z, C := x.Dim(0), x.Dim(1), x.Dim(2), x.Dim(3), x.Dim(4)
	k := cv.K
	if k == 1 {
		return o.reshape(o.lin(cv.Lin, o.reshape(x, B*X*Y*Z, C)), B, X, Y, Z, cv.Lin.W.Dim(0))
	}
	p := k / 2
	// Zero padding, one axis at a time.
	pad := func(t *tensor.Tensor, axis int) *tensor.Tensor {
		if o.err != nil {
			return nil
		}
		d := t.Shape().Dims()
		d[axis] = p
		z := o.do(func() (*tensor.Tensor, error) { return tensor.Zeros(dtype.F32, d, t.Device()) })
		return o.cat(axis, z, t, z)
	}
	xp := pad(pad(pad(x, 1), 2), 3)
	rows := max(1, trellisConvChunkBytes/(B*Y*Z*k*k*k*C*4))
	var outs []*tensor.Tensor
	for x0 := 0; x0 < X && o.err == nil; x0 += rows {
		n := min(rows, X-x0)
		parts := make([]*tensor.Tensor, 0, k*k*k)
		for kx := range k {
			for ky := range k {
				for kz := range k {
					s := o.narrow(o.narrow(o.narrow(xp, 1, x0+kx, n), 2, ky, Y), 3, kz, Z)
					parts = append(parts, o.reshape(s, B*n*Y*Z, C))
				}
			}
		}
		col := o.cat(1, parts...)
		outs = append(outs, o.reshape(o.lin(cv.Lin, col), B, n, Y, Z, cv.Lin.W.Dim(0)))
	}
	if len(outs) == 1 {
		return outs[0]
	}
	return o.cat(1, outs...)
}

// trellisRes is ResBlock3d (layer norms): norm, SiLU, conv, norm, SiLU,
// conv, plus the (1x1-conv) skip.
type trellisRes struct {
	Norm1, Norm2 idolNorm
	Conv1, Conv2 *trellisConv
	Skip         *trellisConv // nil: identity
}

func (r *trellisRes) forward(o *iop, x *tensor.Tensor) *tensor.Tensor {
	h := r.Conv1.forward(o, o.un(ops.SiLU, o.ln(x, r.Norm1.W, r.Norm1.B, r.Norm1.Eps)))
	h = r.Conv2.forward(o, o.un(ops.SiLU, o.ln(h, r.Norm2.W, r.Norm2.B, r.Norm2.Eps)))
	skip := x
	if r.Skip != nil {
		skip = r.Skip.forward(o, x)
	}
	return o.add(h, skip)
}

// trellisDecBlock is a ResBlock3d or an UpsampleBlock3d (conv to 8C', then
// pixel_shuffle_3d by 2).
type trellisDecBlock struct {
	Res *trellisRes
	Up  *trellisConv // output channels reordered to (sx, sy, sz, c') at load
}

// shuffle is pixel_shuffle_3d for channel-last x (B, X, Y, Z, 8·C') whose
// channels are ordered (sx, sy, sz, c'): out[b, 2x+sx, 2y+sy, 2z+sz, c'].
func trellisShuffle(o *iop, x *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	B, X, Y, Z, C8 := x.Dim(0), x.Dim(1), x.Dim(2), x.Dim(3), x.Dim(4)
	c := C8 / 8
	t := o.perm(o.reshape(x, B*X, Y, Z, 4, 2*c), 0, 3, 1, 2, 4) // (BX, (sx, sy), Y, Z, (sz, c'))
	t = o.reshape(t, B*X, 2, 2, Y, 2*Z*c)                       // (BX, sx, sy, Y, (z, sz, c'))
	t = o.perm(t, 0, 1, 3, 2, 4)                                // (BX, sx, Y, sy, (z, sz, c'))
	return o.reshape(t, B, 2*X, 2*Y, 2*Z, c)
}

// TrellisSSDecoder is SparseStructureDecoder.
type TrellisSSDecoder struct {
	Args    TrellisDecoderArgs
	Input   *trellisConv
	Middle  []*trellisRes
	Blocks  []trellisDecBlock
	OutNorm idolNorm
	OutConv *trellisConv
}

// LoadTrellisSSDecoder loads ckpts/<name> (base path without extension)
// onto dev.
func LoadTrellisSSDecoder(base string, dev device.Device) (*TrellisSSDecoder, error) {
	a := TrellisDecoderArgs{NumResBlocksMiddle: 2, NormType: "layer"}
	if err := trellisConfig(base+".json", "SparseStructureDecoder", &a); err != nil {
		return nil, err
	}
	if a.NormType != "layer" {
		return nil, fmt.Errorf("trellis decoder: norm_type %q (only layer is supported)", a.NormType)
	}
	if len(a.Channels) == 0 {
		return nil, fmt.Errorf("trellis decoder: no channels")
	}
	l, err := newIdolLoader(base + ".safetensors")
	if err != nil {
		return nil, err
	}
	l.dev = dev
	m := &TrellisSSDecoder{Args: a}
	m.Input = l.conv3d("input_layer", false)
	norm := func(name string) idolNorm {
		return idolNorm{W: l.param(name + ".weight"), B: l.param(name + ".bias"), Eps: 1e-5}
	}
	res := func(p string) *trellisRes {
		r := &trellisRes{Norm1: norm(p + ".norm1"), Norm2: norm(p + ".norm2"),
			Conv1: l.conv3d(p+".conv1", false), Conv2: l.conv3d(p+".conv2", false)}
		if _, ok := l.hf[p+".skip_connection.weight"]; ok {
			r.Skip = l.conv3d(p+".skip_connection", false)
		}
		return r
	}
	for i := range a.NumResBlocksMiddle {
		m.Middle = append(m.Middle, res(fmt.Sprintf("middle_block.%d", i)))
	}
	n := 0
	for i := range a.Channels {
		for range a.NumResBlocks {
			m.Blocks = append(m.Blocks, trellisDecBlock{Res: res(fmt.Sprintf("blocks.%d", n))})
			n++
		}
		if i < len(a.Channels)-1 {
			m.Blocks = append(m.Blocks, trellisDecBlock{Up: l.conv3d(fmt.Sprintf("blocks.%d.conv", n), true)})
			n++
		}
	}
	m.OutNorm = norm("out_layer.0")
	m.OutConv = l.conv3d("out_layer.2", false)
	if l.err != nil {
		return nil, fmt.Errorf("trellis decoder: %w", l.err)
	}
	return m, nil
}

// conv3d loads a cubic, stride-1 Conv3d (Cout, Cin, k, k, k) as a
// trellisConv. shuffle reorders the output channels from pixel_shuffle_3d's
// (c', sx, sy, sz) to (sx, sy, sz, c').
func (l *idolLoader) conv3d(name string, shuffle bool) *trellisConv {
	if l.err != nil {
		return nil
	}
	w, ok := l.hf[name+".weight"]
	if !ok || w.Rank() != 5 || w.Dim(2) != w.Dim(3) || w.Dim(2) != w.Dim(4) {
		l.err = fmt.Errorf("missing or non-cubic conv %s.weight", name)
		return nil
	}
	co, ci, k := w.Dim(0), w.Dim(1), w.Dim(2)
	k3 := k * k * k
	src := l.host(name + ".weight")
	if l.err != nil {
		return nil
	}
	row := func(o int) int { // destination row of output channel o
		if shuffle {
			return (o%8)*(co/8) + o/8
		}
		return o
	}
	dst := make([]float32, len(src))
	for o := range co {
		r := row(o)
		for c := range ci {
			for s := range k3 {
				dst[(r*k3+s)*ci+c] = src[(o*ci+c)*k3+s]
			}
		}
	}
	if l.hf[name+".weight"], l.err = tensor.New(dst, []int{co, k3 * ci}, device.CPU()); l.err != nil {
		return nil
	}
	if shuffle {
		if b := l.host(name + ".bias"); l.err == nil {
			bd := make([]float32, len(b))
			for o := range b {
				bd[row(o)] = b[o]
			}
			if l.hf[name+".bias"], l.err = tensor.New(bd, []int{co}, device.CPU()); l.err != nil {
				return nil
			}
		}
	}
	return &trellisConv{Lin: l.linear(name), K: k}
}

// Forward returns the occupancy logits (B, out, 4R, 4R, 4R) (for two
// upsamplings) of the latent z (B, latent, R, R, R).
func (m *TrellisSSDecoder) Forward(z *tensor.Tensor) (*tensor.Tensor, error) {
	o := &iop{}
	h := m.Input.forward(o, o.perm(z, 0, 2, 3, 4, 1))
	for _, r := range m.Middle {
		h = r.forward(o, h)
	}
	for _, b := range m.Blocks {
		if b.Res != nil {
			h = b.Res.forward(o, h)
		} else {
			h = trellisShuffle(o, b.Up.forward(o, h))
		}
	}
	h = o.un(ops.SiLU, o.ln(h, m.OutNorm.W, m.OutNorm.B, m.OutNorm.Eps))
	h = m.OutConv.forward(o, h)
	if o.err != nil {
		return nil, o.err
	}
	out := o.reshape(o.perm(h, 0, 4, 1, 2, 3), h.Dim(0), h.Dim(4), h.Dim(1), h.Dim(2), h.Dim(3))
	return out, o.err
}

// TrellisCoords is sample_sparse_structure's coords: argwhere(logits > 0)
// over (B, C, X, Y, Z) logits, keeping (b, x, y, z), in row-major order.
func TrellisCoords(logits *tensor.Tensor) ([][4]int32, error) {
	h, err := logits.To(device.CPU())
	if err != nil {
		return nil, err
	}
	if h, err = h.Contiguous(); err != nil {
		return nil, err
	}
	B, C, X, Y, Z := h.Dim(0), h.Dim(1), h.Dim(2), h.Dim(3), h.Dim(4)
	v := tensor.MustSlice[float32](h)
	var out [][4]int32
	for b := range B {
		for c := range C {
			for x := range X {
				for y := range Y {
					for z := range Z {
						if v[(((b*C+c)*X+x)*Y+y)*Z+z] > 0 {
							out = append(out, [4]int32{int32(b), int32(x), int32(y), int32(z)})
						}
					}
				}
			}
		}
	}
	return out, nil
}
