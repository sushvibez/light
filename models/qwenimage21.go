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
	"light/tensor"
)

// QwenImage21Config is diffusers' QwenImage21Transformer2DModel config.
type QwenImage21Config struct {
	PatchSize       int     `json:"patch_size"`
	InChannels      int     `json:"in_channels"`
	OutChannels     int     `json:"out_channels"`
	NumLayers       int     `json:"num_layers"`
	HeadDim         int     `json:"attention_head_dim"`
	NumHeads        int     `json:"num_attention_heads"`
	ContextInDim    int     `json:"context_in_dim"`
	MLPRatio        int     `json:"mlp_ratio"`
	AxesDims        []int   `json:"axes_dims_rope"`
	Eps             float64 `json:"eps"`
	CausalCondition *bool   `json:"causal_condition"`
}

func (c QwenImage21Config) inner() int { return c.NumHeads * c.HeadDim }

type qwenImage21Block struct {
	Attn             fluxAttn // to_q / to_k / to_v + QK RMSNorm
	Out              *nn.Linear
	Proj, Gate, Down *nn.Linear // SwiGLU: down(silu(gate(x)) · proj(x))
}

// QwenImage21Transformer is the Qwen-Image 2.1 single-stream DiT: text and
// image tokens share one sequence under a block-causal mask (causal over the
// sequence, bidirectional inside each image block), one modulation shared by
// every block (scale and tanh gate, no shift), 3-axis RoPE (frame, height,
// width) and, with causal_condition, text tokens modulated from t = 0.
type QwenImage21Transformer struct {
	Cfg            QwenImage21Config
	TimeL1, TimeL2 *nn.Linear
	TxtNorm        *tensor.Tensor // zero-centred RMSNorm weight + 1, (context_in_dim,)
	TxtIn, TxtOut  *nn.Linear
	ImgIn          *nn.Linear
	Modulation     *nn.Linear // D → 4D: [mod1 scale, mod1 gate, mod2 scale, mod2 gate]
	Blocks         []*qwenImage21Block
	NormOut        *nn.Linear // D → D (scale only)
	ProjOut        *nn.Linear
	ones, zeros    *tensor.Tensor
}

// NewQwenImage21Transformer allocates an uninitialised F32 model.
func NewQwenImage21Transformer(cfg QwenImage21Config, dev device.Device) (*QwenImage21Transformer, error) {
	if cfg.PatchSize == 0 {
		cfg.PatchSize = 1
	}
	if cfg.OutChannels == 0 {
		cfg.OutChannels = cfg.InChannels
	}
	if cfg.MLPRatio == 0 {
		cfg.MLPRatio = 3
	}
	if cfg.Eps == 0 {
		cfg.Eps = 1e-6
	}
	if cfg.PatchSize != 1 {
		return nil, fmt.Errorf("qwen-image 2.1: patch_size %d not supported (1)", cfg.PatchSize)
	}
	sum := 0
	for _, d := range cfg.AxesDims {
		sum += d
	}
	if len(cfg.AxesDims) != 3 || sum != cfg.HeadDim {
		return nil, fmt.Errorf("qwen-image 2.1: axes_dims_rope %v must be 3 axes summing to the head dim %d", cfg.AxesDims, cfg.HeadDim)
	}
	D, M := cfg.inner(), cfg.inner()*cfg.MLPRatio
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
	m := &QwenImage21Transformer{Cfg: cfg,
		TimeL1: lin(256, D), TimeL2: lin(D, D),
		TxtIn: lin(cfg.ContextInDim, D), TxtOut: lin(D, D), ImgIn: lin(cfg.InChannels, D),
		Modulation: lin(D, 4*D), NormOut: lin(D, D), ProjOut: lin(D, cfg.OutChannels)}
	for range cfg.NumLayers {
		m.Blocks = append(m.Blocks, &qwenImage21Block{
			Attn: fluxAttn{Q: lin(D, D), K: lin(D, D), V: lin(D, D), NormQ: rms(), NormK: rms()},
			Out:  lin(D, D), Proj: lin(D, M), Gate: lin(D, M), Down: lin(M, D)})
	}
	if err != nil {
		return nil, err
	}
	if m.TxtNorm, err = tensor.Ones(dtype.F32, []int{cfg.ContextInDim}, dev); err != nil {
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

// QwenImage21Inputs are one text-to-image transformer call (batch 1): the
// target image's latent tokens on an h x w grid after the text.
type QwenImage21Inputs struct {
	Image    *tensor.Tensor // (1, h·w, in_channels)
	Text     *tensor.Tensor // (1, L, context_in_dim)
	TextLen  int            // valid text tokens; the rest are right padding (masked as keys)
	H, W     int            // the target's latent token grid
	Timestep float32        // in [0, 1]
}

// qwenImage21RoPE returns cos / sin (1, S, 1, HD) for the joint sequence of
// L text tokens followed by an h x w image block: text tokens sit at
// position i on all three axes; the image at frame L with a height / width
// grid centred on zero. Angles are F32 products as in the reference, laid
// out per rotated pair (interleaved halves).
func qwenImage21RoPE(axes []int, L, h, w int, dev device.Device) (*tensor.Tensor, *tensor.Tensor, error) {
	HD := axes[0] + axes[1] + axes[2]
	S := L + h*w
	pos := make([][3]int, 0, S)
	for i := range L {
		pos = append(pos, [3]int{i, i, i})
	}
	for r := range h {
		for c := range w {
			pos = append(pos, [3]int{L, r - (h - h/2), c - (w - w/2)})
		}
	}
	inv := make([][]float32, 3)
	for a, d := range axes {
		inv[a] = make([]float32, d/2)
		for j := range d / 2 {
			inv[a][j] = float32(1 / math.Pow(10000, float64(float32(2*j)/float32(d))))
		}
	}
	cos := make([]float32, S*HD)
	sin := make([]float32, S*HD)
	for s, p := range pos {
		o := s * HD
		for a := range 3 {
			for _, f := range inv[a] {
				ang := float64(float32(p[a]) * f)
				c, sn := float32(math.Cos(ang)), float32(math.Sin(ang))
				cos[o], cos[o+1] = c, c
				sin[o], sin[o+1] = sn, sn
				o += 2
			}
		}
	}
	ct, err := tensor.New(cos, []int{1, S, 1, HD}, dev)
	if err != nil {
		return nil, nil, err
	}
	st, err := tensor.New(sin, []int{1, S, 1, HD}, dev)
	return ct, st, err
}

// qwenImage21Mask is the additive (S, S) block-causal mask: key j is
// visible from query i when j <= i or both are in the image block, and
// never when j is a padded text position.
func qwenImage21Mask(L, valid, nImg int, dev device.Device) (*tensor.Tensor, error) {
	S := L + nImg
	neg := float32(math.Inf(-1))
	m := make([]float32, S*S)
	for i := range S {
		for j := range S {
			ok := j <= i || (i >= L && j >= L)
			if j < L && j >= valid {
				ok = false
			}
			if !ok {
				m[i*S+j] = neg
			}
		}
	}
	return tensor.New(m, []int{1, 1, S, S}, dev)
}

// Forward predicts the target image's velocity (1, h·w, out_channels).
func (m *QwenImage21Transformer) Forward(in QwenImage21Inputs) (*tensor.Tensor, error) {
	cfg := m.Cfg
	dev := in.Image.Device()
	L, nImg := in.Text.Dim(1), in.H*in.W
	if in.Image.Dim(1) != nImg {
		return nil, fmt.Errorf("qwen-image 2.1: %d image tokens for a %d x %d grid", in.Image.Dim(1), in.H, in.W)
	}
	valid := in.TextLen
	if valid <= 0 || valid > L {
		valid = L
	}
	D, H, HD := cfg.inner(), cfg.NumHeads, cfg.HeadDim
	// Text projection: zero-centred RMSNorm (weight + 1), Linear, tanh GELU, Linear.
	t, err := ops.RMSNorm(in.Text, m.TxtNorm, cfg.Eps)
	if err != nil {
		return nil, err
	}
	if t, err = m.TxtIn.Forward(t); err != nil {
		return nil, err
	}
	if t, err = ops.Gelu(t); err != nil {
		return nil, err
	}
	if t, err = m.TxtOut.Forward(t); err != nil {
		return nil, err
	}
	x, err := m.ImgIn.Forward(in.Image)
	if err != nil {
		return nil, err
	}
	h, err := ops.Concat([]*tensor.Tensor{t, x}, 1) // (1, S, D)
	if err != nil {
		return nil, err
	}
	// Timestep embeddings: the image's, and t = 0 for the text (causal_condition).
	ts := []float32{in.Timestep * 1000}
	causal := cfg.CausalCondition == nil || *cfg.CausalCondition
	if causal {
		ts = append(ts, 0)
	}
	p, err := timestepEmbedding(ts, dev)
	if err != nil {
		return nil, err
	}
	temb, err := mlpSiLU(m.TimeL1, m.TimeL2, p) // (1 or 2, D)
	if err != nil {
		return nil, err
	}
	act, err := ops.SiLU(temb)
	if err != nil {
		return nil, err
	}
	mod, err := m.Modulation.Forward(act) // (rows, 4D)
	if err != nil {
		return nil, err
	}
	// perToken spreads a (rows, n) parameter over the sequence: text tokens
	// take the t = 0 row, image tokens the sampled one.
	perToken := func(par *tensor.Tensor, off, n int) (*tensor.Tensor, error) {
		img, err := par.Narrow(0, 0, 1)
		if err != nil {
			return nil, err
		}
		if img, err = narrowContig(img, 1, off, n); err != nil {
			return nil, err
		}
		if img, err = img.Reshape(1, 1, n); err != nil {
			return nil, err
		}
		if !causal {
			return img, nil
		}
		txt, err := par.Narrow(0, 1, 1)
		if err != nil {
			return nil, err
		}
		if txt, err = narrowContig(txt, 1, off, n); err != nil {
			return nil, err
		}
		if txt, err = txt.Reshape(1, 1, n); err != nil {
			return nil, err
		}
		a, err := txt.Expand(1, L, n)
		if err != nil {
			return nil, err
		}
		b, err := img.Expand(1, nImg, n)
		if err != nil {
			return nil, err
		}
		return ops.Concat([]*tensor.Tensor{a, b}, 1)
	}
	var mods [4]*tensor.Tensor // scale1, gate1, scale2, gate2
	for i := range mods {
		if mods[i], err = perToken(mod, i*D, D); err != nil {
			return nil, err
		}
	}
	for _, i := range []int{1, 3} {
		if mods[i], err = ops.Tanh(mods[i]); err != nil {
			return nil, err
		}
	}
	cos, sin, err := qwenImage21RoPE(cfg.AxesDims, L, in.H, in.W, dev)
	if err != nil {
		return nil, err
	}
	mask, err := qwenImage21Mask(L, valid, nImg, dev)
	if err != nil {
		return nil, err
	}
	scaleNorm := func(x, scale *tensor.Tensor) (*tensor.Tensor, error) {
		n, err := ops.LayerNorm(x, m.ones, m.zeros, cfg.Eps)
		if err != nil {
			return nil, err
		}
		s1, err := ops.Add(scale, m.ones)
		if err != nil {
			return nil, err
		}
		return ops.Mul(n, s1)
	}
	S := L + nImg
	for bi, b := range m.Blocks {
		n, err := scaleNorm(h, mods[0])
		if err != nil {
			return nil, err
		}
		q, k, v, err := b.Attn.qkv(n, H, HD)
		if err != nil {
			return nil, err
		}
		if q, err = applyFluxRoPE(q, cos, sin); err != nil {
			return nil, err
		}
		if k, err = applyFluxRoPE(k, cos, sin); err != nil {
			return nil, err
		}
		perm := func(t *tensor.Tensor) (*tensor.Tensor, error) {
			p, err := t.Permute(0, 2, 1, 3)
			if err != nil {
				return nil, err
			}
			return p.Contiguous()
		}
		if q, err = perm(q); err != nil {
			return nil, err
		}
		if k, err = perm(k); err != nil {
			return nil, err
		}
		if v, err = perm(v); err != nil {
			return nil, err
		}
		o, err := nn.SDPA(q, k, v, mask)
		if err != nil {
			return nil, fmt.Errorf("qwen-image 2.1 block %d: %w", bi, err)
		}
		if o, err = perm(o); err != nil {
			return nil, err
		}
		if o, err = o.Reshape(1, S, D); err != nil {
			return nil, err
		}
		if o, err = b.Out.Forward(o); err != nil {
			return nil, err
		}
		if h, err = gateAdd(h, mods[1], o); err != nil {
			return nil, err
		}
		if n, err = scaleNorm(h, mods[2]); err != nil {
			return nil, err
		}
		g, err := b.Gate.Forward(n)
		if err != nil {
			return nil, err
		}
		if g, err = ops.SiLU(g); err != nil {
			return nil, err
		}
		pj, err := b.Proj.Forward(n)
		if err != nil {
			return nil, err
		}
		if g, err = ops.Mul(g, pj); err != nil {
			return nil, err
		}
		if g, err = b.Down.Forward(g); err != nil {
			return nil, err
		}
		if h, err = gateAdd(h, mods[3], g); err != nil {
			return nil, err
		}
	}
	// Scale-only output norm (per token, as the modulation), then the target's tokens.
	so, err := m.NormOut.Forward(act)
	if err != nil {
		return nil, err
	}
	sc, err := perToken(so, 0, D)
	if err != nil {
		return nil, err
	}
	if h, err = scaleNorm(h, sc); err != nil {
		return nil, err
	}
	if h, err = h.Narrow(1, L, nImg); err != nil {
		return nil, err
	}
	if h, err = h.Contiguous(); err != nil {
		return nil, err
	}
	return m.ProjOut.Forward(h)
}

// LoadQwenImage21Transformer loads a QwenImage21Transformer2DModel folder.
func LoadQwenImage21Transformer(dir string, dev device.Device) (*QwenImage21Transformer, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	cfg := QwenImage21Config{PatchSize: 1, InChannels: 64, MLPRatio: 3, Eps: 1e-6}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("qwen-image 2.1 config: %w", err)
	}
	hf, err := loadCheckpointTensors(dir)
	if err != nil {
		return nil, err
	}
	m, err := NewQwenImage21Transformer(cfg, dev)
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
	lin(m.TimeL1, "time_text_embed.timestep_embedder.linear_1")
	lin(m.TimeL2, "time_text_embed.timestep_embedder.linear_2")
	lin(m.TxtIn, "txt_in.in_layer")
	lin(m.TxtOut, "txt_in.out_layer")
	lin(m.ImgIn, "img_in")
	lin(m.Modulation, "modulation.1")
	lin(m.NormOut, "norm_out.linear")
	lin(m.ProjOut, "proj_out")
	for i, b := range m.Blocks {
		p := fmt.Sprintf("transformer_blocks.%d.", i)
		lin(b.Attn.Q, p+"attn.to_q")
		lin(b.Attn.K, p+"attn.to_k")
		lin(b.Attn.V, p+"attn.to_v")
		cp(b.Attn.NormQ.Weight, p+"attn.norm_q.weight")
		cp(b.Attn.NormK.Weight, p+"attn.norm_k.weight")
		lin(b.Out, p+"attn.to_out.0")
		lin(b.Proj, p+"img_mlp.proj")
		lin(b.Gate, p+"img_mlp.gate_layer")
		lin(b.Down, p+"img_mlp.out")
	}
	// The text norm stores scale - 1.
	if lerr == nil {
		w, ok := hf["txt_in.text_norm.weight"]
		if !ok {
			lerr = fmt.Errorf("missing txt_in.text_norm.weight")
		} else {
			if w, lerr = ops.Cast(w, dtype.F32); lerr == nil {
				one, _ := tensor.New([]float32{1}, []int{1}, w.Device())
				if w, lerr = ops.Add(w, one); lerr == nil {
					m.TxtNorm, lerr = w.To(dev)
				}
			}
		}
	}
	if lerr != nil {
		return nil, fmt.Errorf("LoadQwenImage21Transformer: %w", lerr)
	}
	return m, nil
}
