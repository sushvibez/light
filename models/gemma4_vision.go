package models

import (
	"encoding/json"
	"fmt"
	"image"
	"math"
	"os"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/hub"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// Gemma4VisionConfig is a Gemma 4 checkpoint's vision_config.
type Gemma4VisionConfig struct {
	HiddenSize       int     `json:"hidden_size"`
	IntermediateSize int     `json:"intermediate_size"`
	NumLayers        int     `json:"num_hidden_layers"`
	NumHeads         int     `json:"num_attention_heads"`
	NumKVHeads       int     `json:"num_key_value_heads"`
	HeadDim          int     `json:"head_dim"`
	PatchSize        int     `json:"patch_size"`
	PoolingKernel    int     `json:"pooling_kernel_size"`
	PosEmbedSize     int     `json:"position_embedding_size"`
	Eps              float64 `json:"rms_norm_eps"`
	Standardize      bool    `json:"standardize"`
	ClippedLinears   bool    `json:"use_clipped_linears"`
	HiddenAct        string  `json:"hidden_activation"`
	SoftTokens       int     `json:"default_output_length"`
	RopeTheta        float64 `json:"-"`
	TextDim          int     `json:"-"`
}

// clippedLinear is Gemma4ClippableLinear: the input and output clamped to
// stored bounds around a bias-free linear.
type clippedLinear struct {
	L              *nn.Linear
	inMin, inMax   float64
	outMin, outMax float64
}

func (c *clippedLinear) forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	var err error
	if !math.IsInf(c.inMin, -1) || !math.IsInf(c.inMax, 1) {
		if x, err = ops.Clamp(x, c.inMin, c.inMax); err != nil {
			return nil, err
		}
	}
	if x, err = c.L.Forward(x); err != nil {
		return nil, err
	}
	if !math.IsInf(c.outMin, -1) || !math.IsInf(c.outMax, 1) {
		return ops.Clamp(x, c.outMin, c.outMax)
	}
	return x, nil
}

// Gemma4Vision is the Gemma 4 vision encoder and its embedder into the LM:
// linear patch embedding with summed learned x / y position tables, pre-
// and post-normed blocks (RMSNorm with direct weights) with q / k / v
// norms, unscaled attention and 2D RoPE (x on the first half of each head,
// y on the second), gated GELU MLPs, clipped linears; then 3 × 3 average
// pooling by patch position, √hidden scaling, a weightless RMSNorm and the
// projection to the LM width.
type Gemma4Vision struct {
	Cfg     Gemma4VisionConfig
	Input   *nn.Linear
	PosX    *tensor.Tensor // (PosEmbedSize, D)
	PosY    *tensor.Tensor
	Layers  []*gemma4VisionLayer
	Project *nn.Linear // embed_vision.embedding_projection
	ones    *tensor.Tensor
}

type gemma4VisionLayer struct {
	InNorm, PostAttn, PreFF, PostFF *nn.RMSNorm
	QNorm, KNorm                    *nn.RMSNorm
	Q, K, V, O                      *clippedLinear
	Gate, Up, Down                  *clippedLinear
}

// Gemma4Image is a preprocessed image: patches (n, P·P·3) in [0, 1]
// ordered row-major with (y, x, c) inside each patch, and the patch grid.
type Gemma4Image struct {
	Patches []float32
	PH, PW  int
}

// SoftTokens is the number of LM tokens the image becomes.
func (im Gemma4Image) SoftTokens(k int) int { return (im.PH / k) * (im.PW / k) }

// Gemma4Preprocess resizes an image as Gemma4ImageProcessor does (aspect
// preserved, the largest size within max soft tokens · k² patches whose
// sides are multiples of k·patch; Pillow bicubic) and rescales to [0, 1].
func (m *Gemma4Vision) Gemma4Preprocess(img image.Image) (Gemma4Image, error) {
	rgb, w, h := rgbBytes(img)
	P, k := m.Cfg.PatchSize, m.Cfg.PoolingKernel
	maxPatches := m.Cfg.SoftTokens * k * k
	target := float64(maxPatches * P * P)
	f := math.Sqrt(target / float64(h*w))
	side := k * P
	th := int(math.Floor(f*float64(h)/float64(side))) * side
	tw := int(math.Floor(f*float64(w)/float64(side))) * side
	maxSide := (maxPatches / (k * k)) * side
	switch {
	case th == 0 && tw == 0:
		return Gemma4Image{}, fmt.Errorf("gemma4 vision: image %dx%d too small", w, h)
	case th == 0:
		th, tw = side, min(int(math.Floor(float64(w)/float64(h)))*side, maxSide)
	case tw == 0:
		tw, th = side, min(int(math.Floor(float64(h)/float64(w)))*side, maxSide)
	}
	if th != h || tw != w {
		rgb = pilResizeRGB(rgb, w, h, tw, th)
	}
	ph, pw := th/P, tw/P
	out := make([]float32, ph*pw*P*P*3)
	const rescale = float32(1.0 / 255)
	n := 0
	for r := range ph {
		for c := range pw {
			for y := range P {
				for x := range P {
					for ch := range 3 {
						out[n] = float32(rgb[((r*P+y)*tw+c*P+x)*3+ch]) * rescale
						n++
					}
				}
			}
		}
	}
	return Gemma4Image{Patches: out, PH: ph, PW: pw}, nil
}

// visionRoPE2D is the cos / sin (n, 1, Dh): the first half of each head
// rotates with x (rotate-half over Dh/2), the second with y.
func (m *Gemma4Vision) rope2D(xs, ys []int, dev device.Device) (*tensor.Tensor, *tensor.Tensor, error) {
	Dh := m.Cfg.HeadDim
	half := Dh / 2
	nf := half / 2
	inv := make([]float32, nf)
	for j := range nf {
		inv[j] = float32(1 / math.Pow(m.Cfg.RopeTheta, float64(2*j)/float64(half)))
	}
	n := len(xs)
	cs, sn := make([]float32, n*Dh), make([]float32, n*Dh)
	for p := range n {
		for a, pos := range []int{xs[p], ys[p]} {
			for j := range nf {
				ang := float32(pos) * inv[j]
				c, s := float32(math.Cos(float64(ang))), float32(math.Sin(float64(ang)))
				for _, off := range []int{0, nf} {
					cs[p*Dh+a*half+off+j], sn[p*Dh+a*half+off+j] = c, s
				}
			}
		}
	}
	c, err := tensor.New(cs, []int{n, 1, Dh}, dev)
	if err != nil {
		return nil, nil, err
	}
	s, err := tensor.New(sn, []int{n, 1, Dh}, dev)
	return c, s, err
}

// Forward encodes images into LM tokens (Σ soft tokens, TextDim).
func (m *Gemma4Vision) Forward(images []Gemma4Image, dev device.Device) (*tensor.Tensor, error) {
	var outs []*tensor.Tensor
	for _, im := range images {
		o, err := m.forwardOne(im, dev)
		if err != nil {
			return nil, err
		}
		outs = append(outs, o)
	}
	if len(outs) == 1 {
		return outs[0], nil
	}
	return ops.Concat(outs, 0)
}

func (m *Gemma4Vision) forwardOne(im Gemma4Image, dev device.Device) (*tensor.Tensor, error) {
	cfg := m.Cfg
	D, H, Dh, k := cfg.HiddenSize, cfg.NumHeads, cfg.HeadDim, cfg.PoolingKernel
	n := im.PH * im.PW
	x, err := tensor.New(im.Patches, []int{n, len(im.Patches) / n}, dev)
	if err != nil {
		return nil, err
	}
	// 2·(v − 0.5)
	two, _ := tensor.New([]float32{2}, []int{1}, dev)
	one, _ := tensor.New([]float32{1}, []int{1}, dev)
	if x, err = ops.Mul(x, two); err != nil {
		return nil, err
	}
	if x, err = ops.Sub(x, one); err != nil {
		return nil, err
	}
	h, err := m.Input.Forward(x)
	if err != nil {
		return nil, err
	}
	xs, ys := make([]int, n), make([]int, n)
	xi, yi := make([]int64, n), make([]int64, n)
	for r := range im.PH {
		for c := range im.PW {
			i := r*im.PW + c
			xs[i], ys[i], xi[i], yi[i] = c, r, int64(c), int64(r)
		}
	}
	xt, _ := tensor.New(xi, []int{n}, dev)
	yt, _ := tensor.New(yi, []int{n}, dev)
	px, err := ops.IndexSelect(m.PosX, xt)
	if err != nil {
		return nil, err
	}
	py, err := ops.IndexSelect(m.PosY, yt)
	if err != nil {
		return nil, err
	}
	if px, err = ops.Add(px, py); err != nil {
		return nil, err
	}
	if h, err = ops.Add(h, px); err != nil {
		return nil, err
	}
	cos, sin, err := m.rope2D(xs, ys, dev)
	if err != nil {
		return nil, err
	}
	vNorm, err := nn.NewRMSNormWithDType(Dh, cfg.Eps, dtype.F32, dev) // weightless (ones)
	if err != nil {
		return nil, err
	}
	for li, l := range m.Layers {
		nrm, err := l.InNorm.Forward(h)
		if err != nil {
			return nil, err
		}
		proj := func(c *clippedLinear, norm *nn.RMSNorm, rope bool) (*tensor.Tensor, error) {
			y, err := c.forward(nrm)
			if err != nil {
				return nil, err
			}
			if y, err = y.Reshape(n, H, Dh); err != nil {
				return nil, err
			}
			if y, err = norm.Forward(y); err != nil {
				return nil, err
			}
			if rope {
				lo, err := y.Narrow(2, 0, Dh/2)
				if err != nil {
					return nil, err
				}
				hi, err := y.Narrow(2, Dh/2, Dh/2)
				if err != nil {
					return nil, err
				}
				cl, _ := cos.Narrow(2, 0, Dh/2)
				sl, _ := sin.Narrow(2, 0, Dh/2)
				ch, _ := cos.Narrow(2, Dh/2, Dh/2)
				sh, _ := sin.Narrow(2, Dh/2, Dh/2)
				if lo, err = rotateHalfLast(lo, cl, sl); err != nil {
					return nil, err
				}
				if hi, err = rotateHalfLast(hi, ch, sh); err != nil {
					return nil, err
				}
				if y, err = ops.Concat([]*tensor.Tensor{lo, hi}, 2); err != nil {
					return nil, err
				}
			}
			if y, err = autograd.Permute(y, 1, 0, 2); err != nil {
				return nil, err
			}
			return autograd.Reshape(y, 1, H, n, Dh)
		}
		q, err := proj(l.Q, l.QNorm, true)
		if err != nil {
			return nil, err
		}
		kk, err := proj(l.K, l.KNorm, true)
		if err != nil {
			return nil, err
		}
		v, err := proj(l.V, vNorm, false)
		if err != nil {
			return nil, err
		}
		a, err := nn.SDPAScaled(q, kk, v, nil, 1)
		if err != nil {
			return nil, fmt.Errorf("gemma4 vision layer %d: %w", li, err)
		}
		if a, err = autograd.Permute(a, 0, 2, 1, 3); err != nil {
			return nil, err
		}
		if a, err = a.Contiguous(); err != nil {
			return nil, err
		}
		if a, err = a.Reshape(n, H*Dh); err != nil {
			return nil, err
		}
		if a, err = l.O.forward(a); err != nil {
			return nil, err
		}
		if a, err = l.PostAttn.Forward(a); err != nil {
			return nil, err
		}
		if h, err = ops.Add(h, a); err != nil {
			return nil, err
		}
		if nrm, err = l.PreFF.Forward(h); err != nil {
			return nil, err
		}
		g, err := l.Gate.forward(nrm)
		if err != nil {
			return nil, err
		}
		if g, err = ops.Gelu(g); err != nil {
			return nil, err
		}
		u, err := l.Up.forward(nrm)
		if err != nil {
			return nil, err
		}
		if g, err = ops.Mul(g, u); err != nil {
			return nil, err
		}
		f, err := l.Down.forward(g)
		if err != nil {
			return nil, err
		}
		if f, err = l.PostFF.Forward(f); err != nil {
			return nil, err
		}
		if h, err = ops.Add(h, f); err != nil {
			return nil, err
		}
	}
	// Pool k × k blocks: (PH/k, k, PW/k, k, D) → mean over the two k axes.
	if h, err = h.Reshape(im.PH/k, k, im.PW/k, k, D); err != nil {
		return nil, err
	}
	if h, err = h.Permute(0, 2, 1, 3, 4); err != nil {
		return nil, err
	}
	if h, err = h.Contiguous(); err != nil {
		return nil, err
	}
	if h, err = h.Reshape((im.PH/k)*(im.PW/k), k*k, D); err != nil {
		return nil, err
	}
	if h, err = ops.Sum(h, ops.ReduceOptions{Axes: []int{1}}); err != nil {
		return nil, err
	}
	sc, _ := tensor.New([]float32{float32(math.Sqrt(float64(D))) / float32(k*k)}, []int{1}, dev)
	if h, err = ops.Mul(h, sc); err != nil {
		return nil, err
	}
	norm, err := nn.NewRMSNormWithDType(D, cfg.Eps, dtype.F32, dev)
	if err != nil {
		return nil, err
	}
	if h, err = norm.Forward(h); err != nil {
		return nil, err
	}
	return m.Project.Forward(h)
}

// Gemma4ImageTokens is the placeholder run for one image: <boi>, one image
// token per soft token, <eoi>.
func (m *Gemma4Vision) Gemma4ImageTokens(im Gemma4Image, boi, img, eoi int64) []int64 {
	out := []int64{boi}
	for range im.SoftTokens(m.Cfg.PoolingKernel) {
		out = append(out, img)
	}
	return append(out, eoi)
}

// LoadGemma4VisionFromHF loads the vision tower and embedder of a Gemma 4
// checkpoint (model.vision_tower.*, model.embed_vision.*) at F32 on dev.
func LoadGemma4VisionFromHF(repo string, dev device.Device) (*Gemma4Vision, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return nil, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c struct {
		Vision *struct {
			Gemma4VisionConfig
			Rope struct {
				Theta float64 `json:"rope_theta"`
			} `json:"rope_parameters"`
		} `json:"vision_config"`
		Text struct {
			HiddenSize int `json:"hidden_size"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, err
	}
	if c.Vision == nil {
		return nil, fmt.Errorf("%s: no vision_config", repo)
	}
	cfg := c.Vision.Gemma4VisionConfig
	cfg.RopeTheta, cfg.TextDim = c.Vision.Rope.Theta, c.Text.HiddenSize
	if cfg.Standardize {
		return nil, fmt.Errorf("gemma4 vision: standardize not supported")
	}
	if cfg.HiddenAct != "gelu_pytorch_tanh" || cfg.NumKVHeads != cfg.NumHeads {
		return nil, fmt.Errorf("gemma4 vision: activation %q / KV heads %d not supported", cfg.HiddenAct, cfg.NumKVHeads)
	}
	if cfg.SoftTokens == 0 {
		cfg.SoftTokens = 280
	}
	D, Dh := cfg.HiddenSize, cfg.HeadDim
	hf, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, err
	}
	var lerr error
	lin := func(in, out int, name string) *nn.Linear {
		if lerr != nil {
			return nil
		}
		var l *nn.Linear
		if l, lerr = nn.NewLinearUninit(in, out, false, dtype.F32, dev); lerr == nil {
			lerr = copyExactCast(l.W, hf, name)
		}
		return l
	}
	scalar := func(name string, def float64) float64 {
		t, ok := hf[name]
		if !ok {
			return def
		}
		t32, err := ops.Cast(t, dtype.F32)
		if err != nil {
			lerr = err
			return def
		}
		return float64(tensor.MustSlice[float32](t32)[0])
	}
	clip := func(in, out int, name string) *clippedLinear {
		cl := &clippedLinear{L: lin(in, out, name+".linear.weight"),
			inMin: math.Inf(-1), inMax: math.Inf(1), outMin: math.Inf(-1), outMax: math.Inf(1)}
		if cfg.ClippedLinears {
			cl.inMin, cl.inMax = scalar(name+".input_min", cl.inMin), scalar(name+".input_max", cl.inMax)
			cl.outMin, cl.outMax = scalar(name+".output_min", cl.outMin), scalar(name+".output_max", cl.outMax)
		}
		return cl
	}
	rms := func(n int, name string) *nn.RMSNorm {
		if lerr != nil {
			return nil
		}
		var r *nn.RMSNorm
		if r, lerr = nn.NewRMSNormWithDType(n, cfg.Eps, dtype.F32, dev); lerr == nil {
			lerr = copyExactCast(r.Weight, hf, name)
		}
		return r
	}
	const p = "model.vision_tower."
	m := &Gemma4Vision{Cfg: cfg, Input: lin(3*cfg.PatchSize*cfg.PatchSize, D, p+"patch_embedder.input_proj.weight"),
		Project: lin(D, cfg.TextDim, "model.embed_vision.embedding_projection.weight")}
	if lerr == nil {
		pt, ok := hf[p+"patch_embedder.position_embedding_table"]
		if !ok {
			return nil, fmt.Errorf("%s: no position_embedding_table", repo)
		}
		for i, dst := range []**tensor.Tensor{&m.PosX, &m.PosY} {
			s, err := pt.Narrow(0, i, 1)
			if err != nil {
				return nil, err
			}
			if s, err = s.Contiguous(); err != nil {
				return nil, err
			}
			if s, err = s.Reshape(cfg.PosEmbedSize, D); err != nil {
				return nil, err
			}
			if *dst, err = ops.Cast(s, dtype.F32); err != nil {
				return nil, err
			}
			if *dst, err = (*dst).To(dev); err != nil {
				return nil, err
			}
		}
	}
	for i := range cfg.NumLayers {
		lp := fmt.Sprintf("%sencoder.layers.%d.", p, i)
		m.Layers = append(m.Layers, &gemma4VisionLayer{
			InNorm: rms(D, lp+"input_layernorm.weight"), PostAttn: rms(D, lp+"post_attention_layernorm.weight"),
			PreFF: rms(D, lp+"pre_feedforward_layernorm.weight"), PostFF: rms(D, lp+"post_feedforward_layernorm.weight"),
			QNorm: rms(Dh, lp+"self_attn.q_norm.weight"), KNorm: rms(Dh, lp+"self_attn.k_norm.weight"),
			Q: clip(D, D, lp+"self_attn.q_proj"), K: clip(D, D, lp+"self_attn.k_proj"),
			V: clip(D, D, lp+"self_attn.v_proj"), O: clip(D, D, lp+"self_attn.o_proj"),
			Gate: clip(D, cfg.IntermediateSize, lp+"mlp.gate_proj"), Up: clip(D, cfg.IntermediateSize, lp+"mlp.up_proj"),
			Down: clip(cfg.IntermediateSize, D, lp+"mlp.down_proj"),
		})
	}
	if lerr != nil {
		return nil, fmt.Errorf("LoadGemma4VisionFromHF: %w", lerr)
	}
	return m, nil
}
