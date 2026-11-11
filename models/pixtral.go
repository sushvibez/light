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

// PixtralConfig is a Mistral3 checkpoint's vision side: the Pixtral
// encoder's vision_config plus the projector settings.
type PixtralConfig struct {
	HiddenSize       int     `json:"hidden_size"`
	IntermediateSize int     `json:"intermediate_size"`
	NumLayers        int     `json:"num_hidden_layers"`
	NumHeads         int     `json:"num_attention_heads"`
	HeadDim          int     `json:"head_dim"`
	PatchSize        int     `json:"patch_size"`
	NumChannels      int     `json:"num_channels"`
	ImageSize        int     `json:"image_size"` // longest image edge
	RopeTheta        float64 `json:"-"`

	MergeSize int     // spatial_merge_size
	TextDim   int     // language model hidden size
	TextEps   float64 // projector RMSNorm eps (the text config's)
}

// Pixtral is the Pixtral vision encoder (Mistral 3 / Ministral 3): patch
// convolution, RMSNorm, pre-RMSNorm blocks with 2D RoPE and SiLU-gated
// MLPs, full attention inside each image; then Mistral3's projector
// (RMSNorm, a learned merge of each 2 × 2 patch block, GELU MLP to the LM
// width).
type Pixtral struct {
	Cfg       PixtralConfig
	PatchConv *nn.Linear // Conv2d with kernel = stride, over (C, P, P) patches
	LnPre     *nn.RMSNorm
	Layers    []*pixtralLayer
	ProjNorm  *nn.RMSNorm
	Merge     *nn.Linear // merging_layer (4·D → D)
	Lin1      *nn.Linear
	Lin2      *nn.Linear
}

type pixtralLayer struct {
	AttnNorm, FFNNorm *nn.RMSNorm
	Q, K, V, O        *nn.Linear
	Gate, Up, Down    *nn.Linear
}

// pixtralRoPE is PixtralVisionRotaryEmbedding: inverse frequencies over
// the head dim, the even ones for the row and the odd ones for the column,
// [row·f_even | col·f_odd] twice; cos / sin as (N, 1, Dh).
func (m *Pixtral) pixtralRoPE(rows, cols []int, dev device.Device) (*tensor.Tensor, *tensor.Tensor, error) {
	Dh := m.Cfg.HeadDim
	half := Dh / 2
	inv := make([]float32, half)
	for j := range half {
		inv[j] = float32(1 / math.Pow(m.Cfg.RopeTheta, float64(2*j)/float64(Dh)))
	}
	q := half / 2
	n := len(rows)
	cs, sn := make([]float32, n*Dh), make([]float32, n*Dh)
	for p := range n {
		for j := range q {
			for k, a := range []float32{float32(rows[p]) * inv[2*j], float32(cols[p]) * inv[2*j+1]} {
				c, s := float32(math.Cos(float64(a))), float32(math.Sin(float64(a)))
				for _, off := range []int{0, half} {
					cs[p*Dh+off+k*q+j], sn[p*Dh+off+k*q+j] = c, s
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

// PixtralImage is one preprocessed image: (C, H, W) normalised pixels with
// H and W multiples of the patch size.
type PixtralImage struct {
	Pixels []float32
	H, W   int
}

// Forward encodes images and projects them to LM tokens: per image,
// (H/P/merge) · (W/P/merge) rows of width TextDim, concatenated in order.
func (m *Pixtral) Forward(images []PixtralImage, dev device.Device) (*tensor.Tensor, error) {
	P, C, D := m.Cfg.PatchSize, m.Cfg.NumChannels, m.Cfg.HiddenSize
	var outs []*tensor.Tensor
	for _, im := range images {
		gh, gw := im.H/P, im.W/P
		n := gh * gw
		patches := make([]float32, n*C*P*P)
		rows, cols := make([]int, n), make([]int, n)
		for r := range gh {
			for c := range gw {
				i := r*gw + c
				rows[i], cols[i] = r, c
				for ch := range C {
					for y := range P {
						for x := range P {
							patches[((i*C+ch)*P+y)*P+x] = im.Pixels[(ch*im.H+r*P+y)*im.W+c*P+x]
						}
					}
				}
			}
		}
		pt, err := tensor.New(patches, []int{n, C * P * P}, dev)
		if err != nil {
			return nil, err
		}
		h, err := m.PatchConv.Forward(pt)
		if err != nil {
			return nil, err
		}
		if h, err = m.LnPre.Forward(h); err != nil {
			return nil, err
		}
		cos, sin, err := m.pixtralRoPE(rows, cols, dev)
		if err != nil {
			return nil, err
		}
		for li, l := range m.Layers {
			if h, err = m.layer(l, h, cos, sin); err != nil {
				return nil, fmt.Errorf("pixtral layer %d: %w", li, err)
			}
		}
		o, err := m.project(h, gh, gw)
		if err != nil {
			return nil, err
		}
		outs = append(outs, o)
		_ = D
	}
	if len(outs) == 1 {
		return outs[0], nil
	}
	return ops.Concat(outs, 0)
}

func (m *Pixtral) layer(l *pixtralLayer, h, cos, sin *tensor.Tensor) (*tensor.Tensor, error) {
	N, H, Dh := h.Dim(0), m.Cfg.NumHeads, m.Cfg.HeadDim
	n, err := l.AttnNorm.Forward(h)
	if err != nil {
		return nil, err
	}
	proj := func(lin *nn.Linear, rope bool) (*tensor.Tensor, error) {
		y, err := lin.Forward(n)
		if err != nil {
			return nil, err
		}
		if y, err = y.Reshape(N, H, Dh); err != nil {
			return nil, err
		}
		if rope {
			if y, err = rotateHalfLast(y, cos, sin); err != nil {
				return nil, err
			}
		}
		if y, err = autograd.Permute(y, 1, 0, 2); err != nil {
			return nil, err
		}
		return autograd.Reshape(y, 1, H, N, Dh)
	}
	q, err := proj(l.Q, true)
	if err != nil {
		return nil, err
	}
	k, err := proj(l.K, true)
	if err != nil {
		return nil, err
	}
	v, err := proj(l.V, false)
	if err != nil {
		return nil, err
	}
	a, err := nn.SDPA(q, k, v, nil)
	if err != nil {
		return nil, err
	}
	if a, err = autograd.Permute(a, 0, 2, 1, 3); err != nil {
		return nil, err
	}
	if a, err = a.Contiguous(); err != nil {
		return nil, err
	}
	if a, err = a.Reshape(N, H*Dh); err != nil {
		return nil, err
	}
	if a, err = l.O.Forward(a); err != nil {
		return nil, err
	}
	if h, err = ops.Add(h, a); err != nil {
		return nil, err
	}
	if n, err = l.FFNNorm.Forward(h); err != nil {
		return nil, err
	}
	g, err := l.Gate.Forward(n)
	if err != nil {
		return nil, err
	}
	if g, err = ops.SiLU(g); err != nil {
		return nil, err
	}
	u, err := l.Up.Forward(n)
	if err != nil {
		return nil, err
	}
	if g, err = ops.Mul(g, u); err != nil {
		return nil, err
	}
	f, err := l.Down.Forward(g)
	if err != nil {
		return nil, err
	}
	return ops.Add(h, f)
}

// project is Mistral3MultiModalProjector on one image's (gh·gw, D) states:
// RMSNorm, each merge × merge block flattened channel-major (unfold), the
// merging layer, linear_1, GELU, linear_2.
func (m *Pixtral) project(h *tensor.Tensor, gh, gw int) (*tensor.Tensor, error) {
	D, M := m.Cfg.HiddenSize, m.Cfg.MergeSize
	n, err := m.ProjNorm.Forward(h)
	if err != nil {
		return nil, err
	}
	// (gh, gw, D) → (gh/M, M, gw/M, M, D) → (gh/M, gw/M, D, M, M) → rows of D·M·M.
	if n, err = n.Reshape(gh/M, M, gw/M, M, D); err != nil {
		return nil, err
	}
	if n, err = n.Permute(0, 2, 4, 1, 3); err != nil {
		return nil, err
	}
	if n, err = n.Contiguous(); err != nil {
		return nil, err
	}
	if n, err = n.Reshape((gh/M)*(gw/M), D*M*M); err != nil {
		return nil, err
	}
	if n, err = m.Merge.Forward(n); err != nil {
		return nil, err
	}
	if n, err = m.Lin1.Forward(n); err != nil {
		return nil, err
	}
	if n, err = ops.GeluErf(n); err != nil {
		return nil, err
	}
	return m.Lin2.Forward(n)
}

// PixtralPreprocess resizes an image as PixtralImageProcessor does (fit
// within the longest edge, then each side rounded up to a multiple of
// patch·merge; Pillow bicubic) and normalises with CLIP's mean / std.
func (m *Pixtral) PixtralPreprocess(img image.Image) (PixtralImage, error) {
	rgb, w, h := rgbBytes(img)
	unit := m.Cfg.PatchSize * m.Cfg.MergeSize
	nh, nw := h, w
	if r := math.Max(float64(h)/float64(m.Cfg.ImageSize), float64(w)/float64(m.Cfg.ImageSize)); r > 1 {
		nh, nw = int(math.Floor(float64(h)/r)), int(math.Floor(float64(w)/r))
	}
	nh, nw = ((nh-1)/unit+1)*unit, ((nw-1)/unit+1)*unit
	if nh != h || nw != w {
		rgb = pilResizeRGB(rgb, w, h, nw, nh)
	}
	mean := [3]float32{0.48145466, 0.4578275, 0.40821073}
	std := [3]float32{0.26862954, 0.26130258, 0.27577711}
	px := make([]float32, 3*nh*nw)
	for c := range 3 {
		m255, s255 := mean[c]*255, std[c]*255
		for i := range nh * nw {
			px[c*nh*nw+i] = (float32(rgb[i*3+c]) - m255) / s255
		}
	}
	return PixtralImage{Pixels: px, H: nh, W: nw}, nil
}

// PixtralTokens is the placeholder layout PixtralProcessor writes for an
// image: per merged row, cols × [IMG] then [IMG_BREAK], the last row ending
// in [IMG_END] instead.
func (m *Pixtral) PixtralTokens(im PixtralImage, img, brk, end int64) []int64 {
	unit := m.Cfg.PatchSize * m.Cfg.MergeSize
	rows, cols := im.H/unit, im.W/unit
	var out []int64
	for r := range rows {
		for range cols {
			out = append(out, img)
		}
		if r == rows-1 {
			out = append(out, end)
		} else {
			out = append(out, brk)
		}
	}
	return out
}

// LoadPixtralFromHF loads the vision tower and projector of a Mistral3
// checkpoint (vision_tower.*, multi_modal_projector.*) at F32 on dev.
func LoadPixtralFromHF(repo string, dev device.Device) (*Pixtral, error) {
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
			PixtralConfig
			Rope struct {
				Theta float64 `json:"rope_theta"`
			} `json:"rope_parameters"`
			HiddenAct string `json:"hidden_act"`
		} `json:"vision_config"`
		Text struct {
			HiddenSize int     `json:"hidden_size"`
			Eps        float64 `json:"rms_norm_eps"`
		} `json:"text_config"`
		Merge        int    `json:"spatial_merge_size"`
		ProjAct      string `json:"projector_hidden_act"`
		ProjBias     bool   `json:"multimodal_projector_bias"`
		FeatureLayer any    `json:"vision_feature_layer"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, err
	}
	if c.Vision == nil {
		return nil, fmt.Errorf("%s: no vision_config", repo)
	}
	if c.Vision.HiddenAct != "silu" || c.ProjAct != "gelu" || c.ProjBias {
		return nil, fmt.Errorf("pixtral: hidden_act %q / projector %q (bias %v) not supported", c.Vision.HiddenAct, c.ProjAct, c.ProjBias)
	}
	if fl, ok := c.FeatureLayer.(float64); !ok || fl != -1 {
		return nil, fmt.Errorf("pixtral: vision_feature_layer %v not supported (-1)", c.FeatureLayer)
	}
	cfg := c.Vision.PixtralConfig
	cfg.RopeTheta, cfg.MergeSize, cfg.TextDim, cfg.TextEps = c.Vision.Rope.Theta, c.Merge, c.Text.HiddenSize, c.Text.Eps
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 10000
	}
	if cfg.HeadDim == 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads
	}
	D, P, Cn := cfg.HiddenSize, cfg.PatchSize, cfg.NumChannels
	var lerr error
	lin := func(in, out int) *nn.Linear {
		if lerr != nil {
			return nil
		}
		var l *nn.Linear
		l, lerr = nn.NewLinearUninit(in, out, false, dtype.F32, dev)
		return l
	}
	rms := func(n int, eps float64) *nn.RMSNorm {
		if lerr != nil {
			return nil
		}
		var r *nn.RMSNorm
		r, lerr = nn.NewRMSNormWithDType(n, eps, dtype.F32, dev)
		return r
	}
	m := &Pixtral{Cfg: cfg, PatchConv: lin(Cn*P*P, D), LnPre: rms(D, 1e-5),
		ProjNorm: rms(D, cfg.TextEps), Merge: lin(D*cfg.MergeSize*cfg.MergeSize, D),
		Lin1: lin(D, cfg.TextDim), Lin2: lin(cfg.TextDim, cfg.TextDim)}
	for range cfg.NumLayers {
		m.Layers = append(m.Layers, &pixtralLayer{AttnNorm: rms(D, 1e-5), FFNNorm: rms(D, 1e-5),
			Q: lin(D, D), K: lin(D, D), V: lin(D, D), O: lin(D, D),
			Gate: lin(D, cfg.IntermediateSize), Up: lin(D, cfg.IntermediateSize), Down: lin(cfg.IntermediateSize, D)})
	}
	if lerr != nil {
		return nil, lerr
	}
	hf, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, err
	}
	if pc, ok := hf["vision_tower.patch_conv.weight"]; ok {
		if hf["vision_tower.patch_conv.weight"], err = pc.Reshape(D, Cn*P*P); err != nil {
			return nil, err
		}
	}
	cp := func(dst *tensor.Tensor, name string) {
		if lerr == nil {
			lerr = copyExactCast(dst, hf, name)
		}
	}
	cp(m.PatchConv.W, "vision_tower.patch_conv.weight")
	cp(m.LnPre.Weight, "vision_tower.ln_pre.weight")
	for i, l := range m.Layers {
		p := fmt.Sprintf("vision_tower.transformer.layers.%d.", i)
		cp(l.AttnNorm.Weight, p+"attention_norm.weight")
		cp(l.FFNNorm.Weight, p+"ffn_norm.weight")
		cp(l.Q.W, p+"attention.q_proj.weight")
		cp(l.K.W, p+"attention.k_proj.weight")
		cp(l.V.W, p+"attention.v_proj.weight")
		cp(l.O.W, p+"attention.o_proj.weight")
		cp(l.Gate.W, p+"feed_forward.gate_proj.weight")
		cp(l.Up.W, p+"feed_forward.up_proj.weight")
		cp(l.Down.W, p+"feed_forward.down_proj.weight")
	}
	cp(m.ProjNorm.Weight, "multi_modal_projector.norm.weight")
	cp(m.Merge.W, "multi_modal_projector.patch_merger.merging_layer.weight")
	cp(m.Lin1.W, "multi_modal_projector.linear_1.weight")
	cp(m.Lin2.W, "multi_modal_projector.linear_2.weight")
	if lerr != nil {
		return nil, fmt.Errorf("LoadPixtralFromHF: %w", lerr)
	}
	return m, nil
}
