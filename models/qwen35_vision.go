package models

import (
	"encoding/json"
	"fmt"
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

// Qwen35VisionConfig is the vision_config of a Qwen3.5 / Qwen3-VL style
// checkpoint.
type Qwen35VisionConfig struct {
	Depth             int    `json:"depth"`
	HiddenSize        int    `json:"hidden_size"`
	IntermediateSize  int    `json:"intermediate_size"`
	NumHeads          int    `json:"num_heads"`
	OutHiddenSize     int    `json:"out_hidden_size"`
	PatchSize         int    `json:"patch_size"`
	TemporalPatchSize int    `json:"temporal_patch_size"`
	SpatialMergeSize  int    `json:"spatial_merge_size"`
	InChannels        int    `json:"in_channels"`
	NumPosEmbeddings  int    `json:"num_position_embeddings"`
	HiddenAct         string `json:"hidden_act"`
	DeepstackIndexes  []int  `json:"deepstack_visual_indexes"`
}

// Qwen35Vision is the Qwen3.5 vision tower (Qwen3-VL ViT without deepstack):
// a linear patch embedding over temporal × spatial patches, a learned 2D
// position table bilinearly resampled to each image's grid, pre-LayerNorm
// blocks with 2D axial RoPE attention inside each image, and a merger that
// concatenates each 2 × 2 patch block into one LM-width token.
type Qwen35Vision struct {
	Cfg        Qwen35VisionConfig
	PatchEmbed *nn.Linear // (C·T·P·P → hidden), the Conv3d with kernel = stride
	PosEmbed   *tensor.Tensor
	Blocks     []*qwen35VisionBlock
	MergeNorm  *nn.LayerNorm
	MergeFC1   *nn.Linear
	MergeFC2   *nn.Linear
}

type qwen35VisionBlock struct {
	Norm1, Norm2 *nn.LayerNorm
	QKV, Proj    *nn.Linear
	FC1, FC2     *nn.Linear
}

// patchDim is the flattened patch width (C·T·P·P).
func (c Qwen35VisionConfig) patchDim() int {
	return c.InChannels * c.TemporalPatchSize * c.PatchSize * c.PatchSize
}

func newQwen35Vision(cfg Qwen35VisionConfig, dev device.Device) (*Qwen35Vision, error) {
	if len(cfg.DeepstackIndexes) > 0 {
		return nil, fmt.Errorf("qwen3.5 vision: deepstack layers %v not supported", cfg.DeepstackIndexes)
	}
	if cfg.HiddenAct != "gelu_pytorch_tanh" {
		return nil, fmt.Errorf("qwen3.5 vision: hidden_act %q not supported", cfg.HiddenAct)
	}
	D := cfg.HiddenSize
	var err error
	lin := func(in, out int) *nn.Linear {
		if err != nil {
			return nil
		}
		var l *nn.Linear
		l, err = nn.NewLinearUninit(in, out, true, dtype.F32, dev)
		return l
	}
	ln := func(n int) *nn.LayerNorm {
		if err != nil {
			return nil
		}
		var l *nn.LayerNorm
		l, err = nn.NewLayerNorm(n, 1e-6, dev)
		return l
	}
	m := &Qwen35Vision{Cfg: cfg, PatchEmbed: lin(cfg.patchDim(), D)}
	for range cfg.Depth {
		m.Blocks = append(m.Blocks, &qwen35VisionBlock{Norm1: ln(D), Norm2: ln(D),
			QKV: lin(D, 3*D), Proj: lin(D, D), FC1: lin(D, cfg.IntermediateSize), FC2: lin(cfg.IntermediateSize, D)})
	}
	mergeIn := D * cfg.SpatialMergeSize * cfg.SpatialMergeSize
	m.MergeNorm, m.MergeFC1, m.MergeFC2 = ln(D), lin(mergeIn, mergeIn), lin(mergeIn, cfg.OutHiddenSize)
	if err != nil {
		return nil, err
	}
	if m.PosEmbed, err = tensor.Zeros(dtype.F32, []int{cfg.NumPosEmbeddings, D}, dev); err != nil {
		return nil, err
	}
	return m, nil
}

// visionRowCol is each patch's (row, col) in an h × w grid, in the
// processor's order: merge × merge blocks row-major, patches row-major
// inside a block.
func visionRowCol(h, w, merge int) (rows, cols []int) {
	for br := range h / merge {
		for bc := range w / merge {
			for ir := range merge {
				for ic := range merge {
					rows = append(rows, br*merge+ir)
					cols = append(cols, bc*merge+ic)
				}
			}
		}
	}
	return
}

// posEmbedFor resamples the learned (side × side) position table to an
// h × w grid (bilinear, align_corners, border taps) in patch order:
// get_vision_interpolation_indices_and_weights.
func (m *Qwen35Vision) posEmbedFor(rows, cols []int, h, w int, dev device.Device) (*tensor.Tensor, error) {
	side := int(math.Sqrt(float64(m.Cfg.NumPosEmbeddings)))
	taps := func(i, size int) ([2]int, [2]float32) {
		den := float32(max(size-1, 1))
		src := float32(i) * float32(side-1) / den
		f := float32(math.Floor(float64(src)))
		var t [2]int
		var wt [2]float32
		for o := range 2 {
			t[o] = min(max(int(f)+o, 0), side-1)
			wt[o] = max(1-float32(math.Abs(float64(src-f-float32(o)))), 0)
		}
		return t, wt
	}
	n := len(rows)
	idx := make([]int64, 4*n)
	wts := make([]float32, 4*n)
	for p := range n {
		ht, hw := taps(rows[p], h)
		wtp, ww := taps(cols[p], w)
		for a := range 2 {
			for b := range 2 {
				idx[4*p+2*a+b] = int64(ht[a]*side + wtp[b])
				wts[4*p+2*a+b] = hw[a] * ww[b]
			}
		}
	}
	it, err := tensor.New(idx, []int{4 * n}, dev)
	if err != nil {
		return nil, err
	}
	g, err := ops.IndexSelect(m.PosEmbed, it) // (4n, D)
	if err != nil {
		return nil, err
	}
	D := m.Cfg.HiddenSize
	if g, err = g.Reshape(n, 4, D); err != nil {
		return nil, err
	}
	wt, err := tensor.New(wts, []int{n, 4, 1}, dev)
	if err != nil {
		return nil, err
	}
	if g, err = ops.Mul(g, wt); err != nil {
		return nil, err
	}
	return ops.Sum(g, ops.ReduceOptions{Axes: []int{1}})
}

// visionRoPE is the axial 2D rope: per patch, [row·f | col·f | row·f |
// col·f] with f = 10000^(−2j/(Dh/2)), j < Dh/4; cos and sin as (n, 1, Dh).
func (m *Qwen35Vision) visionRoPE(rows, cols []int, dev device.Device) (*tensor.Tensor, *tensor.Tensor, error) {
	Dh := m.Cfg.HiddenSize / m.Cfg.NumHeads
	half := Dh / 2
	nf := half / 2
	inv := make([]float32, nf)
	for j := range nf {
		inv[j] = float32(1 / math.Pow(10000, float64(2*j)/float64(half)))
	}
	n := len(rows)
	cs, sn := make([]float32, n*Dh), make([]float32, n*Dh)
	for p := range n {
		for j := range nf {
			for k, pos := range []int{rows[p], cols[p]} {
				a := float32(pos) * inv[j]
				c, s := float32(math.Cos(float64(a))), float32(math.Sin(float64(a)))
				for _, off := range []int{0, half} {
					cs[p*Dh+off+k*nf+j], sn[p*Dh+off+k*nf+j] = c, s
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

// rotateHalfRoPE is x·cos + rotate_half(x)·sin for x (n, H, Dh).
func rotateHalfRoPE(x, cos, sin *tensor.Tensor) (*tensor.Tensor, error) {
	return rotateHalfLast(x, cos, sin)
}

// rotateHalfLast is x·cos + rotate_half(x)·sin over x's last axis, with
// cos / sin broadcast against x.
func rotateHalfLast(x, cos, sin *tensor.Tensor) (*tensor.Tensor, error) {
	ax := x.Rank() - 1
	Dh := x.Dim(ax)
	x1, err := x.Narrow(ax, 0, Dh/2)
	if err != nil {
		return nil, err
	}
	x2, err := x.Narrow(ax, Dh/2, Dh/2)
	if err != nil {
		return nil, err
	}
	n2, err := ops.Neg(x2)
	if err != nil {
		return nil, err
	}
	rot, err := ops.Concat([]*tensor.Tensor{n2, x1}, ax)
	if err != nil {
		return nil, err
	}
	a, err := ops.Mul(x, cos)
	if err != nil {
		return nil, err
	}
	b, err := ops.Mul(rot, sin)
	if err != nil {
		return nil, err
	}
	return ops.Add(a, b)
}

// Forward runs the tower on preprocessed patches (N, C·T·P·P) for images
// with grids grid[i] = (t, h, w) in patches, returning the merged tokens
// (N / merge², out_hidden_size) in the same order.
func (m *Qwen35Vision) Forward(pixels *tensor.Tensor, grid [][3]int) (*tensor.Tensor, error) {
	h, err := m.encode(pixels, grid, len(m.Blocks))
	if err != nil {
		return nil, err
	}
	return m.merge(h)
}

// encode runs the patch embedding and the first nBlocks blocks.
func (m *Qwen35Vision) encode(pixels *tensor.Tensor, grid [][3]int, nBlocks int) (*tensor.Tensor, error) {
	dev := pixels.Device()
	H := m.Cfg.NumHeads
	D := m.Cfg.HiddenSize
	Dh := D / H
	merge := m.Cfg.SpatialMergeSize
	var rows, cols []int
	var segs []int // tokens per attention segment (one per frame)
	var pos []*tensor.Tensor
	for _, g := range grid {
		r, c := visionRowCol(g[1], g[2], merge)
		p, err := m.posEmbedFor(r, c, g[1], g[2], dev)
		if err != nil {
			return nil, err
		}
		for range g[0] {
			rows, cols = append(rows, r...), append(cols, c...)
			segs = append(segs, g[1]*g[2])
			pos = append(pos, p)
		}
	}
	if len(rows) != pixels.Dim(0) {
		return nil, fmt.Errorf("qwen3.5 vision: grids cover %d patches, pixels have %d", len(rows), pixels.Dim(0))
	}
	h, err := m.PatchEmbed.Forward(pixels)
	if err != nil {
		return nil, err
	}
	pe := pos[0]
	if len(pos) > 1 {
		if pe, err = ops.Concat(pos, 0); err != nil {
			return nil, err
		}
	}
	if h, err = ops.Add(h, pe); err != nil {
		return nil, err
	}
	cos, sin, err := m.visionRoPE(rows, cols, dev)
	if err != nil {
		return nil, err
	}
	N := len(rows)
	for i, b := range m.Blocks[:nBlocks] {
		n, err := b.Norm1.Forward(h)
		if err != nil {
			return nil, err
		}
		qkv, err := b.QKV.Forward(n)
		if err != nil {
			return nil, err
		}
		if qkv, err = qkv.Reshape(N, 3, H, Dh); err != nil {
			return nil, err
		}
		var q, k, v *tensor.Tensor
		for j, dst := range []**tensor.Tensor{&q, &k, &v} {
			t, err := qkv.Narrow(1, j, 1)
			if err != nil {
				return nil, err
			}
			if t, err = t.Contiguous(); err != nil {
				return nil, err
			}
			if *dst, err = t.Reshape(N, H, Dh); err != nil {
				return nil, err
			}
		}
		if q, err = rotateHalfRoPE(q, cos, sin); err != nil {
			return nil, err
		}
		if k, err = rotateHalfRoPE(k, cos, sin); err != nil {
			return nil, err
		}
		// Full attention inside each frame.
		var outs []*tensor.Tensor
		off := 0
		for _, L := range segs {
			sl := func(t *tensor.Tensor) (*tensor.Tensor, error) {
				s, err := t.Narrow(0, off, L)
				if err != nil {
					return nil, err
				}
				if s, err = autograd.Permute(s, 1, 0, 2); err != nil { // (H, L, Dh)
					return nil, err
				}
				return autograd.Reshape(s, 1, H, L, Dh)
			}
			qs, err := sl(q)
			if err != nil {
				return nil, err
			}
			ks, err := sl(k)
			if err != nil {
				return nil, err
			}
			vs, err := sl(v)
			if err != nil {
				return nil, err
			}
			o, err := nn.SDPA(qs, ks, vs, nil) // (1, H, L, Dh)
			if err != nil {
				return nil, fmt.Errorf("qwen3.5 vision block %d: %w", i, err)
			}
			if o, err = autograd.Permute(o, 0, 2, 1, 3); err != nil {
				return nil, err
			}
			if o, err = o.Contiguous(); err != nil {
				return nil, err
			}
			if o, err = o.Reshape(L, D); err != nil {
				return nil, err
			}
			outs = append(outs, o)
			off += L
		}
		a := outs[0]
		if len(outs) > 1 {
			if a, err = ops.Concat(outs, 0); err != nil {
				return nil, err
			}
		}
		if a, err = b.Proj.Forward(a); err != nil {
			return nil, err
		}
		if h, err = ops.Add(h, a); err != nil {
			return nil, err
		}
		if n, err = b.Norm2.Forward(h); err != nil {
			return nil, err
		}
		f, err := b.FC1.Forward(n)
		if err != nil {
			return nil, err
		}
		if f, err = ops.Gelu(f); err != nil { // gelu_pytorch_tanh
			return nil, err
		}
		if f, err = b.FC2.Forward(f); err != nil {
			return nil, err
		}
		if h, err = ops.Add(h, f); err != nil {
			return nil, err
		}
	}
	return h, nil
}

// merge is the patch merger: LayerNorm, merge² consecutive patches → one
// row, MLP (exact GELU).
func (m *Qwen35Vision) merge(h *tensor.Tensor) (*tensor.Tensor, error) {
	N, D, merge := h.Dim(0), m.Cfg.HiddenSize, m.Cfg.SpatialMergeSize
	n, err := m.MergeNorm.Forward(h)
	if err != nil {
		return nil, err
	}
	if n, err = n.Reshape(N/(merge*merge), D*merge*merge); err != nil {
		return nil, err
	}
	if n, err = m.MergeFC1.Forward(n); err != nil {
		return nil, err
	}
	if n, err = ops.GeluErf(n); err != nil {
		return nil, err
	}
	return m.MergeFC2.Forward(n)
}

// LoadQwen35VisionFromHF loads the vision tower of a Qwen3.5 checkpoint
// (model.visual.*) at F32 on dev.
func LoadQwen35VisionFromHF(repo string, dev device.Device) (*Qwen35Vision, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return nil, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c struct {
		Vision *Qwen35VisionConfig `json:"vision_config"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, err
	}
	if c.Vision == nil {
		return nil, fmt.Errorf("%s: no vision_config", repo)
	}
	if c.Vision.InChannels == 0 {
		c.Vision.InChannels = 3
	}
	m, err := newQwen35Vision(*c.Vision, dev)
	if err != nil {
		return nil, err
	}
	hf, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, err
	}
	const p = "model.visual."
	pw, ok := hf[p+"patch_embed.proj.weight"]
	if !ok {
		return nil, fmt.Errorf("%s: no %spatch_embed.proj.weight", repo, p)
	}
	if pw, err = pw.Reshape(pw.Dim(0), m.Cfg.patchDim()); err != nil { // Conv3d kernel = stride → linear
		return nil, err
	}
	hf[p+"patch_embed.proj.weight"] = pw
	var lerr error
	lin := func(l *nn.Linear, name string) {
		if lerr == nil {
			lerr = copyExactCast(l.W, hf, p+name+".weight")
		}
		if lerr == nil {
			lerr = copyExactCast(l.B, hf, p+name+".bias")
		}
	}
	ln := func(l *nn.LayerNorm, name string) {
		if lerr == nil {
			lerr = copyExactCast(l.Gamma, hf, p+name+".weight")
		}
		if lerr == nil {
			lerr = copyExactCast(l.Beta, hf, p+name+".bias")
		}
	}
	lin(m.PatchEmbed, "patch_embed.proj")
	if lerr == nil {
		lerr = copyExactCast(m.PosEmbed, hf, p+"pos_embed.weight")
	}
	for i, b := range m.Blocks {
		bp := fmt.Sprintf("blocks.%d.", i)
		ln(b.Norm1, bp+"norm1")
		ln(b.Norm2, bp+"norm2")
		lin(b.QKV, bp+"attn.qkv")
		lin(b.Proj, bp+"attn.proj")
		lin(b.FC1, bp+"mlp.linear_fc1")
		lin(b.FC2, bp+"mlp.linear_fc2")
	}
	ln(m.MergeNorm, "merger.norm")
	lin(m.MergeFC1, "merger.linear_fc1")
	lin(m.MergeFC2, "merger.linear_fc2")
	if lerr != nil {
		return nil, fmt.Errorf("LoadQwen35VisionFromHF: %w", lerr)
	}
	return m, nil
}
