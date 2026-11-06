package models

import (
	"fmt"
	"math"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// MimiConfig describes Kyutai's Mimi neural audio codec (transformers
// modeling_mimi.py; kyutai/mimi): a causal SEANet encoder, an 8-layer
// sliding-window transformer, a stride-2 downsample to 12.5 Hz, a split
// residual vector quantizer (1 semantic + 31 acoustic codebooks of 2048
// entries), and the mirror image on the decoder side.
type MimiConfig struct {
	Channels           int
	NumFilters         int
	HiddenSize         int
	KernelSize         int
	LastKernelSize     int
	ResidualKernelSize int
	Compress           int
	DilationGrowth     int
	NumResidualLayers  int
	UpsamplingRatios   []int
	NumLayers          int // transformer layers
	NumHeads           int
	HeadDim            int
	IntermediateSize   int
	NormEps            float64
	RopeBase           float64
	SlidingWindow      int
	CodebookSize       int
	CodebookDim        int
	NumQuantizers      int
	NumSemantic        int
	LatentStride       int // 0 = no down/upsample between the transformer and the quantizer
	UpsampleGroups     int
	SampleRate         int
	FrameRate          float64
	LayerScaleInit     float64
}

// DefaultMimiConfig is kyutai/mimi.
func DefaultMimiConfig() MimiConfig {
	return MimiConfig{
		Channels: 1, NumFilters: 64, HiddenSize: 512, KernelSize: 7, LastKernelSize: 3, ResidualKernelSize: 3,
		Compress: 2, DilationGrowth: 2, NumResidualLayers: 1, UpsamplingRatios: []int{8, 6, 5, 4},
		NumLayers: 8, NumHeads: 8, HeadDim: 64, IntermediateSize: 2048, NormEps: 1e-5, RopeBase: 1e4,
		SlidingWindow: 250, CodebookSize: 2048, CodebookDim: 256, NumQuantizers: 32, NumSemantic: 1,
		LatentStride: 2, UpsampleGroups: 512, SampleRate: 24000, FrameRate: 12.5, LayerScaleInit: 0.01,
	}
}

// HopLength is the number of audio samples per codec frame (1920 at 12.5 Hz).
func (cfg MimiConfig) HopLength() int {
	h := 1
	for _, r := range cfg.UpsamplingRatios {
		h *= r
	}
	if cfg.LatentStride > 0 {
		h *= cfg.LatentStride
	}
	return h
}

// mimiConv is MimiConv1d: a causal convolution padded on the left by
// (kernel−1)·dilation+1−stride and on the right by whatever makes the
// input an exact number of strides, with zero or edge-replicate padding.
type mimiConv struct {
	W, B      *tensor.Tensor // (Cout, Cin, k)
	stride    int
	dilation  int
	replicate bool
}

func (c *mimiConv) kernelEff() int { return (c.W.Dim(2)-1)*c.dilation + 1 }

// mimiConvState is a causal conv's streaming memory: the last kernel−stride
// input samples (zeros, or the first sample when the conv pads by
// replication, before anything was seen).
type mimiConvState struct {
	prev *tensor.Tensor
}

// forward is the full-sequence conv (st == nil) or one streaming step: the
// chunk, whose length must be a multiple of the stride, is prefixed with the
// remembered samples so every output frame is complete and identical to the
// one the full-sequence conv would produce at the same position.
func (c *mimiConv) forward(x *tensor.Tensor, st *mimiConvState) (*tensor.Tensor, error) {
	L := x.Dim(2)
	k, s := c.kernelEff(), c.stride
	padTotal := k - s
	if c.dilation != 1 {
		return nil, fmt.Errorf("mimi: dilation %d not supported", c.dilation)
	}
	if st != nil {
		if L%s != 0 {
			return nil, fmt.Errorf("mimi: streaming conv chunk %d is not a multiple of stride %d", L, s)
		}
		if padTotal > 0 {
			var err error
			if st.prev == nil {
				if c.replicate {
					if st.prev, err = replicatePad(x, padTotal, 0); err != nil {
						return nil, err
					}
					if st.prev, err = st.prev.Narrow(2, 0, padTotal); err != nil {
						return nil, err
					}
				} else if st.prev, err = tensor.Zeros(x.DType(), []int{x.Dim(0), x.Dim(1), padTotal}, x.Device()); err != nil {
					return nil, err
				}
			}
			if x, err = ops.Concat([]*tensor.Tensor{st.prev, x}, 2); err != nil {
				return nil, err
			}
			if st.prev, err = x.Narrow(2, x.Dim(2)-padTotal, padTotal); err != nil {
				return nil, err
			}
			if st.prev, err = autograd.Contiguous(st.prev); err != nil {
				return nil, err
			}
		}
		return ops.Conv1d(x, c.W, c.B, s, 0)
	}
	nFrames := int(math.Ceil(float64(L-k+padTotal)/float64(s)+1)) - 1
	ideal := nFrames*s + k - padTotal
	extra := ideal - L
	var err error
	if c.replicate {
		x, err = replicatePad(x, padTotal, extra)
	} else {
		x, err = ops.Pad(x, 2, padTotal, extra)
	}
	if err != nil {
		return nil, err
	}
	return ops.Conv1d(x, c.W, c.B, s, 0)
}

// replicatePad pads the last axis by repeating the edge samples.
func replicatePad(x *tensor.Tensor, left, right int) (*tensor.Tensor, error) {
	parts := make([]*tensor.Tensor, 0, left+right+1)
	if left > 0 {
		first, err := x.Narrow(2, 0, 1)
		if err != nil {
			return nil, err
		}
		for range left {
			parts = append(parts, first)
		}
	}
	parts = append(parts, x)
	if right > 0 {
		last, err := x.Narrow(2, x.Dim(2)-1, 1)
		if err != nil {
			return nil, err
		}
		for range right {
			parts = append(parts, last)
		}
	}
	if len(parts) == 1 {
		return x, nil
	}
	return ops.Concat(parts, 2)
}

// mimiConvT is MimiConvTranspose1d (causal, trim_right_ratio 1): the
// transposed convolution's trailing kernel−stride samples are dropped.
type mimiConvT struct {
	W, B   *tensor.Tensor // (Cin, Cout/groups, k), (Cout,) or nil
	stride int
	groups int
}

// mimiConvTState is a transposed conv's streaming memory: the trailing
// kernel−stride outputs of the last chunk (bias removed), which overlap-add
// onto the head of the next chunk's output.
type mimiConvTState struct {
	partial *tensor.Tensor
}

// forward is the full-sequence transposed conv (st == nil, trailing
// kernel−stride samples dropped) or one streaming step producing exactly
// stride·T samples.
func (c *mimiConvT) forward(x *tensor.Tensor, st *mimiConvTState) (*tensor.Tensor, error) {
	y, err := ops.ConvTranspose1d(x, c.W, c.B, c.stride, 0, 0, 1, c.groups)
	if err != nil {
		return nil, err
	}
	trim := c.W.Dim(2) - c.stride
	if trim == 0 {
		return y, nil
	}
	if st != nil {
		n := y.Dim(2)
		head, err := y.Narrow(2, 0, trim)
		if err != nil {
			return nil, err
		}
		if st.partial != nil {
			if head, err = ops.Add(head, st.partial); err != nil {
				return nil, err
			}
		}
		tail, err := y.Narrow(2, trim, n-trim)
		if err != nil {
			return nil, err
		}
		if y, err = ops.Concat([]*tensor.Tensor{head, tail}, 2); err != nil {
			return nil, err
		}
		partial, err := y.Narrow(2, n-trim, trim)
		if err != nil {
			return nil, err
		}
		if c.B != nil {
			b, err := c.B.Reshape(1, c.B.Dim(0), 1)
			if err != nil {
				return nil, err
			}
			if partial, err = ops.Sub(partial, b); err != nil {
				return nil, err
			}
		}
		if st.partial, err = autograd.Contiguous(partial); err != nil {
			return nil, err
		}
	}
	y, err = y.Narrow(2, 0, y.Dim(2)-trim)
	if err != nil {
		return nil, err
	}
	return autograd.Contiguous(y)
}

// mimiStream holds every per-module streaming state of one direction
// (encoder or decoder): conv memories keyed by module and the transformer's
// KV cache.
type mimiStream struct {
	convs  map[*mimiConv]*mimiConvState
	convTs map[*mimiConvT]*mimiConvTState
	cache  *KVCache
}

func newMimiStream(layers int) *mimiStream {
	return &mimiStream{convs: map[*mimiConv]*mimiConvState{}, convTs: map[*mimiConvT]*mimiConvTState{}, cache: NewKVCache(layers)}
}

func (s *mimiStream) conv(c *mimiConv) *mimiConvState {
	if s == nil {
		return nil
	}
	st, ok := s.convs[c]
	if !ok {
		st = &mimiConvState{}
		s.convs[c] = st
	}
	return st
}

func (s *mimiStream) convT(c *mimiConvT) *mimiConvTState {
	if s == nil {
		return nil
	}
	st, ok := s.convTs[c]
	if !ok {
		st = &mimiConvTState{}
		s.convTs[c] = st
	}
	return st
}

func (s *mimiStream) kv() *KVCache {
	if s == nil {
		return nil
	}
	return s.cache
}

// mimiResnet: x + conv1x1(elu(conv3(elu(x)))).
type mimiResnet struct {
	Conv1, Conv2 *mimiConv
}

func (r *mimiResnet) forward(x *tensor.Tensor, s *mimiStream) (*tensor.Tensor, error) {
	h, err := ops.ELU(x, 1)
	if err != nil {
		return nil, err
	}
	if h, err = r.Conv1.forward(h, s.conv(r.Conv1)); err != nil {
		return nil, err
	}
	if h, err = ops.ELU(h, 1); err != nil {
		return nil, err
	}
	if h, err = r.Conv2.forward(h, s.conv(r.Conv2)); err != nil {
		return nil, err
	}
	return ops.Add(x, h)
}

// mimiSeanetLayer is one entry of the HF `layers` ModuleList (ELU entries
// hold nothing and are applied in order).
type mimiSeanetLayer struct {
	conv   *mimiConv
	convT  *mimiConvT
	resnet *mimiResnet
	elu    bool
}

func (l *mimiSeanetLayer) forward(x *tensor.Tensor, s *mimiStream) (*tensor.Tensor, error) {
	switch {
	case l.elu:
		return ops.ELU(x, 1)
	case l.conv != nil:
		return l.conv.forward(x, s.conv(l.conv))
	case l.convT != nil:
		return l.convT.forward(x, s.convT(l.convT))
	}
	return l.resnet.forward(x, s)
}

type mimiTransformerLayer struct {
	InNorm, PostAttnNorm       *nn.LayerNorm
	QProj, KProj, VProj, OProj *nn.Linear
	FC1, FC2                   *nn.Linear
	AttnScale, MLPScale        *tensor.Tensor // layer scale vectors (D,)
}

type mimiTransformer struct {
	Layers []*mimiTransformerLayer
}

type mimiCodebook struct {
	EmbedSum, ClusterUsage *tensor.Tensor // (K, D), (K,)
	embed                  *tensor.Tensor // (K, D) = EmbedSum / max(ClusterUsage, eps)
	embedT                 *tensor.Tensor // (D, K)
	embedSq                *tensor.Tensor // (1, K) squared norms
}

type mimiRVQ struct {
	InProj, OutProj *tensor.Tensor // (Dq, D, 1), (D, Dq, 1) 1×1 convs, nil when Dq == D
	Layers          []*mimiCodebook
}

// Mimi holds the codec. Weights come from LoadMimiFromHF.
type Mimi struct {
	Cfg        MimiConfig
	Encoder    []*mimiSeanetLayer
	EncoderTF  *mimiTransformer
	Downsample *mimiConv // nil without LatentStride
	Semantic   *mimiRVQ
	Acoustic   *mimiRVQ
	Upsample   *mimiConvT
	DecoderTF  *mimiTransformer
	Decoder    []*mimiSeanetLayer

	scale *tensor.Tensor
	tabs  gemma4Tables
}

func (m *Mimi) dev() device.Device { return m.Encoder[0].conv.W.Device() }

// NewMimi allocates an uninitialised codec.
func NewMimi(cfg MimiConfig, dev device.Device) (*Mimi, error) {
	m := &Mimi{Cfg: cfg}
	D := cfg.HiddenSize
	var err error
	conv := func(cin, cout, k, stride int, bias, replicate bool) (*mimiConv, error) {
		w, err := tensor.Zeros(dtype.F32, []int{cout, cin, k}, dev)
		if err != nil {
			return nil, err
		}
		c := &mimiConv{W: w, stride: stride, dilation: 1, replicate: replicate}
		if bias {
			if c.B, err = tensor.Zeros(dtype.F32, []int{cout}, dev); err != nil {
				return nil, err
			}
		}
		return c, nil
	}
	resnet := func(dim int) (*mimiResnet, error) {
		hidden := dim / cfg.Compress
		c1, err := conv(dim, hidden, cfg.ResidualKernelSize, 1, true, false)
		if err != nil {
			return nil, err
		}
		c2, err := conv(hidden, dim, 1, 1, true, false)
		if err != nil {
			return nil, err
		}
		return &mimiResnet{Conv1: c1, Conv2: c2}, nil
	}
	if cfg.NumResidualLayers != 1 {
		return nil, fmt.Errorf("mimi: num_residual_layers %d not supported (dilated residual blocks)", cfg.NumResidualLayers)
	}
	// Encoder SEANet.
	c0, err := conv(cfg.Channels, cfg.NumFilters, cfg.KernelSize, 1, true, false)
	if err != nil {
		return nil, err
	}
	m.Encoder = []*mimiSeanetLayer{{conv: c0}}
	scaling := 1
	for i := len(cfg.UpsamplingRatios) - 1; i >= 0; i-- {
		ratio := cfg.UpsamplingRatios[i]
		cur := scaling * cfg.NumFilters
		r, err := resnet(cur)
		if err != nil {
			return nil, err
		}
		c, err := conv(cur, cur*2, ratio*2, ratio, true, false)
		if err != nil {
			return nil, err
		}
		m.Encoder = append(m.Encoder, &mimiSeanetLayer{resnet: r}, &mimiSeanetLayer{elu: true}, &mimiSeanetLayer{conv: c})
		scaling *= 2
	}
	cl, err := conv(scaling*cfg.NumFilters, D, cfg.LastKernelSize, 1, true, false)
	if err != nil {
		return nil, err
	}
	m.Encoder = append(m.Encoder, &mimiSeanetLayer{elu: true}, &mimiSeanetLayer{conv: cl})
	// Transformers.
	tf := func() (*mimiTransformer, error) {
		t := &mimiTransformer{}
		for range cfg.NumLayers {
			l := &mimiTransformerLayer{}
			var errs []error
			push := func(e error) {
				if e != nil {
					errs = append(errs, e)
				}
			}
			l.InNorm, err = nn.NewLayerNorm(D, cfg.NormEps, dev)
			push(err)
			l.PostAttnNorm, err = nn.NewLayerNorm(D, cfg.NormEps, dev)
			push(err)
			l.QProj, err = nn.NewLinearUninit(D, cfg.NumHeads*cfg.HeadDim, false, dtype.F32, dev)
			push(err)
			l.KProj, err = nn.NewLinearUninit(D, cfg.NumHeads*cfg.HeadDim, false, dtype.F32, dev)
			push(err)
			l.VProj, err = nn.NewLinearUninit(D, cfg.NumHeads*cfg.HeadDim, false, dtype.F32, dev)
			push(err)
			l.OProj, err = nn.NewLinearUninit(cfg.NumHeads*cfg.HeadDim, D, false, dtype.F32, dev)
			push(err)
			l.FC1, err = nn.NewLinearUninit(D, cfg.IntermediateSize, false, dtype.F32, dev)
			push(err)
			l.FC2, err = nn.NewLinearUninit(cfg.IntermediateSize, D, false, dtype.F32, dev)
			push(err)
			l.AttnScale, err = tensor.Zeros(dtype.F32, []int{D}, dev)
			push(err)
			l.MLPScale, err = tensor.Zeros(dtype.F32, []int{D}, dev)
			push(err)
			if len(errs) > 0 {
				return nil, errs[0]
			}
			t.Layers = append(t.Layers, l)
		}
		return t, nil
	}
	if m.EncoderTF, err = tf(); err != nil {
		return nil, err
	}
	if m.DecoderTF, err = tf(); err != nil {
		return nil, err
	}
	if cfg.LatentStride > 1 {
		if m.Downsample, err = conv(D, D, 2*cfg.LatentStride, cfg.LatentStride, false, true); err != nil {
			return nil, err
		}
		w, err := tensor.Zeros(dtype.F32, []int{D, D / cfg.UpsampleGroups, 2 * cfg.LatentStride}, dev)
		if err != nil {
			return nil, err
		}
		m.Upsample = &mimiConvT{W: w, stride: cfg.LatentStride, groups: cfg.UpsampleGroups}
	}
	// Quantizers.
	rvq := func(n int) (*mimiRVQ, error) {
		q := &mimiRVQ{}
		if cfg.CodebookDim != D {
			if q.InProj, err = tensor.Zeros(dtype.F32, []int{cfg.CodebookDim, D, 1}, dev); err != nil {
				return nil, err
			}
			if q.OutProj, err = tensor.Zeros(dtype.F32, []int{D, cfg.CodebookDim, 1}, dev); err != nil {
				return nil, err
			}
		}
		for range n {
			cb := &mimiCodebook{}
			if cb.EmbedSum, err = tensor.Zeros(dtype.F32, []int{cfg.CodebookSize, cfg.CodebookDim}, dev); err != nil {
				return nil, err
			}
			if cb.ClusterUsage, err = tensor.Zeros(dtype.F32, []int{cfg.CodebookSize}, dev); err != nil {
				return nil, err
			}
			q.Layers = append(q.Layers, cb)
		}
		return q, nil
	}
	if m.Semantic, err = rvq(cfg.NumSemantic); err != nil {
		return nil, err
	}
	if m.Acoustic, err = rvq(cfg.NumQuantizers - cfg.NumSemantic); err != nil {
		return nil, err
	}
	// Decoder SEANet.
	scaling = 1 << len(cfg.UpsamplingRatios)
	d0, err := conv(D, scaling*cfg.NumFilters, cfg.KernelSize, 1, true, false)
	if err != nil {
		return nil, err
	}
	m.Decoder = []*mimiSeanetLayer{{conv: d0}}
	for _, ratio := range cfg.UpsamplingRatios {
		cur := scaling * cfg.NumFilters
		w, err := tensor.Zeros(dtype.F32, []int{cur, cur / 2, ratio * 2}, dev)
		if err != nil {
			return nil, err
		}
		b, err := tensor.Zeros(dtype.F32, []int{cur / 2}, dev)
		if err != nil {
			return nil, err
		}
		r, err := resnet(cur / 2)
		if err != nil {
			return nil, err
		}
		m.Decoder = append(m.Decoder, &mimiSeanetLayer{elu: true}, &mimiSeanetLayer{convT: &mimiConvT{W: w, B: b, stride: ratio, groups: 1}}, &mimiSeanetLayer{resnet: r})
		scaling /= 2
	}
	dl, err := conv(cfg.NumFilters, cfg.Channels, cfg.LastKernelSize, 1, true, false)
	if err != nil {
		return nil, err
	}
	m.Decoder = append(m.Decoder, &mimiSeanetLayer{elu: true}, &mimiSeanetLayer{conv: dl})
	if m.scale, err = tensor.New([]float32{float32(1 / math.Sqrt(float64(cfg.HeadDim)))}, []int{1}, dev); err != nil {
		return nil, err
	}
	return m, nil
}

// Parameters lists every weight and codebook buffer.
func (m *Mimi) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	seanet := func(ls []*mimiSeanetLayer) {
		for _, l := range ls {
			switch {
			case l.conv != nil:
				ps = append(ps, l.conv.W)
				if l.conv.B != nil {
					ps = append(ps, l.conv.B)
				}
			case l.convT != nil:
				ps = append(ps, l.convT.W)
				if l.convT.B != nil {
					ps = append(ps, l.convT.B)
				}
			case l.resnet != nil:
				ps = append(ps, l.resnet.Conv1.W, l.resnet.Conv1.B, l.resnet.Conv2.W, l.resnet.Conv2.B)
			}
		}
	}
	tf := func(t *mimiTransformer) {
		for _, l := range t.Layers {
			ps = append(ps, l.InNorm.Gamma, l.InNorm.Beta, l.PostAttnNorm.Gamma, l.PostAttnNorm.Beta,
				l.QProj.W, l.KProj.W, l.VProj.W, l.OProj.W, l.FC1.W, l.FC2.W, l.AttnScale, l.MLPScale)
		}
	}
	rvq := func(q *mimiRVQ) {
		if q.InProj != nil {
			ps = append(ps, q.InProj, q.OutProj)
		}
		for _, cb := range q.Layers {
			ps = append(ps, cb.EmbedSum, cb.ClusterUsage)
		}
	}
	seanet(m.Encoder)
	tf(m.EncoderTF)
	if m.Downsample != nil {
		ps = append(ps, m.Downsample.W, m.Upsample.W)
	}
	rvq(m.Semantic)
	rvq(m.Acoustic)
	tf(m.DecoderTF)
	seanet(m.Decoder)
	return ps
}

func (m *Mimi) ParamCount() int {
	n := 0
	for _, p := range m.Parameters() {
		n += p.NumElements()
	}
	return n
}

func runSeanet(ls []*mimiSeanetLayer, x *tensor.Tensor, s *mimiStream) (*tensor.Tensor, error) {
	var err error
	for i, l := range ls {
		if x, err = l.forward(x, s); err != nil {
			return nil, fmt.Errorf("seanet layer %d: %w", i, err)
		}
	}
	return x, nil
}

// transformer runs x (B, T, D) through a causal sliding-window transformer
// with RoPE, pre-LayerNorms and layer scales. With a cache the positions
// continue from cache.Len, past keys are attended through the window and
// only the last SlidingWindow keys are kept.
func (m *Mimi) transformer(t *mimiTransformer, x *tensor.Tensor, cache *KVCache) (*tensor.Tensor, error) {
	cfg := m.Cfg
	B, T := x.Dim(0), x.Dim(1)
	NH, HD := cfg.NumHeads, cfg.HeadDim
	dev := x.Device()
	prior := 0
	if cache != nil {
		prior = cache.Len
	}
	posData := make([]int64, T)
	for i := range posData {
		posData[i] = int64(prior + i)
	}
	pos, err := tensor.New(posData, []int{T}, dev)
	if err != nil {
		return nil, err
	}
	for li, l := range t.Layers {
		h, err := l.InNorm.Forward(x)
		if err != nil {
			return nil, err
		}
		q, err := l.QProj.Forward(h)
		if err != nil {
			return nil, err
		}
		k, err := l.KProj.Forward(h)
		if err != nil {
			return nil, err
		}
		v, err := l.VProj.Forward(h)
		if err != nil {
			return nil, err
		}
		if q, err = headsForward(q, B, T, NH, HD); err != nil {
			return nil, err
		}
		if k, err = headsForward(k, B, T, NH, HD); err != nil {
			return nil, err
		}
		if v, err = headsForward(v, B, T, NH, HD); err != nil {
			return nil, err
		}
		if q, err = ops.RoPE(q, pos, cfg.RopeBase); err != nil {
			return nil, err
		}
		if k, err = ops.RoPE(k, pos, cfg.RopeBase); err != nil {
			return nil, err
		}
		kvPrior := 0
		if cache != nil {
			if cache.K[li] != nil {
				kvPrior = cache.K[li].Dim(2)
				if k, err = ops.Concat([]*tensor.Tensor{cache.K[li], k}, 2); err != nil {
					return nil, err
				}
				if v, err = ops.Concat([]*tensor.Tensor{cache.V[li], v}, 2); err != nil {
					return nil, err
				}
			}
			cache.K[li], cache.V[li] = k, v
			if total := kvPrior + T; total > cfg.SlidingWindow {
				if cache.K[li], err = trimKV(k, total-cfg.SlidingWindow, cfg.SlidingWindow); err != nil {
					return nil, err
				}
				if cache.V[li], err = trimKV(v, total-cfg.SlidingWindow, cfg.SlidingWindow); err != nil {
					return nil, err
				}
			}
		}
		total := kvPrior + T
		var mask *tensor.Tensor
		if T > 1 || total > cfg.SlidingWindow {
			if mask, err = m.tabs.mask(T, kvPrior, cfg.SlidingWindow, false, x.DType(), dev); err != nil {
				return nil, err
			}
		}
		if q, err = autograd.Reshape(q, B*NH, T, HD); err != nil {
			return nil, err
		}
		if k, err = autograd.Reshape(k, B*NH, total, HD); err != nil {
			return nil, err
		}
		if v, err = autograd.Reshape(v, B*NH, total, HD); err != nil {
			return nil, err
		}
		kT, err := autograd.Transpose(k, 1, 2)
		if err != nil {
			return nil, err
		}
		scores, err := autograd.MatMul(q, kT)
		if err != nil {
			return nil, err
		}
		if scores, err = ops.Mul(scores, m.scale); err != nil {
			return nil, err
		}
		if mask != nil {
			if scores, err = ops.Add(scores, mask); err != nil {
				return nil, err
			}
		}
		attn, err := ops.Softmax(scores, -1)
		if err != nil {
			return nil, err
		}
		ctx, err := autograd.MatMul(attn, v)
		if err != nil {
			return nil, err
		}
		if ctx, err = autograd.Reshape(ctx, B, NH, T, HD); err != nil {
			return nil, err
		}
		if ctx, err = autograd.Permute(ctx, 0, 2, 1, 3); err != nil {
			return nil, err
		}
		if ctx, err = autograd.Contiguous(ctx); err != nil {
			return nil, err
		}
		if ctx, err = autograd.Reshape(ctx, B, T, NH*HD); err != nil {
			return nil, err
		}
		o, err := l.OProj.Forward(ctx)
		if err != nil {
			return nil, err
		}
		if o, err = ops.Mul(o, l.AttnScale); err != nil {
			return nil, err
		}
		if x, err = ops.Add(x, o); err != nil {
			return nil, err
		}
		if h, err = l.PostAttnNorm.Forward(x); err != nil {
			return nil, err
		}
		if h, err = l.FC1.Forward(h); err != nil {
			return nil, err
		}
		if h, err = ops.GeluErf(h); err != nil {
			return nil, err
		}
		if h, err = l.FC2.Forward(h); err != nil {
			return nil, err
		}
		if h, err = ops.Mul(h, l.MLPScale); err != nil {
			return nil, err
		}
		if x, err = ops.Add(x, h); err != nil {
			return nil, fmt.Errorf("transformer layer %d: %w", li, err)
		}
	}
	if cache != nil {
		cache.Len += T
	}
	return x, nil
}

// Latents runs the encoder: audio (B, C, L) F32 → latents (B, D, T) at
// the codec frame rate.
func (m *Mimi) Latents(audio *tensor.Tensor) (*tensor.Tensor, error) {
	return m.latents(audio, nil)
}

func (m *Mimi) latents(audio *tensor.Tensor, s *mimiStream) (*tensor.Tensor, error) {
	if audio.Rank() != 3 || audio.Dim(1) != m.Cfg.Channels {
		return nil, fmt.Errorf("Mimi.Latents: audio must be (B, %d, L), got %v", m.Cfg.Channels, audio.Shape())
	}
	h, err := runSeanet(m.Encoder, audio, s)
	if err != nil {
		return nil, err
	}
	if h, err = autograd.Permute(h, 0, 2, 1); err != nil { // (B, T, D)
		return nil, err
	}
	if h, err = autograd.Contiguous(h); err != nil {
		return nil, err
	}
	if h, err = m.transformer(m.EncoderTF, h, s.kv()); err != nil {
		return nil, err
	}
	if h, err = autograd.Permute(h, 0, 2, 1); err != nil {
		return nil, err
	}
	if h, err = autograd.Contiguous(h); err != nil {
		return nil, err
	}
	if m.Downsample != nil {
		if h, err = m.Downsample.forward(h, s.conv(m.Downsample)); err != nil {
			return nil, err
		}
	}
	return h, nil
}

func (cb *mimiCodebook) prepare() error {
	if cb.embed != nil {
		return nil
	}
	usage, err := ops.Clamp(cb.ClusterUsage, 1e-5, math.MaxFloat32)
	if err != nil {
		return err
	}
	u, err := usage.Reshape(cb.ClusterUsage.Dim(0), 1)
	if err != nil {
		return err
	}
	if cb.embed, err = ops.Div(cb.EmbedSum, u); err != nil {
		return err
	}
	t, err := autograd.Transpose(cb.embed, 0, 1)
	if err != nil {
		return err
	}
	if cb.embedT, err = autograd.Contiguous(t); err != nil {
		return err
	}
	sq, err := ops.Mul(cb.embed, cb.embed)
	if err != nil {
		return err
	}
	if sq, err = ops.Sum(sq, ops.ReduceOptions{Axes: []int{1}}); err != nil {
		return err
	}
	cb.embedSq, err = sq.Reshape(1, cb.embed.Dim(0))
	return err
}

// encode returns the nearest-codebook index per row of x (N, D) as I64 (N,).
// Small inputs (≤ 25 rows, i.e. streaming or short clips) use the exact
// |x−e|² distance; larger ones the expanded |e|² − 2x·e form. torch.cdist
// always takes the expanded form here (the codebook has > 25 rows), so on
// near-ties (gap ≲ 1e-3) transformers can pick the second-nearest centroid.
func (cb *mimiCodebook) encode(x *tensor.Tensor) (*tensor.Tensor, error) {
	if err := cb.prepare(); err != nil {
		return nil, err
	}
	N, D := x.Dim(0), x.Dim(1)
	K := cb.embed.Dim(0)
	if N <= 25 {
		x3, err := x.Reshape(N, 1, D)
		if err != nil {
			return nil, err
		}
		e3, err := cb.embed.Reshape(1, K, D)
		if err != nil {
			return nil, err
		}
		diff, err := ops.Sub(x3, e3) // (N, K, D)
		if err != nil {
			return nil, err
		}
		if diff, err = ops.Mul(diff, diff); err != nil {
			return nil, err
		}
		d, err := ops.Sum(diff, ops.ReduceOptions{Axes: []int{2}})
		if err != nil {
			return nil, err
		}
		idx, err := ops.Argmin(d, 1, false)
		if err != nil {
			return nil, err
		}
		if idx.DType() != dtype.I64 {
			return ops.Cast(idx, dtype.I64)
		}
		return idx, nil
	}
	xe, err := autograd.MatMul(x, cb.embedT) // (N, K)
	if err != nil {
		return nil, err
	}
	two, err := tensor.New([]float32{-2}, []int{1}, x.Device())
	if err != nil {
		return nil, err
	}
	if xe, err = ops.Mul(xe, two); err != nil {
		return nil, err
	}
	d, err := ops.Add(xe, cb.embedSq) // |e|² − 2x·e (|x|² is constant per row)
	if err != nil {
		return nil, err
	}
	idx, err := ops.Argmin(d, 1, false)
	if err != nil {
		return nil, err
	}
	if idx.DType() != dtype.I64 {
		return ops.Cast(idx, dtype.I64)
	}
	return idx, nil
}

// decode gathers codebook rows: idx (N,) → (N, D).
func (cb *mimiCodebook) decode(idx *tensor.Tensor) (*tensor.Tensor, error) {
	if err := cb.prepare(); err != nil {
		return nil, err
	}
	return ops.IndexSelect(cb.embed, idx)
}

// encode quantises x (B, D, T) with the first n layers → codes (n, B, T).
func (q *mimiRVQ) encode(x *tensor.Tensor, n int) ([]*tensor.Tensor, error) {
	var err error
	if q.InProj != nil {
		if x, err = ops.Conv1d(x, q.InProj, nil, 1, 0); err != nil {
			return nil, err
		}
	}
	B, Dq, T := x.Dim(0), x.Dim(1), x.Dim(2)
	if x, err = autograd.Permute(x, 0, 2, 1); err != nil {
		return nil, err
	}
	if x, err = autograd.Contiguous(x); err != nil {
		return nil, err
	}
	if x, err = autograd.Reshape(x, B*T, Dq); err != nil {
		return nil, err
	}
	residual := x
	var codes []*tensor.Tensor
	for _, cb := range q.Layers[:n] {
		idx, err := cb.encode(residual)
		if err != nil {
			return nil, err
		}
		qv, err := cb.decode(idx)
		if err != nil {
			return nil, err
		}
		if residual, err = ops.Sub(residual, qv); err != nil {
			return nil, err
		}
		c, err := idx.Reshape(B, T)
		if err != nil {
			return nil, err
		}
		codes = append(codes, c)
	}
	return codes, nil
}

// decode sums the codebook vectors of codes (per layer (B, T)) → (B, D, T).
func (q *mimiRVQ) decode(codes []*tensor.Tensor) (*tensor.Tensor, error) {
	var sum *tensor.Tensor
	B, T := codes[0].Dim(0), codes[0].Dim(1)
	for i, c := range codes {
		flat, err := c.Reshape(B * T)
		if err != nil {
			return nil, err
		}
		qv, err := q.Layers[i].decode(flat) // (B·T, Dq)
		if err != nil {
			return nil, err
		}
		if sum == nil {
			sum = qv
		} else if sum, err = ops.Add(sum, qv); err != nil {
			return nil, err
		}
	}
	Dq := sum.Dim(1)
	x, err := autograd.Reshape(sum, B, T, Dq)
	if err != nil {
		return nil, err
	}
	if x, err = autograd.Permute(x, 0, 2, 1); err != nil {
		return nil, err
	}
	if x, err = autograd.Contiguous(x); err != nil {
		return nil, err
	}
	if q.OutProj != nil {
		return ops.Conv1d(x, q.OutProj, nil, 1, 0)
	}
	return x, nil
}

// Encode turns audio (B, C, L) F32 into codes (B, n, T) I64 using the
// first n codebooks (n ≤ NumQuantizers; Moshi uses 8).
func (m *Mimi) Encode(audio *tensor.Tensor, n int) (*tensor.Tensor, error) {
	return m.encode(audio, n, nil)
}

func (m *Mimi) encode(audio *tensor.Tensor, n int, s *mimiStream) (*tensor.Tensor, error) {
	if n <= 0 || n > m.Cfg.NumQuantizers {
		n = m.Cfg.NumQuantizers
	}
	if n < m.Cfg.NumSemantic {
		return nil, fmt.Errorf("Mimi.Encode: need at least %d codebooks", m.Cfg.NumSemantic)
	}
	lat, err := m.latents(audio, s)
	if err != nil {
		return nil, err
	}
	codes, err := m.Semantic.encode(lat, m.Cfg.NumSemantic)
	if err != nil {
		return nil, err
	}
	if n > m.Cfg.NumSemantic {
		ac, err := m.Acoustic.encode(lat, n-m.Cfg.NumSemantic)
		if err != nil {
			return nil, err
		}
		codes = append(codes, ac...)
	}
	B, T := codes[0].Dim(0), codes[0].Dim(1)
	for i := range codes {
		if codes[i], err = codes[i].Reshape(B, 1, T); err != nil {
			return nil, err
		}
	}
	return ops.Concat(codes, 1)
}

// Decode turns codes (B, n, T) I64 back into audio (B, C, T·HopLength).
func (m *Mimi) Decode(codes *tensor.Tensor) (*tensor.Tensor, error) {
	return m.decode(codes, nil)
}

func (m *Mimi) decode(codes *tensor.Tensor, s *mimiStream) (*tensor.Tensor, error) {
	if codes.Rank() != 3 {
		return nil, fmt.Errorf("Mimi.Decode: codes must be (B, n, T), got %v", codes.Shape())
	}
	n := codes.Dim(1)
	split := func(from, to int) ([]*tensor.Tensor, error) {
		var out []*tensor.Tensor
		for i := from; i < to; i++ {
			c, err := codes.Narrow(1, i, 1)
			if err != nil {
				return nil, err
			}
			if c, err = autograd.Contiguous(c); err != nil {
				return nil, err
			}
			if c, err = c.Reshape(codes.Dim(0), codes.Dim(2)); err != nil {
				return nil, err
			}
			out = append(out, c)
		}
		return out, nil
	}
	sem, err := split(0, min(n, m.Cfg.NumSemantic))
	if err != nil {
		return nil, err
	}
	x, err := m.Semantic.decode(sem)
	if err != nil {
		return nil, err
	}
	if n > m.Cfg.NumSemantic {
		ac, err := split(m.Cfg.NumSemantic, n)
		if err != nil {
			return nil, err
		}
		y, err := m.Acoustic.decode(ac)
		if err != nil {
			return nil, err
		}
		if x, err = ops.Add(x, y); err != nil {
			return nil, err
		}
	}
	return m.decodeLatents(x, s)
}

// DecodeLatents runs the decoder on quantised latents (B, D, T).
func (m *Mimi) DecodeLatents(x *tensor.Tensor) (*tensor.Tensor, error) {
	return m.decodeLatents(x, nil)
}

func (m *Mimi) decodeLatents(x *tensor.Tensor, s *mimiStream) (*tensor.Tensor, error) {
	var err error
	if m.Upsample != nil {
		if x, err = m.Upsample.forward(x, s.convT(m.Upsample)); err != nil {
			return nil, err
		}
	}
	if x, err = autograd.Permute(x, 0, 2, 1); err != nil {
		return nil, err
	}
	if x, err = autograd.Contiguous(x); err != nil {
		return nil, err
	}
	if x, err = m.transformer(m.DecoderTF, x, s.kv()); err != nil {
		return nil, err
	}
	if x, err = autograd.Permute(x, 0, 2, 1); err != nil {
		return nil, err
	}
	if x, err = autograd.Contiguous(x); err != nil {
		return nil, err
	}
	return runSeanet(m.Decoder, x, s)
}

// MimiStream is a stateful frame-by-frame codec: EncodeStep turns each
// 80 ms chunk of audio into one frame of codes and DecodeStep each frame of
// codes into 80 ms of audio, with the conv memories and transformer KV
// caches of kyutai's streaming modules, so the results equal the
// full-sequence Encode / Decode of the concatenated input (no right padding
// at chunk ends). Encoder and decoder states are independent; Reset clears
// both.
type MimiStream struct {
	m        *Mimi
	enc, dec *mimiStream
}

// NewStream starts a streaming session.
func (m *Mimi) NewStream() *MimiStream {
	return &MimiStream{m: m, enc: newMimiStream(m.Cfg.NumLayers), dec: newMimiStream(m.Cfg.NumLayers)}
}

// Reset forgets all state.
func (s *MimiStream) Reset() {
	s.enc, s.dec = newMimiStream(s.m.Cfg.NumLayers), newMimiStream(s.m.Cfg.NumLayers)
}

// EncodeStep encodes audio (B, C, k·HopLength) into codes (B, n, k).
func (s *MimiStream) EncodeStep(audio *tensor.Tensor, n int) (*tensor.Tensor, error) {
	if audio.Rank() != 3 || audio.Dim(2) == 0 || audio.Dim(2)%s.m.Cfg.HopLength() != 0 {
		return nil, fmt.Errorf("Mimi.EncodeStep: audio must be (B, C, k·%d), got %v", s.m.Cfg.HopLength(), audio.Shape())
	}
	return s.m.encode(audio, n, s.enc)
}

// DecodeStep decodes codes (B, n, k) into audio (B, C, k·HopLength).
func (s *MimiStream) DecodeStep(codes *tensor.Tensor) (*tensor.Tensor, error) {
	return s.m.decode(codes, s.dec)
}
