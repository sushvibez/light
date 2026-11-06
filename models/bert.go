package models

import (
	"fmt"
	"math"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// BertConfig describes one BERT-family encoder (BERT, BGE, MiniLM, …).
//
// Compared with the decoder LLMs in this package BERT uses learned ABSOLUTE
// position embeddings plus token-type (segment) embeddings, post-LayerNorm
// residual blocks, fully bidirectional attention (a key-padding mask only)
// and bias on every Linear.
type BertConfig struct {
	VocabSize        int
	HiddenSize       int
	NumLayers        int
	NumHeads         int
	IntermediateSize int
	MaxPosition      int
	TypeVocabSize    int     // segment types (2 for BERT, 1 for RoBERTa)
	LayerNormEps     float64 // 1e-12 for BERT
	HiddenAct        string  // "gelu" (exact erf, HF default) or "gelu_new" (tanh approximation)
	PadTokenID       int
}

// BertBaseUncased is the config for bert-base-uncased (110M).
func BertBaseUncased() BertConfig {
	return BertConfig{
		VocabSize: 30522, HiddenSize: 768, NumLayers: 12, NumHeads: 12,
		IntermediateSize: 3072, MaxPosition: 512, TypeVocabSize: 2,
		LayerNormEps: 1e-12, HiddenAct: "gelu", PadTokenID: 0,
	}
}

// HeadDim returns HiddenSize / NumHeads.
func (c BertConfig) HeadDim() int { return c.HiddenSize / c.NumHeads }

// BertLayer is one post-LN encoder block:
//
//	h = LN1(x + AttnOut(SelfAttn(x)))
//	y = LN2(h + Out(act(Inter(h))))
type BertLayer struct {
	Q, K, V, AttnOut *nn.Linear
	AttnLN           *nn.LayerNorm
	Inter, Out       *nn.Linear
	OutLN            *nn.LayerNorm
}

// BertModel is the encoder stack (embeddings + N layers) with an optional
// pooler (Linear + tanh on the [CLS] token).
type BertModel struct {
	Cfg     BertConfig
	WordEmb *nn.Embedding
	PosEmb  *nn.Embedding
	TypeEmb *nn.Embedding
	EmbLN   *nn.LayerNorm
	Layers  []*BertLayer
	Pooler  *nn.Linear // nil when the checkpoint has no pooler

	dev   device.Device
	scale *tensor.Tensor // (1,) F32 = 1/sqrt(headDim)
}

// NewBertModel builds a BERT encoder with random weights. `withPooler`
// allocates the [CLS] pooler head.
func NewBertModel(cfg BertConfig, withPooler bool, dev device.Device) (*BertModel, error) {
	if cfg.HiddenSize%cfg.NumHeads != 0 {
		return nil, fmt.Errorf("models.Bert: HiddenSize %d not divisible by NumHeads %d", cfg.HiddenSize, cfg.NumHeads)
	}
	switch cfg.HiddenAct {
	case "", "gelu", "gelu_new":
	default:
		return nil, fmt.Errorf("models.Bert: unsupported hidden_act %q (gelu, gelu_new)", cfg.HiddenAct)
	}
	if cfg.TypeVocabSize <= 0 {
		cfg.TypeVocabSize = 1
	}
	H := cfg.HiddenSize
	m := &BertModel{Cfg: cfg, dev: dev}
	var err error
	if m.WordEmb, err = nn.NewEmbeddingUninit(cfg.VocabSize, H, dtype.F32, dev); err != nil {
		return nil, err
	}
	if m.PosEmb, err = nn.NewEmbeddingUninit(cfg.MaxPosition, H, dtype.F32, dev); err != nil {
		return nil, err
	}
	if m.TypeEmb, err = nn.NewEmbeddingUninit(cfg.TypeVocabSize, H, dtype.F32, dev); err != nil {
		return nil, err
	}
	if m.EmbLN, err = nn.NewLayerNorm(H, cfg.LayerNormEps, dev); err != nil {
		return nil, err
	}
	m.Layers = make([]*BertLayer, cfg.NumLayers)
	for i := range m.Layers {
		l := &BertLayer{}
		if l.Q, err = nn.NewLinear(H, H, true, dev); err != nil {
			return nil, err
		}
		if l.K, err = nn.NewLinear(H, H, true, dev); err != nil {
			return nil, err
		}
		if l.V, err = nn.NewLinear(H, H, true, dev); err != nil {
			return nil, err
		}
		if l.AttnOut, err = nn.NewLinear(H, H, true, dev); err != nil {
			return nil, err
		}
		if l.AttnLN, err = nn.NewLayerNorm(H, cfg.LayerNormEps, dev); err != nil {
			return nil, err
		}
		if l.Inter, err = nn.NewLinear(H, cfg.IntermediateSize, true, dev); err != nil {
			return nil, err
		}
		if l.Out, err = nn.NewLinear(cfg.IntermediateSize, H, true, dev); err != nil {
			return nil, err
		}
		if l.OutLN, err = nn.NewLayerNorm(H, cfg.LayerNormEps, dev); err != nil {
			return nil, err
		}
		m.Layers[i] = l
	}
	if withPooler {
		if m.Pooler, err = nn.NewLinear(H, H, true, dev); err != nil {
			return nil, err
		}
	}
	scale := float32(1.0 / math.Sqrt(float64(cfg.HeadDim())))
	if m.scale, err = tensor.New([]float32{scale}, []int{1}, dev); err != nil {
		return nil, err
	}
	return m, nil
}

// Device returns the device the parameters live on.
func (m *BertModel) Device() device.Device { return m.dev }

// Forward runs the encoder and returns the last hidden state (B, T, H).
//
//	inputIDs:      I64 (B, T)
//	attentionMask: I64 or F32 (B, T) with 1 for real tokens and 0 for padding,
//	               or nil for no padding. Padded keys receive -inf attention
//	               scores so they never influence real positions.
//
// Token-type ids are all zero (single segment); use ForwardSegments for pairs.
func (m *BertModel) Forward(inputIDs, attentionMask *tensor.Tensor) (*tensor.Tensor, error) {
	return m.ForwardSegments(inputIDs, nil, attentionMask)
}

// ForwardSegments is Forward with explicit token-type ids (I64 (B, T), or
// nil for all-zero).
func (m *BertModel) ForwardSegments(inputIDs, tokenTypeIDs, attentionMask *tensor.Tensor) (*tensor.Tensor, error) {
	if inputIDs.Rank() != 2 {
		return nil, fmt.Errorf("Bert: inputIDs must be (B, T), got %v", inputIDs.Shape())
	}
	if inputIDs.DType() != dtype.I64 {
		return nil, fmt.Errorf("Bert: inputIDs must be I64, got %v", inputIDs.DType())
	}
	B, T := inputIDs.Dim(0), inputIDs.Dim(1)
	if T > m.Cfg.MaxPosition {
		return nil, fmt.Errorf("Bert: sequence length %d exceeds MaxPosition %d", T, m.Cfg.MaxPosition)
	}

	x, err := m.embed(inputIDs, tokenTypeIDs, B, T)
	if err != nil {
		return nil, err
	}
	var addMask *tensor.Tensor
	if attentionMask != nil {
		if addMask, err = additiveKeyMask(attentionMask, B, T); err != nil {
			return nil, err
		}
	}
	for i, l := range m.Layers {
		if x, err = m.layerForward(l, x, addMask, B, T); err != nil {
			return nil, fmt.Errorf("Bert layer %d: %w", i, err)
		}
	}
	return x, nil
}

// Pool applies the pooler (tanh(W·h[:,0]+b)) to a last hidden state and
// returns (B, H). Errors when the model has no pooler.
func (m *BertModel) Pool(hidden *tensor.Tensor) (*tensor.Tensor, error) {
	if m.Pooler == nil {
		return nil, fmt.Errorf("Bert: model has no pooler")
	}
	cls, err := CLSPooled(hidden)
	if err != nil {
		return nil, err
	}
	y, err := m.Pooler.Forward(cls)
	if err != nil {
		return nil, err
	}
	return ops.Tanh(y)
}

func (m *BertModel) embed(inputIDs, tokenTypeIDs *tensor.Tensor, B, T int) (*tensor.Tensor, error) {
	dev := inputIDs.Device()
	words, err := m.WordEmb.Forward(inputIDs) // (B, T, H)
	if err != nil {
		return nil, err
	}
	posIdx := make([]int64, B*T)
	for b := range B {
		for t := range T {
			posIdx[b*T+t] = int64(t)
		}
	}
	posIDs, err := tensor.New(posIdx, []int{B, T}, dev)
	if err != nil {
		return nil, err
	}
	pos, err := m.PosEmb.Forward(posIDs)
	if err != nil {
		return nil, err
	}
	if tokenTypeIDs == nil {
		if tokenTypeIDs, err = tensor.Zeros(dtype.I64, []int{B, T}, dev); err != nil {
			return nil, err
		}
	}
	typ, err := m.TypeEmb.Forward(tokenTypeIDs)
	if err != nil {
		return nil, err
	}
	x, err := ops.Add(words, pos)
	if err != nil {
		return nil, err
	}
	if x, err = ops.Add(x, typ); err != nil {
		return nil, err
	}
	return m.EmbLN.Forward(x)
}

// additiveKeyMask turns a (B, T) {0,1} mask into (B, 1, 1, T) F32 with 0 for
// kept keys and -inf for padded keys, computed as log(mask) on-device so no
// host round-trip is needed.
func additiveKeyMask(mask *tensor.Tensor, B, T int) (*tensor.Tensor, error) {
	if mask.Rank() != 2 || mask.Dim(0) != B || mask.Dim(1) != T {
		return nil, fmt.Errorf("Bert: attentionMask must be (%d, %d), got %v", B, T, mask.Shape())
	}
	f := mask
	var err error
	if f.DType() != dtype.F32 {
		if f, err = ops.Cast(mask, dtype.F32); err != nil {
			return nil, err
		}
	}
	if f, err = f.Contiguous(); err != nil {
		return nil, err
	}
	if f, err = ops.Log(f); err != nil { // log(1)=0, log(0)=-inf
		return nil, err
	}
	return f.Reshape(B, 1, 1, T)
}

// splitHeads: (B, T, H) → (B*NH, T, HD).
func splitHeads(x *tensor.Tensor, B, T, NH, HD int) (*tensor.Tensor, error) {
	r, err := x.Reshape(B, T, NH, HD)
	if err != nil {
		return nil, err
	}
	p, err := r.Permute(0, 2, 1, 3)
	if err != nil {
		return nil, err
	}
	c, err := p.Contiguous()
	if err != nil {
		return nil, err
	}
	return c.Reshape(B*NH, T, HD)
}

func (m *BertModel) layerForward(l *BertLayer, x, addMask *tensor.Tensor, B, T int) (*tensor.Tensor, error) {
	NH, HD, H := m.Cfg.NumHeads, m.Cfg.HeadDim(), m.Cfg.HiddenSize

	q, err := l.Q.Forward(x)
	if err != nil {
		return nil, err
	}
	k, err := l.K.Forward(x)
	if err != nil {
		return nil, err
	}
	v, err := l.V.Forward(x)
	if err != nil {
		return nil, err
	}
	if q, err = splitHeads(q, B, T, NH, HD); err != nil {
		return nil, err
	}
	if k, err = splitHeads(k, B, T, NH, HD); err != nil {
		return nil, err
	}
	if v, err = splitHeads(v, B, T, NH, HD); err != nil {
		return nil, err
	}
	kT, err := k.Transpose(1, 2) // (B*NH, HD, T)
	if err != nil {
		return nil, err
	}
	if kT, err = kT.Contiguous(); err != nil {
		return nil, err
	}
	scores, err := ops.MatMul(q, kT) // (B*NH, T, T)
	if err != nil {
		return nil, err
	}
	if scores, err = ops.Mul(scores, m.scale); err != nil {
		return nil, err
	}
	if addMask != nil {
		s4, err := scores.Reshape(B, NH, T, T)
		if err != nil {
			return nil, err
		}
		if s4, err = ops.Add(s4, addMask); err != nil {
			return nil, err
		}
		if scores, err = s4.Reshape(B*NH, T, T); err != nil {
			return nil, err
		}
	}
	probs, err := ops.Softmax(scores, -1)
	if err != nil {
		return nil, err
	}
	ctx, err := ops.MatMul(probs, v) // (B*NH, T, HD)
	if err != nil {
		return nil, err
	}
	// Merge heads: (B*NH, T, HD) → (B, NH, T, HD) → (B, T, NH, HD) → (B, T, H).
	c4, err := ctx.Reshape(B, NH, T, HD)
	if err != nil {
		return nil, err
	}
	cp, err := c4.Permute(0, 2, 1, 3)
	if err != nil {
		return nil, err
	}
	if cp, err = cp.Contiguous(); err != nil {
		return nil, err
	}
	merged, err := cp.Reshape(B, T, H)
	if err != nil {
		return nil, err
	}
	attnOut, err := l.AttnOut.Forward(merged)
	if err != nil {
		return nil, err
	}
	// Post-LN residual 1.
	h, err := ops.Add(x, attnOut)
	if err != nil {
		return nil, err
	}
	if h, err = l.AttnLN.Forward(h); err != nil {
		return nil, err
	}
	// FFN.
	inter, err := l.Inter.Forward(h)
	if err != nil {
		return nil, err
	}
	if inter, err = m.activation(inter); err != nil {
		return nil, err
	}
	ffn, err := l.Out.Forward(inter)
	if err != nil {
		return nil, err
	}
	y, err := ops.Add(h, ffn)
	if err != nil {
		return nil, err
	}
	return l.OutLN.Forward(y)
}

func (m *BertModel) activation(x *tensor.Tensor) (*tensor.Tensor, error) {
	if m.Cfg.HiddenAct == "gelu_new" {
		return ops.Gelu(x) // tanh approximation
	}
	return ops.GeluErf(x) // exact erf GELU (HF "gelu")
}

// Parameters returns every learnable tensor.
func (m *BertModel) Parameters() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.WordEmb.Weight, m.PosEmb.Weight, m.TypeEmb.Weight}
	ps = append(ps, m.EmbLN.Parameters()...)
	for _, l := range m.Layers {
		ps = append(ps, l.Q.Parameters()...)
		ps = append(ps, l.K.Parameters()...)
		ps = append(ps, l.V.Parameters()...)
		ps = append(ps, l.AttnOut.Parameters()...)
		ps = append(ps, l.AttnLN.Parameters()...)
		ps = append(ps, l.Inter.Parameters()...)
		ps = append(ps, l.Out.Parameters()...)
		ps = append(ps, l.OutLN.Parameters()...)
	}
	if m.Pooler != nil {
		ps = append(ps, m.Pooler.Parameters()...)
	}
	return ps
}

// ParamCount returns the total number of parameters.
func (m *BertModel) ParamCount() int {
	n := 0
	for _, p := range m.Parameters() {
		n += p.NumElements()
	}
	return n
}

// ---- sentence-embedding helpers ----

// CLSPooled returns hidden[:, 0, :] as a contiguous (B, H) tensor.
func CLSPooled(hidden *tensor.Tensor) (*tensor.Tensor, error) {
	if hidden.Rank() != 3 {
		return nil, fmt.Errorf("CLSPooled: expected (B, T, H), got %v", hidden.Shape())
	}
	cls, err := hidden.Narrow(1, 0, 1)
	if err != nil {
		return nil, err
	}
	if cls, err = cls.Contiguous(); err != nil {
		return nil, err
	}
	return cls.Reshape(hidden.Dim(0), hidden.Dim(2))
}

// MeanPooled averages hidden (B, T, H) over the time axis, weighting by
// mask (B, T) when non-nil (I64 or F32 with 1 for real tokens), returning
// (B, H).
func MeanPooled(hidden, mask *tensor.Tensor) (*tensor.Tensor, error) {
	if hidden.Rank() != 3 {
		return nil, fmt.Errorf("MeanPooled: expected (B, T, H), got %v", hidden.Shape())
	}
	B, T := hidden.Dim(0), hidden.Dim(1)
	if mask == nil {
		return ops.Mean(hidden, ops.ReduceOptions{Axes: []int{1}})
	}
	if mask.Rank() != 2 || mask.Dim(0) != B || mask.Dim(1) != T {
		return nil, fmt.Errorf("MeanPooled: mask must be (%d, %d), got %v", B, T, mask.Shape())
	}
	f := mask
	var err error
	if f.DType() != hidden.DType() {
		if f, err = ops.Cast(mask, hidden.DType()); err != nil {
			return nil, err
		}
	}
	if f, err = f.Contiguous(); err != nil {
		return nil, err
	}
	f3, err := f.Reshape(B, T, 1)
	if err != nil {
		return nil, err
	}
	weighted, err := ops.Mul(hidden, f3)
	if err != nil {
		return nil, err
	}
	num, err := ops.Sum(weighted, ops.ReduceOptions{Axes: []int{1}}) // (B, H)
	if err != nil {
		return nil, err
	}
	den, err := ops.Sum(f3, ops.ReduceOptions{Axes: []int{1}}) // (B, 1)
	if err != nil {
		return nil, err
	}
	return ops.Div(num, den)
}

// Normalize L2-normalises each row of a (B, H) tensor.
func Normalize(x *tensor.Tensor) (*tensor.Tensor, error) {
	norm, err := ops.L2Norm(x, ops.ReduceOptions{Axes: []int{x.Rank() - 1}, KeepDims: true})
	if err != nil {
		return nil, err
	}
	return ops.Div(x, norm)
}

// ---- masked-LM head ----

// BertForMaskedLM is BertModel plus the MLM head: Linear + act + LayerNorm,
// then a decoder projection tied to the word embeddings with its own bias.
type BertForMaskedLM struct {
	Bert        *BertModel
	Transform   *nn.Linear
	TransformLN *nn.LayerNorm
	Decoder     *nn.Linear // W tied to Bert.WordEmb.Weight; B is the per-vocab bias
}

// NewBertForMaskedLM builds the encoder (no pooler) and the MLM head.
func NewBertForMaskedLM(cfg BertConfig, dev device.Device) (*BertForMaskedLM, error) {
	bert, err := NewBertModel(cfg, false, dev)
	if err != nil {
		return nil, err
	}
	m := &BertForMaskedLM{Bert: bert}
	if m.Transform, err = nn.NewLinear(cfg.HiddenSize, cfg.HiddenSize, true, dev); err != nil {
		return nil, err
	}
	if m.TransformLN, err = nn.NewLayerNorm(cfg.HiddenSize, cfg.LayerNormEps, dev); err != nil {
		return nil, err
	}
	if m.Decoder, err = nn.NewLinearTied(bert.WordEmb.Weight, true, dev); err != nil {
		return nil, err
	}
	return m, nil
}

// Forward returns MLM logits (B, T, V).
func (m *BertForMaskedLM) Forward(inputIDs, attentionMask *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := m.Bert.Forward(inputIDs, attentionMask)
	if err != nil {
		return nil, err
	}
	if h, err = m.Transform.Forward(h); err != nil {
		return nil, err
	}
	if h, err = m.Bert.activation(h); err != nil {
		return nil, err
	}
	if h, err = m.TransformLN.Forward(h); err != nil {
		return nil, err
	}
	return m.Decoder.Forward(h)
}

// Parameters returns the encoder parameters plus the head's own tensors
// (the tied decoder weight is not repeated).
func (m *BertForMaskedLM) Parameters() []*tensor.Tensor {
	ps := m.Bert.Parameters()
	ps = append(ps, m.Transform.Parameters()...)
	ps = append(ps, m.TransformLN.Parameters()...)
	ps = append(ps, m.Decoder.B)
	return ps
}
