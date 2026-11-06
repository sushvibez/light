package models

import (
	"fmt"
	"math"
	"unsafe"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/nn"
	"light/tensor"
)

// GPT2Config mirrors HuggingFace's `GPT2Config` field names for the subset
// we implement. Presets for the public GPT-2 sizes are on the struct.
type GPT2Config struct {
	VocabSize  int // e.g. 50257
	NPositions int // max sequence length (1024 for GPT-2)
	NEmbd      int // hidden size
	NLayer     int // number of transformer blocks
	NHead      int // number of attention heads (NEmbd must be divisible by NHead)
	LayerNorm  float64
	Causal     bool // always true for GPT — kept as a flag for clarity
	Tied       bool // tie LM-head weight with the token embedding (HF default)
}

// GPT2Small returns the configuration for `gpt2` (124M params).
func GPT2Small() GPT2Config {
	return GPT2Config{
		VocabSize: 50257, NPositions: 1024,
		NEmbd: 768, NLayer: 12, NHead: 12,
		LayerNorm: 1e-5, Causal: true, Tied: true,
	}
}

// GPT2Medium → `gpt2-medium` (355M).
func GPT2Medium() GPT2Config {
	return GPT2Config{
		VocabSize: 50257, NPositions: 1024,
		NEmbd: 1024, NLayer: 24, NHead: 16,
		LayerNorm: 1e-5, Causal: true, Tied: true,
	}
}

// GPT2Tiny is a scaled-down config for quick smoke tests.
func GPT2Tiny() GPT2Config {
	return GPT2Config{
		VocabSize: 256, NPositions: 128,
		NEmbd: 64, NLayer: 2, NHead: 4,
		LayerNorm: 1e-5, Causal: true, Tied: true,
	}
}

// gpt2Block is one pre-norm GPT-2 decoder layer: LN → MHA → residual →
// LN → FFN → residual. FFN is Linear(H, 4H) → GELU → Linear(4H, H),
// matching the HF GPT-2 FFN exactly.
type gpt2Block struct {
	Norm1 *nn.LayerNorm
	Attn  *nn.MultiHeadAttention
	Norm2 *nn.LayerNorm
	FC1   *nn.Linear
	FC2   *nn.Linear
}

func newGPT2Block(cfg GPT2Config, dev device.Device) (*gpt2Block, error) {
	n1, err := nn.NewLayerNorm(cfg.NEmbd, cfg.LayerNorm, dev)
	if err != nil {
		return nil, err
	}
	attn, err := nn.NewMultiHeadAttention(cfg.NEmbd, cfg.NHead, cfg.Causal, true, dev)
	if err != nil {
		return nil, err
	}
	n2, err := nn.NewLayerNorm(cfg.NEmbd, cfg.LayerNorm, dev)
	if err != nil {
		return nil, err
	}
	fc1, err := nn.NewLinear(cfg.NEmbd, 4*cfg.NEmbd, true, dev)
	if err != nil {
		return nil, err
	}
	fc2, err := nn.NewLinear(4*cfg.NEmbd, cfg.NEmbd, true, dev)
	if err != nil {
		return nil, err
	}
	return &gpt2Block{Norm1: n1, Attn: attn, Norm2: n2, FC1: fc1, FC2: fc2}, nil
}

func (b *gpt2Block) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	// Attn sub-layer
	h, err := b.Norm1.Forward(x)
	if err != nil {
		return nil, err
	}
	h, err = b.Attn.Forward(h)
	if err != nil {
		return nil, err
	}
	x, err = autograd.Add(x, h)
	if err != nil {
		return nil, err
	}
	// FFN sub-layer
	h, err = b.Norm2.Forward(x)
	if err != nil {
		return nil, err
	}
	h, err = b.FC1.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = autograd.GELU(h)
	if err != nil {
		return nil, err
	}
	h, err = b.FC2.Forward(h)
	if err != nil {
		return nil, err
	}
	return autograd.Add(x, h)
}

func (b *gpt2Block) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, b.Norm1.Parameters()...)
	ps = append(ps, b.Attn.Parameters()...)
	ps = append(ps, b.Norm2.Parameters()...)
	ps = append(ps, b.FC1.Parameters()...)
	ps = append(ps, b.FC2.Parameters()...)
	return ps
}

// GPT2 is the GPT-2 language model. Takes token IDs `(B, T)` and returns
// logits of shape `(B, T, VocabSize)`.
//
// Uses learned absolute position embeddings (not RoPE — that's for Llama).
// The LM head is a Linear(NEmbd → VocabSize) with its own weight, not tied
// to the token embedding (HF's GPT-2 ties them; a future wave can add that).
type GPT2 struct {
	Cfg       GPT2Config
	TokenEmb  *nn.Embedding // (VocabSize, NEmbd)
	PosEmb    *nn.Embedding // (NPositions, NEmbd)
	Blocks    []*gpt2Block
	FinalNorm *nn.LayerNorm
	LMHead    *nn.Linear // NEmbd → VocabSize, no bias
}

// NewGPT2 constructs a GPT-2 model on the given device with random init.
func NewGPT2(cfg GPT2Config, dev device.Device) (*GPT2, error) {
	if cfg.NEmbd%cfg.NHead != 0 {
		return nil, fmt.Errorf("models.GPT2: NEmbd (%d) not divisible by NHead (%d)", cfg.NEmbd, cfg.NHead)
	}
	tokEmb, err := nn.NewEmbedding(cfg.VocabSize, cfg.NEmbd, dev)
	if err != nil {
		return nil, err
	}
	posEmb, err := nn.NewEmbedding(cfg.NPositions, cfg.NEmbd, dev)
	if err != nil {
		return nil, err
	}
	blocks := make([]*gpt2Block, cfg.NLayer)
	for i := range blocks {
		b, err := newGPT2Block(cfg, dev)
		if err != nil {
			return nil, err
		}
		blocks[i] = b
	}
	ln, err := nn.NewLayerNorm(cfg.NEmbd, cfg.LayerNorm, dev)
	if err != nil {
		return nil, err
	}
	head, err := nn.NewLinear(cfg.NEmbd, cfg.VocabSize, false, dev)
	if err != nil {
		return nil, err
	}
	if cfg.Tied {
		// Share storage with the token embedding. The freshly-allocated W is
		// dropped to GC. Both shapes are (VocabSize, NEmbd).
		head.W = tokEmb.Weight
	}
	return &GPT2{Cfg: cfg, TokenEmb: tokEmb, PosEmb: posEmb, Blocks: blocks, FinalNorm: ln, LMHead: head}, nil
}

func (m *GPT2) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, m.TokenEmb.Parameters()...)
	ps = append(ps, m.PosEmb.Parameters()...)
	for _, b := range m.Blocks {
		ps = append(ps, b.Parameters()...)
	}
	ps = append(ps, m.FinalNorm.Parameters()...)
	// Skip LMHead.W when it's tied to the token embedding to avoid double
	// counting (and double gradient accumulation if training).
	for _, p := range m.LMHead.Parameters() {
		if p == m.TokenEmb.Weight {
			continue
		}
		ps = append(ps, p)
	}
	return ps
}

// Forward takes `tokens` of shape `(B, T)` (I64) and returns logits of
// shape `(B, T, VocabSize)` (F32). Uses absolute position indices 0..T-1.
func (m *GPT2) Forward(tokens *tensor.Tensor) (*tensor.Tensor, error) {
	if tokens.Rank() != 2 {
		return nil, fmt.Errorf("GPT2: tokens must be (B, T), got %v", tokens.Shape())
	}
	if tokens.DType() != dtype.I64 {
		return nil, fmt.Errorf("GPT2: tokens must be I64, got %v", tokens.DType())
	}
	B := tokens.Dim(0)
	T := tokens.Dim(1)
	if T > m.Cfg.NPositions {
		return nil, fmt.Errorf("GPT2: sequence length %d > NPositions %d", T, m.Cfg.NPositions)
	}

	// Token embeddings: (B, T) → (B, T, D).
	tokH, err := m.TokenEmb.Forward(tokens)
	if err != nil {
		return nil, err
	}
	// Position indices: [0, 1, …, T-1], broadcast across the batch.
	posIdx, err := makePosIdx(T, B, tokens.Device())
	if err != nil {
		return nil, err
	}
	posH, err := m.PosEmb.Forward(posIdx) // (B, T, D)
	if err != nil {
		return nil, err
	}
	h, err := autograd.Add(tokH, posH)
	if err != nil {
		return nil, err
	}
	// Transformer blocks.
	for _, block := range m.Blocks {
		h, err = block.Forward(h)
		if err != nil {
			return nil, err
		}
	}
	h, err = m.FinalNorm.Forward(h)
	if err != nil {
		return nil, err
	}
	// LM head: (B, T, D) → (B, T, V). nn.Linear auto-flattens batch dims.
	return m.LMHead.Forward(h)
}

// makePosIdx creates a (B, T) I64 tensor where each row is [0, 1, …, T-1].
// Built on CPU and transferred to `dev` if needed.
func makePosIdx(T, B int, dev device.Device) (*tensor.Tensor, error) {
	data := make([]int64, B*T)
	for b := range B {
		for t := range T {
			data[b*T+t] = int64(t)
		}
	}
	return tensor.New(data, []int{B, T}, dev)
}

// ParamCount returns the total number of trainable elements. Honors the tied
// LM head (counts the shared embedding once).
func (m *GPT2) ParamCount() int {
	total := 0
	for _, p := range m.Parameters() {
		total += p.NumElements()
	}
	return total
}

// Avoid unused-import warnings when math/unsafe aren't needed at build.
var (
	_ = math.Pi
	_ = unsafe.Pointer(nil)
)
