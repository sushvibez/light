package models

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/hub"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// T5Config is the subset of a T5 / T5 v1.1 / Flan-T5 config.json the
// encoder needs.
type T5Config struct {
	VocabSize       int     `json:"vocab_size"`
	DModel          int     `json:"d_model"`
	DKV             int     `json:"d_kv"`
	DFF             int     `json:"d_ff"`
	NumLayers       int     `json:"num_layers"`
	NumHeads        int     `json:"num_heads"`
	NumBuckets      int     `json:"relative_attention_num_buckets"`
	MaxDistance     int     `json:"relative_attention_max_distance"`
	Eps             float64 `json:"layer_norm_epsilon"`
	FeedForwardProj string  `json:"feed_forward_proj"` // "relu" (T5) or "gated-gelu" (v1.1, Flan, the T5-XXL of Flux / SD3)
}

func (c T5Config) gated() bool { return c.FeedForwardProj == "gated-gelu" }

// T5Encoder is the encoder stack of T5: pre-RMSNorm blocks of unscaled
// self-attention with a learned relative position bias (computed by the
// first block, shared by all) and a ReLU or gated-GELU (tanh) MLP. It is
// the text encoder of Flux and SD3 (T5-v1.1-XXL).
type T5Encoder struct {
	Cfg       T5Config
	Embed     *nn.Embedding
	RelBias   *nn.Embedding // (NumBuckets, NumHeads)
	Blocks    []*t5Block
	FinalNorm *nn.RMSNorm
}

type t5Block struct {
	AttnNorm, FFNorm *nn.RMSNorm
	Q, K, V, O       *nn.Linear
	Wi, Wi1, Wo      *nn.Linear // Wi1 only for gated-gelu (Wi is wi_0 there)
}

// NewT5Encoder allocates an uninitialised encoder at dt (F32 or BF16).
func NewT5Encoder(cfg T5Config, dt dtype.DType, dev device.Device) (*T5Encoder, error) {
	if cfg.MaxDistance == 0 {
		cfg.MaxDistance = 128
	}
	if cfg.Eps == 0 {
		cfg.Eps = 1e-6
	}
	emb, err := nn.NewEmbeddingUninit(cfg.VocabSize, cfg.DModel, dt, dev)
	if err != nil {
		return nil, err
	}
	rel, err := nn.NewEmbeddingUninit(cfg.NumBuckets, cfg.NumHeads, dt, dev)
	if err != nil {
		return nil, err
	}
	m := &T5Encoder{Cfg: cfg, Embed: emb, RelBias: rel}
	lin := func(in, out int) (*nn.Linear, error) { return nn.NewLinearUninit(in, out, false, dt, dev) }
	norm := func() (*nn.RMSNorm, error) { return nn.NewRMSNormWithDType(cfg.DModel, cfg.Eps, dt, dev) }
	inner := cfg.NumHeads * cfg.DKV
	for range cfg.NumLayers {
		b := &t5Block{}
		for _, p := range []**nn.RMSNorm{&b.AttnNorm, &b.FFNorm} {
			if *p, err = norm(); err != nil {
				return nil, err
			}
		}
		for _, l := range []struct {
			p       **nn.Linear
			in, out int
		}{{&b.Q, cfg.DModel, inner}, {&b.K, cfg.DModel, inner}, {&b.V, cfg.DModel, inner}, {&b.O, inner, cfg.DModel}, {&b.Wi, cfg.DModel, cfg.DFF}, {&b.Wo, cfg.DFF, cfg.DModel}} {
			if *l.p, err = lin(l.in, l.out); err != nil {
				return nil, err
			}
		}
		if cfg.gated() {
			if b.Wi1, err = lin(cfg.DModel, cfg.DFF); err != nil {
				return nil, err
			}
		}
		m.Blocks = append(m.Blocks, b)
	}
	if m.FinalNorm, err = norm(); err != nil {
		return nil, err
	}
	return m, nil
}

// t5Bucket is transformers' _relative_position_bucket (bidirectional) for
// rel = key − query, in F32 as torch computes it.
func t5Bucket(rel, numBuckets, maxDistance int) int {
	nb := numBuckets / 2
	b := 0
	if rel > 0 {
		b = nb
	}
	n := rel
	if n < 0 {
		n = -n
	}
	maxExact := nb / 2
	if n < maxExact {
		return b + n
	}
	l := float32(math.Log(float64(float32(n) / float32(maxExact))))
	v := maxExact + int(l/float32(math.Log(float64(maxDistance)/float64(maxExact)))*float32(nb-maxExact))
	return b + min(v, nb-1)
}

// positionBias is the (1, H, T, T) additive relative position bias.
func (m *T5Encoder) positionBias(T int, dev device.Device) (*tensor.Tensor, error) {
	ids := make([]int64, T*T)
	for q := range T {
		for k := range T {
			ids[q*T+k] = int64(t5Bucket(k-q, m.Cfg.NumBuckets, m.Cfg.MaxDistance))
		}
	}
	it, err := tensor.New(ids, []int{T * T}, dev)
	if err != nil {
		return nil, err
	}
	b, err := m.RelBias.Forward(it) // (T·T, H)
	if err != nil {
		return nil, err
	}
	if b, err = b.Reshape(T, T, m.Cfg.NumHeads); err != nil {
		return nil, err
	}
	if b, err = b.Permute(2, 0, 1); err != nil {
		return nil, err
	}
	if b, err = b.Contiguous(); err != nil {
		return nil, err
	}
	return b.Reshape(1, m.Cfg.NumHeads, T, T)
}

// Forward encodes ids (B, T) I64 into hidden states (B, T, DModel) at the
// model dtype (no padding mask: every position attends to every other).
func (m *T5Encoder) Forward(ids *tensor.Tensor) (*tensor.Tensor, error) {
	if ids.Rank() != 2 || ids.DType() != dtype.I64 {
		return nil, fmt.Errorf("T5Encoder: ids must be (B, T) I64, got %v %v", ids.Shape(), ids.DType())
	}
	B, T := ids.Dim(0), ids.Dim(1)
	H, DK := m.Cfg.NumHeads, m.Cfg.DKV
	dev := ids.Device()
	h, err := m.Embed.Forward(ids)
	if err != nil {
		return nil, err
	}
	bias, err := m.positionBias(T, dev)
	if err != nil {
		return nil, err
	}
	if bias.DType() != h.DType() {
		if bias, err = ops.Cast(bias, h.DType()); err != nil {
			return nil, err
		}
	}
	heads := func(x *tensor.Tensor) (*tensor.Tensor, error) {
		x, err := autograd.Reshape(x, B, T, H, DK)
		if err != nil {
			return nil, err
		}
		return autograd.Permute(x, 0, 2, 1, 3)
	}
	for i, b := range m.Blocks {
		n, err := b.AttnNorm.Forward(h)
		if err != nil {
			return nil, err
		}
		var qkv [3]*tensor.Tensor
		for j, l := range []*nn.Linear{b.Q, b.K, b.V} {
			x, err := l.Forward(n)
			if err != nil {
				return nil, err
			}
			if qkv[j], err = heads(x); err != nil {
				return nil, err
			}
		}
		ctx, err := nn.SDPAScaled(qkv[0], qkv[1], qkv[2], bias, 1)
		if err != nil {
			return nil, fmt.Errorf("T5Encoder: block %d: %w", i, err)
		}
		if ctx, err = autograd.Permute(ctx, 0, 2, 1, 3); err != nil {
			return nil, err
		}
		if ctx, err = autograd.Contiguous(ctx); err != nil {
			return nil, err
		}
		if ctx, err = autograd.Reshape(ctx, B, T, H*DK); err != nil {
			return nil, err
		}
		o, err := b.O.Forward(ctx)
		if err != nil {
			return nil, err
		}
		if h, err = ops.Add(h, o); err != nil {
			return nil, err
		}
		if n, err = b.FFNorm.Forward(h); err != nil {
			return nil, err
		}
		a, err := b.Wi.Forward(n)
		if err != nil {
			return nil, err
		}
		if m.Cfg.gated() {
			if a, err = ops.Gelu(a); err != nil { // gelu_new (tanh)
				return nil, err
			}
			g, err := b.Wi1.Forward(n)
			if err != nil {
				return nil, err
			}
			if a, err = ops.Mul(a, g); err != nil {
				return nil, err
			}
		} else if a, err = ops.Relu(a); err != nil {
			return nil, err
		}
		f, err := b.Wo.Forward(a)
		if err != nil {
			return nil, err
		}
		if h, err = ops.Add(h, f); err != nil {
			return nil, err
		}
	}
	return m.FinalNorm.Forward(h)
}

// LoadT5EncoderFromHF loads the encoder of a T5 checkpoint (an HF repo or a
// local directory: T5 / T5 v1.1 / Flan-T5, or a T5EncoderModel such as
// Flux's text_encoder_2) at dt.
func LoadT5EncoderFromHF(repo string, dev device.Device, dt dtype.DType) (*T5Encoder, error) {
	var cfgPath string
	if st, err := os.Stat(repo); err == nil && st.IsDir() {
		cfgPath = filepath.Join(repo, "config.json")
	} else if cfgPath, err = hub.Download(repo, "main", "config.json"); err != nil {
		return nil, err
	}
	raw, err := os.ReadFile(cfgPath)
	if err != nil {
		return nil, err
	}
	var cfg T5Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("T5 config: %w", err)
	}
	if cfg.FeedForwardProj == "" {
		cfg.FeedForwardProj = "relu"
	}
	if cfg.FeedForwardProj != "relu" && cfg.FeedForwardProj != "gated-gelu" {
		return nil, fmt.Errorf("T5: feed_forward_proj %q not supported (relu, gated-gelu)", cfg.FeedForwardProj)
	}
	hf, err := loadCheckpointTensors(repo)
	if err != nil {
		return nil, err
	}
	m, err := NewT5Encoder(cfg, dt, dev)
	if err != nil {
		return nil, err
	}
	emb := "encoder.embed_tokens.weight"
	if _, ok := hf[emb]; !ok {
		emb = "shared.weight"
	}
	if err := copyExactCast(m.Embed.Weight, hf, emb); err != nil {
		return nil, err
	}
	if err := copyExactCast(m.RelBias.Weight, hf, "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"); err != nil {
		return nil, err
	}
	for i, b := range m.Blocks {
		p := fmt.Sprintf("encoder.block.%d.layer.", i)
		pairs := []struct {
			dst  *tensor.Tensor
			name string
		}{
			{b.AttnNorm.Weight, p + "0.layer_norm.weight"},
			{b.Q.W, p + "0.SelfAttention.q.weight"},
			{b.K.W, p + "0.SelfAttention.k.weight"},
			{b.V.W, p + "0.SelfAttention.v.weight"},
			{b.O.W, p + "0.SelfAttention.o.weight"},
			{b.FFNorm.Weight, p + "1.layer_norm.weight"},
			{b.Wo.W, p + "1.DenseReluDense.wo.weight"},
		}
		if cfg.gated() {
			pairs = append(pairs, struct {
				dst  *tensor.Tensor
				name string
			}{b.Wi.W, p + "1.DenseReluDense.wi_0.weight"}, struct {
				dst  *tensor.Tensor
				name string
			}{b.Wi1.W, p + "1.DenseReluDense.wi_1.weight"})
		} else {
			pairs = append(pairs, struct {
				dst  *tensor.Tensor
				name string
			}{b.Wi.W, p + "1.DenseReluDense.wi.weight"})
		}
		for _, pr := range pairs {
			if err := copyExactCast(pr.dst, hf, pr.name); err != nil {
				return nil, err
			}
		}
	}
	if err := copyExactCast(m.FinalNorm.Weight, hf, "encoder.final_layer_norm.weight"); err != nil {
		return nil, err
	}
	return m, nil
}
