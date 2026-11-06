package models

import (
	"fmt"
	"os"
	"path/filepath"

	"light/device"
	"light/gguf"
	"light/hub"
	"light/nn"
)

// MoshikoRepo holds the Q8_0 GGUF Candle uses for Moshi v0.1 (8.2 GB) and
// the SentencePiece text tokenizer.
const (
	MoshikoRepo      = "kyutai/moshiko-candle-q8"
	MoshikoGGUF      = "model.q8.gguf"
	MoshikoTokenizer = "tokenizer_spm_32k_3.model"
)

// LoadMoshiFromHF downloads (or finds in the HF cache) the GGUF of repo and
// loads it. The tokenizer path is returned alongside for
// tokenizer.LoadSentencePieceModel.
func LoadMoshiFromHF(repo string, dev device.Device) (*Moshi, string, error) {
	if repo == "" {
		repo = MoshikoRepo
	}
	path, err := hub.Download(repo, "", MoshikoGGUF)
	if err != nil {
		return nil, "", err
	}
	tok, err := hub.Download(repo, "", MoshikoTokenizer)
	if err != nil {
		return nil, "", err
	}
	m, err := LoadMoshiGGUF(path, dev)
	return m, tok, err
}

// LoadMoshiGGUF loads a kyutai moshi-core GGUF (candle quantized_var_builder
// naming: transformer.layers.N.self_attn.in_proj_weight, …gating.linear_in,
// normN.alpha, emb.K, text_emb, text_linear, out_norm, depformer.S.*).
// Linear and embedding weights stay in their block format (Q8_0 → fused
// MatMulQ on CUDA); norms are F32. Layer, codebook and slice counts are
// taken from the file; widths from DefaultMoshiConfig.
func LoadMoshiGGUF(path string, dev device.Device) (*Moshi, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fh.Close()
	g, err := gguf.Open(fh)
	if err != nil {
		return nil, fmt.Errorf("LoadMoshiGGUF: %w", err)
	}
	cfg := DefaultMoshiConfig()
	count := func(format string) int {
		n := 0
		for {
			if _, ok := g.Tensors[fmt.Sprintf(format, n)]; !ok {
				return n
			}
			n++
		}
	}
	cfg.Transformer.NumLayers = count("transformer.layers.%d.norm1.alpha")
	cfg.AudioCodebooks = count("emb.%d.weight")
	cfg.DepQ = count("depformer.%d.linear_in.weight")
	cfg.Depformer.NumLayers = count("depformer.0.transformer.layers.%d.norm1.alpha")
	cfg.Depformer.Context = cfg.DepQ
	if cfg.Transformer.NumLayers == 0 || cfg.AudioCodebooks == 0 {
		return nil, fmt.Errorf("LoadMoshiGGUF: %s does not look like a moshi-core checkpoint", filepath.Base(path))
	}
	if info, ok := g.Tensors["transformer.layers.0.gating.linear_out.weight"]; ok {
		cfg.Transformer.Hidden = info.Dims[0] // GGUF dims are innermost-first: (K, N)
	}
	if info, ok := g.Tensors["depformer.0.transformer.layers.0.gating.linear_out.weight"]; ok {
		cfg.Depformer.Hidden = info.Dims[0]
	}
	if info, ok := g.Tensors["text_emb.weight"]; ok {
		cfg.TextInVocab, cfg.Transformer.Dim = info.Dims[1], info.Dims[0]
	}
	if info, ok := g.Tensors["text_linear.weight"]; ok {
		cfg.TextOutVocab = info.Dims[1]
	}
	if info, ok := g.Tensors["emb.0.weight"]; ok {
		cfg.AudioVocab = info.Dims[1]
	}
	if info, ok := g.Tensors["depformer.0.linear_in.weight"]; ok {
		cfg.Depformer.Dim = info.Dims[1]
	}
	m, err := NewMoshi(cfg, dev)
	if err != nil {
		return nil, err
	}
	lin := func(name string, l *nn.Linear) error {
		w, err := loadGGUFLinearWeight(g, name, dev)
		if err != nil {
			return err
		}
		if err := l.SetWeight(w); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		return nil
	}
	norm := func(name string, n *nn.RMSNorm) error {
		t, err := g.TensorRowMajor(name, dev)
		if err != nil {
			return err
		}
		if t, err = t.Reshape(t.NumElements()); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		if !t.Shape().Equal(n.Weight.Shape()) || t.DType() != n.Weight.DType() {
			return fmt.Errorf("%s: got %v %v, want %v F32", name, t.Shape(), t.DType(), n.Weight.Shape())
		}
		n.Weight = t
		return nil
	}
	emb := func(name string, e *nn.Embedding) error {
		w, err := loadGGUFLinearWeight(g, name, dev)
		if err != nil {
			return err
		}
		if !w.Shape().Equal(e.Weight.Shape()) {
			return fmt.Errorf("%s: got %v, want %v", name, w.Shape(), e.Weight.Shape())
		}
		e.Weight = w
		return nil
	}
	transformer := func(p string, t *moshiTransformer) error {
		for i, l := range t.Layers {
			lp := fmt.Sprintf("%slayers.%d.", p, i)
			for name, lin_ := range map[string]*nn.Linear{
				"self_attn.in_proj_weight": l.InProj, "self_attn.out_proj.weight": l.OutProj,
				"gating.linear_in.weight": l.GateIn, "gating.linear_out.weight": l.GateOut,
			} {
				if err := lin(lp+name, lin_); err != nil {
					return err
				}
			}
			if err := norm(lp+"norm1.alpha", l.Norm1); err != nil {
				return err
			}
			if err := norm(lp+"norm2.alpha", l.Norm2); err != nil {
				return err
			}
		}
		return nil
	}
	if err := transformer("transformer.", m.Transformer); err != nil {
		return nil, err
	}
	if err := emb("text_emb.weight", m.TextEmb); err != nil {
		return nil, err
	}
	for k, e := range m.AudioEmb {
		if err := emb(fmt.Sprintf("emb.%d.weight", k), e); err != nil {
			return nil, err
		}
	}
	if err := norm("out_norm.alpha", m.OutNorm); err != nil {
		return nil, err
	}
	if err := lin("text_linear.weight", m.TextLinear); err != nil {
		return nil, err
	}
	for s, sl := range m.Depformer {
		p := fmt.Sprintf("depformer.%d.", s)
		if err := transformer(p+"transformer.", sl.Transformer); err != nil {
			return nil, err
		}
		if err := emb(p+"emb.weight", sl.Emb); err != nil {
			return nil, err
		}
		if err := lin(p+"linear_in.weight", sl.LinearIn); err != nil {
			return nil, err
		}
		if err := lin(p+"linear_out.weight", sl.LinearOut); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// moshiGGUFPath returns the cached moshiko GGUF, or "" when absent.
func moshiGGUFPath() string {
	home, _ := os.UserHomeDir()
	m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--kyutai--moshiko-candle-q8/snapshots/*", MoshikoGGUF))
	if len(m) == 0 {
		return ""
	}
	if st, err := os.Stat(m[0]); err != nil || st.Size() == 0 {
		return ""
	}
	return m[0]
}
