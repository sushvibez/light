package models

import (
	"fmt"

	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// MTPHead is MiMo-7B's multi-token-prediction layer (model.mtp_layers.0,
// transformers-free reading of the repo's modeling_mimo.MiMoMTPLayers): for
// position t it combines the main model's final hidden state h_t with the
// embedding of token t+1 and predicts token t+2,
//
//	x = input_proj([hidden_norm(h_t), token_norm(emb(t+1))])
//	x = decoder layer (attention with its own KV cache, MLP)
//	logits = lm_head(final_norm(x))   (embedding and LM head shared)
//
// It is used as a draft head for greedy speculative decoding; the main
// model verifies every draft, so the output never depends on it.
type MTPHead struct {
	TokenNorm, HiddenNorm, FinalNorm *nn.RMSNorm
	InputProj                        *nn.Linear
	Block                            *llamaBlock
}

// MTPDraft runs the draft head on main-model hidden states (1, T, D) and
// the T tokens that follow them, appending to its own one-layer cache, and
// returns F32 logits (1, T, V) — row i predicts the token after tokens[i].
func (m *Llama) MTPDraft(hidden *tensor.Tensor, tokens []int64, cache *KVCache) (*tensor.Tensor, error) {
	if m.MTP == nil {
		return nil, fmt.Errorf("Llama.MTPDraft: model has no MTP head")
	}
	T := len(tokens)
	if hidden.Rank() != 3 || hidden.Dim(0) != 1 || hidden.Dim(1) != T {
		return nil, fmt.Errorf("Llama.MTPDraft: hidden %v for %d tokens", hidden.Shape(), T)
	}
	dev := hidden.Device()
	ids, err := tensor.New(tokens, []int{1, T}, dev)
	if err != nil {
		return nil, err
	}
	emb, err := m.TokenEmb.Forward(ids)
	if err != nil {
		return nil, err
	}
	if emb, err = m.MTP.TokenNorm.Forward(emb); err != nil {
		return nil, err
	}
	h, err := m.MTP.HiddenNorm.Forward(hidden)
	if err != nil {
		return nil, err
	}
	if h.DType() != emb.DType() {
		if h, err = ops.Cast(h, emb.DType()); err != nil {
			return nil, err
		}
	}
	x, err := ops.Concat([]*tensor.Tensor{h, emb}, 2)
	if err != nil {
		return nil, err
	}
	if x, err = m.MTP.InputProj.Forward(x); err != nil {
		return nil, err
	}
	prior := cache.Len
	if x, err = m.runBlock(m.MTP.Block, x, 1, T, prior, dev, cache, 0); err != nil {
		return nil, err
	}
	cache.Len = prior + T
	if x, err = m.MTP.FinalNorm.Forward(x); err != nil {
		return nil, err
	}
	logits, err := m.LMHead.Forward(x)
	if err != nil {
		return nil, err
	}
	if logits.DType() != dtype.F32 {
		return ops.Cast(logits, dtype.F32)
	}
	return logits, nil
}

// loadMTPHead builds the MTP head from `prefix` (e.g. "model.mtp_layers.0.")
// when the checkpoint has one; it returns nil, nil otherwise.
func loadMTPHead(m *Llama, hf map[string]*tensor.Tensor, prefix string) (*MTPHead, error) {
	if _, ok := hf[prefix+"input_proj.weight"]; !ok {
		return nil, nil
	}
	cfg := m.Cfg
	cfg.NumExperts = 0 // the MTP layer has a dense MLP
	dev := m.TokenEmb.Weight.Device()
	wdt := cfg.WeightDType
	if wdt != dtype.BF16 {
		wdt = dtype.F32
	}
	cfg.LazyLinear = false
	cfg.SkipInit = true
	blk, err := newLlamaBlock(cfg, 0, dev)
	if err != nil {
		return nil, err
	}
	norm := func() (*nn.RMSNorm, error) { return nn.NewRMSNormWithDType(cfg.HiddenSize, cfg.RmsEps, wdt, dev) }
	h := &MTPHead{Block: blk}
	if h.TokenNorm, err = norm(); err != nil {
		return nil, err
	}
	if h.HiddenNorm, err = norm(); err != nil {
		return nil, err
	}
	if h.FinalNorm, err = norm(); err != nil {
		return nil, err
	}
	if h.InputProj, err = nn.NewLinearUninit(2*cfg.HiddenSize, cfg.HiddenSize, false, wdt, dev); err != nil {
		return nil, err
	}
	pairs := []struct {
		dst  *tensor.Tensor
		name string
	}{
		{h.TokenNorm.Weight, "token_layernorm.weight"}, {h.HiddenNorm.Weight, "hidden_layernorm.weight"},
		{h.FinalNorm.Weight, "final_layernorm.weight"}, {h.InputProj.W, "input_proj.weight"},
		{blk.N1.Weight, "input_layernorm.weight"}, {blk.N2.Weight, "post_attention_layernorm.weight"},
		{blk.QProj.W, "self_attn.q_proj.weight"}, {blk.KProj.W, "self_attn.k_proj.weight"},
		{blk.VProj.W, "self_attn.v_proj.weight"}, {blk.OProj.W, "self_attn.o_proj.weight"},
		{blk.GateProj.W, "mlp.gate_proj.weight"}, {blk.UpProj.W, "mlp.up_proj.weight"},
		{blk.DownProj.W, "mlp.down_proj.weight"},
	}
	if cfg.QKVBias {
		pairs = append(pairs, struct {
			dst  *tensor.Tensor
			name string
		}{blk.QProj.B, "self_attn.q_proj.bias"}, struct {
			dst  *tensor.Tensor
			name string
		}{blk.KProj.B, "self_attn.k_proj.bias"}, struct {
			dst  *tensor.Tensor
			name string
		}{blk.VProj.B, "self_attn.v_proj.bias"})
	}
	for _, p := range pairs {
		if err := copyExactCast(p.dst, hf, prefix+p.name); err != nil {
			return nil, fmt.Errorf("MTP head: %w", err)
		}
	}
	return h, nil
}
