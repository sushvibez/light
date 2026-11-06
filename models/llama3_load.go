package models

import (
	"encoding/json"
	"fmt"
	"os"

	"light/device"
	"light/dtype"
	"light/hub"
	"light/tensor"
)

// llamaHFConfig is the subset of a LLaMA-family HuggingFace config.json that
// differs from Qwen2's: the rope_scaling block and the eos_token_id field,
// which may be an int or a list.
type llamaHFConfig struct {
	RopeScaling *hfRopeParams `json:"rope_scaling"`
	// transformers ≥ 5 writes rope settings to rope_parameters instead: either
	// one flat block, or one block per layer type ("full_attention" /
	// "sliding_attention", Mellum 2.1).
	RopeParameters json.RawMessage `json:"rope_parameters"`
	// Per-layer "dense" / "sparse" FFN (Mellum 2.1).
	MLPLayerTypes []string `json:"mlp_layer_types"`
	// Phi-3 keeps original_max_position_embeddings at the top level.
	OriginalMaxPositions int             `json:"original_max_position_embeddings"`
	MaxPositions         int             `json:"max_position_embeddings"`
	EOSTokenID           json.RawMessage `json:"eos_token_id"`
	ModelType            string          `json:"model_type"`
	// Sliding-window attention. Mistral v0.1: sliding_window=4096 (null in
	// v0.2+). Qwen2 lists sliding_window but gates it on use_sliding_window
	// and applies it only to layers >= max_window_layers. Gemma 2/3 list
	// per-layer layer_types ("sliding_attention" / "full_attention").
	SlidingWindow    *int     `json:"sliding_window"`
	UseSlidingWindow *bool    `json:"use_sliding_window"`
	MaxWindowLayers  *int     `json:"max_window_layers"`
	LayerTypes       []string `json:"layer_types"`
	NumHiddenLayers  int      `json:"num_hidden_layers"`
}

// hfRopeParams is one HF rope_scaling / rope_parameters block.
type hfRopeParams struct {
	RopeType             string    `json:"rope_type"`
	Type                 string    `json:"type"` // older configs use "type"
	RopeTheta            float64   `json:"rope_theta"`
	Factor               float64   `json:"factor"`
	LowFreqFactor        float64   `json:"low_freq_factor"`
	HighFreqFactor       float64   `json:"high_freq_factor"`
	OriginalMaxPositions int       `json:"original_max_position_embeddings"`
	ShortFactor          []float64 `json:"short_factor"` // longrope
	LongFactor           []float64 `json:"long_factor"`  // longrope
	BetaFast             float64   `json:"beta_fast"`    // yarn
	BetaSlow             float64   `json:"beta_slow"`
	AttentionFactor      float64   `json:"attention_factor"`
	MScale               float64   `json:"mscale"`
	MScaleAllDim         float64   `json:"mscale_all_dim"`
}

// scaling converts the block to a RopeScaling (nil for plain RoPE).
// origFallback / maxPositions are the config's top-level
// original_max_position_embeddings and max_position_embeddings.
func (p *hfRopeParams) scaling(origFallback, maxPositions int) (*RopeScaling, error) {
	if p == nil {
		return nil, nil
	}
	typ := p.RopeType
	if typ == "" {
		typ = p.Type
	}
	orig := p.OriginalMaxPositions
	if orig == 0 {
		orig = origFallback
	}
	switch typ {
	case "llama3":
		return &RopeScaling{
			Type: "llama3", Factor: p.Factor,
			LowFreqFactor: p.LowFreqFactor, HighFreqFactor: p.HighFreqFactor,
			OriginalMaxPositions: p.OriginalMaxPositions,
		}, nil
	case "longrope":
		return &RopeScaling{
			Type: "longrope", ShortFactor: p.ShortFactor, LongFactor: p.LongFactor,
			OriginalMaxPositions: orig, MaxPositions: maxPositions,
		}, nil
	case "yarn":
		return &RopeScaling{
			Type: "yarn", Factor: p.Factor, OriginalMaxPositions: orig, MaxPositions: maxPositions,
			BetaFast: p.BetaFast, BetaSlow: p.BetaSlow, AttnFactor: p.AttentionFactor,
			MScale: p.MScale, MScaleAllDim: p.MScaleAllDim,
		}, nil
	case "", "default":
		return nil, nil
	default:
		return nil, fmt.Errorf("rope type %q not supported", typ)
	}
}

// ropeParameters splits rope_parameters into the full-attention (or only)
// block and the sliding-attention block, which is nil unless the config
// keys its rope settings by layer type.
func (hf *llamaHFConfig) ropeParameters() (full, sliding *hfRopeParams, err error) {
	if len(hf.RopeParameters) == 0 || string(hf.RopeParameters) == "null" {
		return nil, nil, nil
	}
	var byType map[string]json.RawMessage
	if err := json.Unmarshal(hf.RopeParameters, &byType); err != nil {
		return nil, nil, fmt.Errorf("rope_parameters: %w", err)
	}
	_, hasFull := byType["full_attention"]
	_, hasSliding := byType["sliding_attention"]
	if !hasFull && !hasSliding {
		var flat hfRopeParams
		if err := json.Unmarshal(hf.RopeParameters, &flat); err != nil {
			return nil, nil, fmt.Errorf("rope_parameters: %w", err)
		}
		return &flat, nil, nil
	}
	parse := func(key string) (*hfRopeParams, error) {
		raw, ok := byType[key]
		if !ok {
			return nil, nil
		}
		var p hfRopeParams
		if err := json.Unmarshal(raw, &p); err != nil {
			return nil, fmt.Errorf("rope_parameters.%s: %w", key, err)
		}
		return &p, nil
	}
	if full, err = parse("full_attention"); err != nil {
		return nil, nil, err
	}
	if sliding, err = parse("sliding_attention"); err != nil {
		return nil, nil, err
	}
	return full, sliding, nil
}

// slidingWindowConfig derives (SlidingWindow, SlidingLayers) from the HF
// config using the model-family conventions documented on llamaHFConfig.
func (hf *llamaHFConfig) slidingWindowConfig() (int, []bool) {
	if hf.SlidingWindow == nil || *hf.SlidingWindow <= 0 {
		return 0, nil
	}
	if hf.UseSlidingWindow != nil && !*hf.UseSlidingWindow {
		return 0, nil
	}
	w := *hf.SlidingWindow
	if len(hf.LayerTypes) > 0 {
		layers := make([]bool, len(hf.LayerTypes))
		any := false
		for i, t := range hf.LayerTypes {
			layers[i] = t == "sliding_attention"
			any = any || layers[i]
		}
		if !any {
			return 0, nil
		}
		return w, layers
	}
	if hf.MaxWindowLayers != nil && hf.NumHiddenLayers > 0 && *hf.MaxWindowLayers > 0 {
		if *hf.MaxWindowLayers >= hf.NumHiddenLayers {
			return 0, nil
		}
		layers := make([]bool, hf.NumHiddenLayers)
		for i := range layers {
			layers[i] = i >= *hf.MaxWindowLayers
		}
		return w, layers
	}
	return w, nil
}

// LoadLlama3FromHF downloads (or reads from the HF cache) a LLaMA-3.x
// checkpoint such as `meta-llama/Llama-3.2-1B-Instruct` and loads it into
// a Llama with F32 weights. Llama-3 RoPE frequency scaling is applied when
// the config carries a `rope_scaling` block.
func LoadLlama3FromHF(repo string, dev device.Device) (*Llama, error) {
	return loadLlama3(repo, dev, dtype.F32)
}

// LoadLlama3FromHFMixed is LoadLlama3FromHF with BF16 weights (half the
// GPU memory; matmuls on tensor cores with F32 accumulate).
func LoadLlama3FromHFMixed(repo string, dev device.Device) (*Llama, error) {
	return loadLlama3(repo, dev, dtype.BF16)
}

func loadLlama3(repo string, dev device.Device, weightDT dtype.DType) (*Llama, error) {
	cfg, err := fetchLlamaConfig(repo)
	if err != nil {
		return nil, err
	}
	cfg.WeightDType = weightDT
	cfg.SkipInit = true
	tensors, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, fmt.Errorf("LoadLlama3FromHF: %w", err)
	}
	llama, err := NewLlama(cfg, dev)
	if err != nil {
		return nil, err
	}
	if err := mapHFToLlama(llama, tensors); err != nil {
		return nil, err
	}
	return llama, nil
}

// fetchLlamaConfig reads config.json for any LLaMA-architecture repo
// (llama, qwen2, mistral, ...) and returns a LlamaConfig including RoPE
// scaling. It reuses the Qwen2 field mapping for everything else.
func fetchLlamaConfig(repo string) (LlamaConfig, error) {
	cfg, err := fetchQwenConfig(repo)
	if err != nil {
		return LlamaConfig{}, err
	}
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return LlamaConfig{}, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return LlamaConfig{}, err
	}
	var hf llamaHFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return LlamaConfig{}, fmt.Errorf("fetchLlamaConfig: parse: %w", err)
	}
	cfg.SlidingWindow, cfg.SlidingLayers = hf.slidingWindowConfig()
	if cfg.RopeScaling, err = hf.RopeScaling.scaling(hf.OriginalMaxPositions, hf.MaxPositions); err != nil {
		return LlamaConfig{}, fmt.Errorf("fetchLlamaConfig: rope_scaling: %w", err)
	}
	full, sliding, err := hf.ropeParameters()
	if err != nil {
		return LlamaConfig{}, fmt.Errorf("fetchLlamaConfig: %w", err)
	}
	if full != nil {
		if full.RopeTheta > 0 {
			cfg.RopeBase = full.RopeTheta
		}
		if cfg.RopeScaling == nil {
			if cfg.RopeScaling, err = full.scaling(hf.OriginalMaxPositions, hf.MaxPositions); err != nil {
				return LlamaConfig{}, fmt.Errorf("fetchLlamaConfig: rope_parameters: %w", err)
			}
		}
	}
	if sliding != nil && cfg.SlidingWindow > 0 {
		cfg.SlidingRopeBase = sliding.RopeTheta
		if cfg.SlidingRopeBase == 0 {
			cfg.SlidingRopeBase = cfg.RopeBase
		}
		if cfg.SlidingRopeScaling, err = sliding.scaling(hf.OriginalMaxPositions, hf.MaxPositions); err != nil {
			return LlamaConfig{}, fmt.Errorf("fetchLlamaConfig: rope_parameters.sliding_attention: %w", err)
		}
	}
	// mlp_layer_types: "dense" layers keep the SwiGLU MLP in an MoE model.
	for i, t := range hf.MLPLayerTypes {
		if t == "dense" {
			cfg.MLPOnlyLayers = append(cfg.MLPOnlyLayers, i)
		}
	}
	return cfg, nil
}

// LoadPhi3FromHF loads a Phi-3 / Phi-3.5 checkpoint (e.g.
// `microsoft/Phi-3.5-mini-instruct`). Phi-3 is the Llama architecture with
// fused `qkv_proj` / `gate_up_proj` weights and LongRoPE scaling; the fused
// weights are split on load so the model is a plain *Llama. weightDT is
// dtype.F32 or dtype.BF16 (3.8B params: 15 GB vs 7.6 GB).
func LoadPhi3FromHF(repo string, dev device.Device, weightDT dtype.DType) (*Llama, error) {
	cfg, err := fetchLlamaConfig(repo)
	if err != nil {
		return nil, err
	}
	cfg.WeightDType = weightDT
	cfg.SkipInit = true
	tensors, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, fmt.Errorf("LoadPhi3FromHF: %w", err)
	}
	if err := splitPhi3Fused(tensors, cfg); err != nil {
		return nil, err
	}
	llama, err := NewLlama(cfg, dev)
	if err != nil {
		return nil, err
	}
	if err := mapHFToLlama(llama, tensors); err != nil {
		return nil, err
	}
	return llama, nil
}

// splitPhi3Fused rewrites Phi-3's fused projections into the standard Llama
// keys: `self_attn.qkv_proj.weight` (Q·HD + 2·KH·HD, D) → q/k/v rows in that
// order, and `mlp.gate_up_proj.weight` (2·IM, D) → gate rows then up rows
// (Phi3MLP: `gate, up = up_states.chunk(2, dim=-1)`). Rows are narrowed and
// contiguified on CPU; the fused entries are dropped.
func splitPhi3Fused(hf map[string]*tensor.Tensor, cfg LlamaConfig) error {
	qRows := cfg.NumHeads * cfg.HeadDim
	kvRows := cfg.NumKVHeads * cfg.HeadDim
	split := func(fused string, parts []string, rows []int) error {
		t, ok := hf[fused]
		if !ok {
			return nil // already in split form
		}
		total := 0
		for _, r := range rows {
			total += r
		}
		if t.Rank() != 2 || t.Dim(0) != total {
			return fmt.Errorf("%s: shape %v, want (%d, D)", fused, t.Shape(), total)
		}
		off := 0
		for i, name := range parts {
			v, err := t.Narrow(0, off, rows[i])
			if err != nil {
				return fmt.Errorf("%s: %w", fused, err)
			}
			c, err := v.Contiguous()
			if err != nil {
				return fmt.Errorf("%s: %w", fused, err)
			}
			hf[name] = c
			off += rows[i]
		}
		delete(hf, fused)
		return nil
	}
	for i := 0; i < cfg.NumLayers; i++ {
		p := fmt.Sprintf("model.layers.%d.", i)
		if err := split(p+"self_attn.qkv_proj.weight",
			[]string{p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"},
			[]int{qRows, kvRows, kvRows}); err != nil {
			return err
		}
		if err := split(p+"mlp.gate_up_proj.weight",
			[]string{p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"},
			[]int{cfg.IntermediateSize, cfg.IntermediateSize}); err != nil {
			return err
		}
	}
	return nil
}

// LoadCausalLMFromHF picks the loader from config.json's model_type and
// returns the model: "llama" (Llama 2/3.x, incl. Llama-3 RoPE scaling),
// "qwen2"/"qwen2_5", "mistral" all share the Llama architecture. weightDT is
// dtype.F32 or dtype.BF16.
func LoadCausalLMFromHF(repo string, dev device.Device, weightDT dtype.DType) (*Llama, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return nil, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var hf llamaHFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return nil, fmt.Errorf("LoadCausalLMFromHF: parse config: %w", err)
	}
	switch hf.ModelType {
	case "llama", "mistral":
		return loadLlama3(repo, dev, weightDT)
	case "qwen2", "qwen2_5", "qwen3", "qwen3_moe", "mellum", "sdar":
		return loadQwen2(repo, dev, weightDT)
	case "phi3":
		return LoadPhi3FromHF(repo, dev, weightDT)
	default:
		return nil, fmt.Errorf("LoadCausalLMFromHF: model_type %q not supported (llama, mistral, qwen2, qwen3, qwen3_moe, mellum, sdar, phi3)", hf.ModelType)
	}
}

// EOSTokenIDs returns the end-of-sequence token ids for a repo, from
// generation_config.json when present (Llama-3 lists several: end_of_text,
// eom, eot) and config.json otherwise. Both accept an int or a list.
func EOSTokenIDs(repo string) ([]int64, error) {
	var ids []int64
	if p, err := hub.Download(repo, "main", "generation_config.json"); err == nil {
		if got, err := eosFromJSON(p); err == nil && len(got) > 0 {
			ids = got
		}
	}
	if len(ids) == 0 {
		p, err := hub.Download(repo, "main", "config.json")
		if err != nil {
			return nil, err
		}
		got, err := eosFromJSON(p)
		if err != nil {
			return nil, err
		}
		ids = got
	}
	return ids, nil
}

// BOSTokenID returns the token the HF tokenizer prepends to every encoding,
// if any. The authoritative signal is tokenizer.json's post_processor: a
// TemplateProcessing whose `single` template starts with a SpecialToken
// (Llama-3: <|begin_of_text|> = 128000). config.json's bos_token_id is NOT
// used — Qwen2 sets it to <|endoftext|> yet its tokenizer adds nothing, and
// prepending it degrades generation.
func BOSTokenID(repo string) (int64, bool) {
	p, err := hub.Download(repo, "main", "tokenizer.json")
	if err != nil {
		return 0, false
	}
	raw, err := os.ReadFile(p)
	if err != nil {
		return 0, false
	}
	var tj struct {
		AddedTokens []struct {
			ID      int64  `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
		Model struct {
			Vocab map[string]int64 `json:"vocab"`
		} `json:"model"`
		PostProcessor json.RawMessage `json:"post_processor"`
	}
	if err := json.Unmarshal(raw, &tj); err != nil || len(tj.PostProcessor) == 0 {
		return 0, false
	}
	name, ok := leadingSpecialToken(tj.PostProcessor)
	if !ok {
		return 0, false
	}
	for _, at := range tj.AddedTokens {
		if at.Content == name {
			return at.ID, true
		}
	}
	if id, ok := tj.Model.Vocab[name]; ok {
		return id, true
	}
	return 0, false
}

// leadingSpecialToken walks a post_processor block (TemplateProcessing, or a
// Sequence containing one) and returns the special token that opens the
// `single` template, if the template starts with one.
func leadingSpecialToken(raw json.RawMessage) (string, bool) {
	var pp struct {
		Type       string            `json:"type"`
		Processors []json.RawMessage `json:"processors"`
		Single     []struct {
			SpecialToken *struct {
				ID string `json:"id"`
			} `json:"SpecialToken"`
		} `json:"single"`
	}
	if err := json.Unmarshal(raw, &pp); err != nil {
		return "", false
	}
	switch pp.Type {
	case "TemplateProcessing":
		if len(pp.Single) > 0 && pp.Single[0].SpecialToken != nil {
			return pp.Single[0].SpecialToken.ID, true
		}
	case "Sequence":
		for _, sub := range pp.Processors {
			if name, ok := leadingSpecialToken(sub); ok {
				return name, true
			}
		}
	}
	return "", false
}

func eosFromJSON(path string) ([]int64, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var hf struct {
		EOSTokenID json.RawMessage `json:"eos_token_id"`
	}
	if err := json.Unmarshal(raw, &hf); err != nil {
		return nil, err
	}
	if len(hf.EOSTokenID) == 0 {
		return nil, nil
	}
	var one int64
	if err := json.Unmarshal(hf.EOSTokenID, &one); err == nil {
		return []int64{one}, nil
	}
	var many []int64
	if err := json.Unmarshal(hf.EOSTokenID, &many); err != nil {
		return nil, fmt.Errorf("eos_token_id: %w", err)
	}
	return many, nil
}

// HFModelType returns config.json's model_type for a Hub repo.
func HFModelType(repo string) (string, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return "", err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	var hf struct {
		ModelType string `json:"model_type"`
	}
	if err := json.Unmarshal(raw, &hf); err != nil {
		return "", err
	}
	return hf.ModelType, nil
}
