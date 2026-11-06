package models

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"

	"light/device"
	"light/dtype"
	"light/hub"
	"light/nn"
	"light/ops"
	"light/safetensors"
	"light/tensor"
)

// loadSafetensorsAuto downloads either the single-file or sharded layout for
// a HF repo and returns the merged tensor map. Tensors are always loaded to
// CPU regardless of `dev`; the per-parameter copy step (copyExactCast) is
// what casts BF16→F32 and transfers to the target device. Loading to CPU
// avoids ever staging both the source dtype (BF16, ~2N bytes) and the F32
// model (~4N bytes) on the GPU at the same time, which OOMs even on a 16 GB
// card for 3B+ params.
func loadSafetensorsAuto(repo string) (map[string]*tensor.Tensor, error) {
	paths, err := hub.DownloadModelSafetensors(repo, "main")
	if err != nil {
		return nil, err
	}
	tensors, _, err := safetensors.LoadFiles(paths, device.CPU())
	return tensors, err
}

// qwenHFConfig is the subset of Qwen2's HuggingFace `config.json` we map.
type qwenHFConfig struct {
	VocabSize         int     `json:"vocab_size"`
	HiddenSize        int     `json:"hidden_size"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	HeadDim           int     `json:"head_dim"` // optional; defaults to hidden/heads
	MaxPositionEmbed  int     `json:"max_position_embeddings"`
	RopeTheta         float64 `json:"rope_theta"`
	RmsNormEps        float64 `json:"rms_norm_eps"`
	TieWordEmbed      bool    `json:"tie_word_embeddings"`
	PartialRotary     float64 `json:"partial_rotary_factor"` // Phi-4-mini 0.75
	AttentionBias     bool    `json:"attention_bias"`        // LLaMA-style flag
	ModelType         string  `json:"model_type"`
	// Qwen3-MoE
	NumExperts          int   `json:"num_experts"`
	NumExpertsPerTok    int   `json:"num_experts_per_tok"`
	MoEIntermediateSize int   `json:"moe_intermediate_size"`
	NormTopKProb        bool  `json:"norm_topk_prob"`
	DecoderSparseStep   int   `json:"decoder_sparse_step"`
	MLPOnlyLayers       []int `json:"mlp_only_layers"`
	RopeParameters      *struct {
		RopeTheta float64 `json:"rope_theta"`
	} `json:"rope_parameters"` // transformers ≥ 5 nests rope_theta here
	TextConfig json.RawMessage `json:"text_config"` // multimodal wrappers (qwen3_asr) nest the decoder config
}

// LoadQwen2FromHF downloads a pretrained Qwen2 model (e.g. `Qwen/Qwen2-0.5B-Instruct`,
// `Qwen/Qwen2.5-0.5B-Instruct`) and copies its weights into a fresh Llama
// at F32 weight dtype. The first run downloads ~1 GB into
// `~/.cache/huggingface/hub/`; subsequent runs reuse it.
func LoadQwen2FromHF(repo string, dev device.Device) (*Llama, error) {
	return loadQwen2(repo, dev, dtype.F32)
}

// LoadQwen2FromHFMixed loads Qwen2 with Linear / embedding weights kept at
// BF16 — half the GPU memory of the F32 path. Activations stay F32, so
// every non-matmul op (RMSNorm, softmax, RoPE, residuals) runs at full
// precision; only the GEMMs go through cublasGemmEx with BF16 inputs and
// F32 accumulate.
//
// On a 16 GB GPU this drops 3B-param models from ~12 GB to ~6 GB and lets
// 7B-class models fit too.
func LoadQwen2FromHFMixed(repo string, dev device.Device) (*Llama, error) {
	return loadQwen2(repo, dev, dtype.BF16)
}

func loadQwen2(repo string, dev device.Device, weightDT dtype.DType) (*Llama, error) {
	cfg, err := fetchLlamaConfig(repo)
	if err != nil {
		return nil, err
	}
	cfg.WeightDType = weightDT
	cfg.SkipInit = true // we're about to overwrite every weight from disk
	tensors, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, fmt.Errorf("LoadQwen2FromHF: %w", err)
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

func fetchQwenConfig(repo string) (LlamaConfig, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return LlamaConfig{}, fmt.Errorf("fetchQwenConfig: %w", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return LlamaConfig{}, err
	}
	var hf qwenHFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return LlamaConfig{}, fmt.Errorf("fetchQwenConfig: parse: %w", err)
	}
	if hf.HiddenSize == 0 && len(hf.TextConfig) > 0 {
		if err := json.Unmarshal(hf.TextConfig, &hf); err != nil {
			return LlamaConfig{}, fmt.Errorf("fetchQwenConfig: parse text_config: %w", err)
		}
	}
	if hf.RopeTheta == 0 && hf.RopeParameters != nil {
		hf.RopeTheta = hf.RopeParameters.RopeTheta
	}
	if hf.RmsNormEps == 0 {
		hf.RmsNormEps = 1e-6
	}
	if hf.RopeTheta == 0 {
		hf.RopeTheta = 10000
	}
	if hf.NumKeyValueHeads == 0 {
		hf.NumKeyValueHeads = hf.NumAttentionHeads
	}
	headDim := hf.HeadDim
	if headDim == 0 {
		headDim = hf.HiddenSize / hf.NumAttentionHeads
	}
	// Qwen2 always carries Q/K/V bias regardless of the `attention_bias` flag;
	// LLaMA/Mistral do not. We pick by model_type.
	qkvBias := hf.AttentionBias
	if hf.ModelType == "qwen2" || hf.ModelType == "qwen2_5" {
		qkvBias = true
	}
	return LlamaConfig{
		VocabSize:           hf.VocabSize,
		HiddenSize:          hf.HiddenSize,
		IntermediateSize:    hf.IntermediateSize,
		NumLayers:           hf.NumHiddenLayers,
		NumHeads:            hf.NumAttentionHeads,
		NumKVHeads:          hf.NumKeyValueHeads,
		HeadDim:             headDim,
		MaxPosition:         hf.MaxPositionEmbed,
		RopeBase:            hf.RopeTheta,
		RmsEps:              hf.RmsNormEps,
		TieEmbed:            hf.TieWordEmbed,
		PartialRotary:       hf.PartialRotary,
		QKVBias:             qkvBias,
		QKNorm:              hf.ModelType == "qwen3" || hf.ModelType == "qwen3_moe" || hf.ModelType == "sdar",
		NumExperts:          hf.NumExperts,
		NumExpertsPerTok:    hf.NumExpertsPerTok,
		MoEIntermediateSize: hf.MoEIntermediateSize,
		NormTopKProb:        hf.NormTopKProb,
		DecoderSparseStep:   hf.DecoderSparseStep,
		MLPOnlyLayers:       hf.MLPOnlyLayers,
	}, nil
}

func mapHFToLlama(m *Llama, hf map[string]*tensor.Tensor) error {
	return mapHFToLlamaPrefix(m, hf, "model.")
}

// mapHFToLlamaPrefix copies a HF Llama-family decoder whose tensors start
// with prefix ("model." for a plain LM, "model.language_model." inside a
// multimodal wrapper such as Qwen3-ASR). The LM head is read from
// "lm_head.weight" when untied.
func mapHFToLlamaPrefix(m *Llama, hf map[string]*tensor.Tensor, prefix string) error {
	if err := copyExactCast(m.TokenEmb.Weight, hf, prefix+"embed_tokens.weight"); err != nil {
		return err
	}
	for i, blk := range m.Blocks {
		p := prefix + "layers." + strconv.Itoa(i) + "."
		if err := copyExactCast(blk.N1.Weight, hf, p+"input_layernorm.weight"); err != nil {
			return err
		}
		if err := copyExactCast(blk.QProj.W, hf, p+"self_attn.q_proj.weight"); err != nil {
			return err
		}
		if err := copyExactCast(blk.KProj.W, hf, p+"self_attn.k_proj.weight"); err != nil {
			return err
		}
		if err := copyExactCast(blk.VProj.W, hf, p+"self_attn.v_proj.weight"); err != nil {
			return err
		}
		if err := copyExactCast(blk.OProj.W, hf, p+"self_attn.o_proj.weight"); err != nil {
			return err
		}
		if m.Cfg.QKVBias {
			if err := copyExactCast(blk.QProj.B, hf, p+"self_attn.q_proj.bias"); err != nil {
				return err
			}
			if err := copyExactCast(blk.KProj.B, hf, p+"self_attn.k_proj.bias"); err != nil {
				return err
			}
			if err := copyExactCast(blk.VProj.B, hf, p+"self_attn.v_proj.bias"); err != nil {
				return err
			}
		}
		if blk.QNorm != nil {
			if err := copyExactCast(blk.QNorm.Weight, hf, p+"self_attn.q_norm.weight"); err != nil {
				return err
			}
			if err := copyExactCast(blk.KNorm.Weight, hf, p+"self_attn.k_norm.weight"); err != nil {
				return err
			}
		}
		if err := copyExactCast(blk.N2.Weight, hf, p+"post_attention_layernorm.weight"); err != nil {
			return err
		}
		if blk.MoE != nil {
			if err := mapHFMoE(blk.MoE, hf, p+"mlp."); err != nil {
				return err
			}
			continue
		}
		if err := copyExactCast(blk.GateProj.W, hf, p+"mlp.gate_proj.weight"); err != nil {
			return err
		}
		if err := copyExactCast(blk.UpProj.W, hf, p+"mlp.up_proj.weight"); err != nil {
			return err
		}
		if err := copyExactCast(blk.DownProj.W, hf, p+"mlp.down_proj.weight"); err != nil {
			return err
		}
	}
	if err := copyExactCast(m.FinalNorm.Weight, hf, prefix+"norm.weight"); err != nil {
		return err
	}
	if !m.Cfg.TieEmbed {
		if err := copyExactCast(m.LMHead.W, hf, "lm_head.weight"); err != nil {
			return err
		}
	}
	return nil
}

// copyExactCast copies hf[name] into dst, casting to dst's dtype when the
// source storage differs (e.g. HF Qwen2 ships BF16; the F32 path casts
// up, the mixed-precision path keeps BF16 verbatim). Source is on CPU,
// destination on whichever device the model lives on; CopyFrom streams the
// bytes directly so we don't allocate a transient on the destination.
//
// After a successful copy the entry is deleted from `hf` so its CPU buffer
// becomes eligible for GC.
func copyExactCast(dst *tensor.Tensor, hf map[string]*tensor.Tensor, name string) error {
	src, ok := hf[name]
	if !ok {
		return fmt.Errorf("HF Llama: missing tensor %q", name)
	}
	// Stable Diffusion stores Transformer2DModel's `proj_in`/`proj_out` and
	// old AutoencoderKL projections as Conv2d 1×1 — so HF ships shape
	// (out, in, 1, 1) where our Linear holds (out, in). Squeeze unit
	// trailing dims so the shapes match without forcing the model to use
	// Conv2d 1×1.
	if !dst.Shape().Equal(src.Shape()) {
		if reduced := squeezeTrailingOnes(src, dst.Rank()); reduced != nil {
			src = reduced
		}
	}
	if !dst.Shape().Equal(src.Shape()) {
		return fmt.Errorf("%s: shape mismatch (model %v, HF %v)", name, dst.Shape(), src.Shape())
	}
	if src.DType() != dst.DType() {
		castSrc, err := ops.Cast(src, dst.DType()) // on src.Device() — typically CPU
		if err != nil {
			return fmt.Errorf("%s: cast %v→%v: %w", name, src.DType(), dst.DType(), err)
		}
		src = castSrc
	}
	if err := dst.CopyFrom(src); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	delete(hf, name)
	return nil
}

// squeezeTrailingOnes reduces `t`'s rank down to `targetRank` by removing
// trailing unit dimensions, returning a reshaped view. Returns nil when the
// trailing dims aren't all 1 (caller falls back to the shape-mismatch error).
func squeezeTrailingOnes(t *tensor.Tensor, targetRank int) *tensor.Tensor {
	if t.Rank() <= targetRank {
		return nil
	}
	dims := t.Shape().Dims()
	for i := targetRank; i < len(dims); i++ {
		if dims[i] != 1 {
			return nil
		}
	}
	r, err := t.Reshape(dims[:targetRank]...)
	if err != nil {
		return nil
	}
	return r
}

// mapHFMoE fills an nn.SparseMoE from either HF expert layout:
//
//   - transformers ≥ 5 fused: `experts.gate_up_proj` (E, 2I, H),
//     `experts.down_proj` (E, H, I), router `router.weight` or `gate.weight`
//   - hub checkpoints: `gate.weight` (E, H) and per-expert
//     `experts.{e}.gate_proj|up_proj|down_proj.weight`
//
// Per-expert matrices are stacked into the fused layout with ops.Concat /
// ops.Stack on the CPU, then copied once to the model device.
func mapHFMoE(m *nn.SparseMoE, hf map[string]*tensor.Tensor, p string) error {
	routerName := p + "gate.weight"
	if _, ok := hf[routerName]; !ok {
		routerName = p + "router.weight"
	}
	if err := copyExactCast(m.Gate.W, hf, routerName); err != nil {
		return err
	}
	if _, ok := hf[p+"experts.gate_up_proj"]; ok {
		if err := copyExactCast(m.GateUp, hf, p+"experts.gate_up_proj"); err != nil {
			return err
		}
		return copyExactCast(m.Down, hf, p+"experts.down_proj")
	}
	E := m.GateUp.Dim(0)
	gu := make([]*tensor.Tensor, E)
	dn := make([]*tensor.Tensor, E)
	for e := range E {
		pe := p + "experts." + strconv.Itoa(e) + "."
		g, okG := hf[pe+"gate_proj.weight"]
		u, okU := hf[pe+"up_proj.weight"]
		d, okD := hf[pe+"down_proj.weight"]
		if !okG || !okU || !okD {
			return fmt.Errorf("mapHFMoE: expert %d tensors missing under %s", e, pe)
		}
		cat, err := ops.Concat([]*tensor.Tensor{g, u}, 0)
		if err != nil {
			return err
		}
		gu[e], dn[e] = cat, d
		delete(hf, pe+"gate_proj.weight")
		delete(hf, pe+"up_proj.weight")
		delete(hf, pe+"down_proj.weight")
	}
	stackedGU, err := ops.Stack(gu, 0)
	if err != nil {
		return err
	}
	stackedDN, err := ops.Stack(dn, 0)
	if err != nil {
		return err
	}
	hf[p+"experts.gate_up_proj"] = stackedGU
	hf[p+"experts.down_proj"] = stackedDN
	if err := copyExactCast(m.GateUp, hf, p+"experts.gate_up_proj"); err != nil {
		return err
	}
	return copyExactCast(m.Down, hf, p+"experts.down_proj")
}
