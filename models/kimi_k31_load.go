package models

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"light/device"
	"light/ops"
	"light/safetensors"
	"light/tensor"
)

// KimiK31Config is Kimi K3.1's text_config (vLLM's KimiLinearConfig with
// the K3 / K3.1 fields). Layer lists in linear_attn_config and dsa_config
// are 1-based, as in the release.
type KimiK31Config struct {
	VocabSize        int      `json:"vocab_size"`
	HiddenSize       int      `json:"hidden_size"`
	IntermediateSize int      `json:"intermediate_size"`
	MoEIntermediate  int      `json:"moe_intermediate_size"`
	NumLayers        int      `json:"num_hidden_layers"`
	NumHeads         int      `json:"num_attention_heads"`
	HiddenAct        string   `json:"hidden_act"`
	SituBeta         *float64 `json:"activation_situ_beta"`
	SituLinearBeta   *float64 `json:"activation_situ_linear_beta"`
	RMSNormEps       float64  `json:"rms_norm_eps"`
	TieEmbeddings    bool     `json:"tie_word_embeddings"`
	QLoraRank        int      `json:"q_lora_rank"`
	KVLoraRank       int      `json:"kv_lora_rank"`
	QKNopeHeadDim    int      `json:"qk_nope_head_dim"`
	QKRopeHeadDim    int      `json:"qk_rope_head_dim"`
	VHeadDim         int      `json:"v_head_dim"`
	MLAUseNope       bool     `json:"mla_use_nope"`
	MLAUseOutputGate bool     `json:"mla_use_output_gate"`
	NumExperts       int      `json:"num_experts"`
	ExpertsPerTok    int      `json:"num_experts_per_token"`
	NumShared        int      `json:"num_shared_experts"`
	MoERenormalize   bool     `json:"moe_renormalize"`
	RouterAct        string   `json:"moe_router_activation_func"`
	RoutedScaling    float64  `json:"routed_scaling_factor"`
	FirstKDense      int      `json:"first_k_dense_replace"`
	MoELayerFreq     int      `json:"moe_layer_freq"`
	UseGroupedTopk   bool     `json:"use_grouped_topk"`
	NumExpertGroup   int      `json:"num_expert_group"`
	TopkGroup        int      `json:"topk_group"`
	LatentDim        int      `json:"routed_expert_hidden_size"`
	LatentMoEUseNorm bool     `json:"latent_moe_use_norm"`
	AttnResBlockSize int      `json:"attn_res_block_size"`
	LinearAttnConfig *struct {
		FullAttnLayers  []int    `json:"full_attn_layers"`
		KDALayers       []int    `json:"kda_layers"`
		NumHeads        int      `json:"num_heads"`
		HeadDim         int      `json:"head_dim"`
		ConvKernel      int      `json:"short_conv_kernel_size"`
		GateLowerBound  *float64 `json:"gate_lower_bound"`
		UseFullRankGate bool     `json:"use_full_rank_gate"`
	} `json:"linear_attn_config"`
	DSAConfig *struct {
		DSALayers      []int  `json:"dsa_layers"`
		IndexTopk      int    `json:"index_topk"`
		IndexHeadDim   int    `json:"index_head_dim"`
		IndexNumHeads  int    `json:"index_num_heads"`
		IndexRopeDim   int    `json:"index_rope_dim"`
		IndexKNormType string `json:"index_k_norm_type"`
		IndexTopkFreq  int    `json:"index_topk_freq"`
	} `json:"dsa_config"`
	Quant *struct {
		Method string `json:"quant_method"`
		Format string `json:"format"`
	} `json:"quantization_config"`
}

func (c KimiK31Config) isKDA(i int) bool {
	return c.LinearAttnConfig != nil && slices.Contains(c.LinearAttnConfig.KDALayers, i+1)
}

func (c KimiK31Config) isDSA(i int) bool {
	return c.DSAConfig != nil && slices.Contains(c.DSAConfig.DSALayers, i+1)
}

func (c KimiK31Config) isMoE(i int) bool {
	return c.NumExperts > 0 && i >= c.FirstKDense && i%max(c.MoELayerFreq, 1) == 0
}

// LoadKimiK31 loads Kimi K3.1's language model (moonshotai/Kimi-K3.1-Pro:
// a folder with config.json and safetensors in the released naming,
// language_model.model.* / language_model.lm_head) onto dev at F32 (16-bit
// linear weights under nn.WithWeightStorage16). The compressed-tensors
// mxfp4-pack-quantized routed experts (weight_packed: e2m1 pairs, the even
// element in the low nibble; weight_scale: E8M0 per 32 inputs) are widened
// to F32 on the host. The vision tower and projector are not loaded.
func LoadKimiK31(dir string, dev device.Device) (*KimiK31, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var top struct {
		TextConfig json.RawMessage `json:"text_config"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, err
	}
	cfg := KimiK31Config{RMSNormEps: 1e-6, HiddenAct: "silu", MoERenormalize: true, RouterAct: "sigmoid",
		RoutedScaling: 1, MoELayerFreq: 1, NumExpertGroup: 1, TopkGroup: 1}
	src := raw
	if len(top.TextConfig) > 0 {
		src = top.TextConfig
	}
	if err := json.Unmarshal(src, &cfg); err != nil {
		return nil, fmt.Errorf("kimi_k31 config: %w", err)
	}
	if err := kimiK31Check(cfg); err != nil {
		return nil, err
	}
	paths, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	hf, _, err := safetensors.LoadFiles(paths, device.CPU())
	if err != nil {
		return nil, err
	}
	prefix, head := "language_model.model.", "language_model.lm_head"
	if _, ok := hf[prefix+"embed_tokens.weight"]; !ok {
		prefix, head = "model.", "lm_head"
	}
	l := &idolLoader{hf: hf, dev: dev}
	m := &KimiK31{Cfg: cfg, dev: dev}
	m.aux = &GLM5{Cfg: GLM5Config{RMSNormEps: cfg.RMSNormEps}, ones: map[int]*tensor.Tensor{}, dev: dev}
	m.Embed = l.param(prefix + "embed_tokens.weight")
	m.NormW = l.param(prefix + "norm.weight")
	if cfg.TieEmbeddings {
		hf[head+".weight"] = hf[prefix+"embed_tokens.weight"]
	}
	m.LMHead = l.linear(head)
	if cfg.AttnResBlockSize > 0 {
		m.OutResW = kimiK31ResWeight(l, prefix+"output_attn_res_norm.weight", prefix+"output_attn_res_proj.weight")
	}
	mlp := func(p string) *kimiK31MLP {
		return &kimiK31MLP{Gate: l.linear(p + ".gate_proj"), Up: l.linear(p + ".up_proj"), Down: l.linear(p + ".down_proj")}
	}
	for i := range cfg.NumLayers {
		p := fmt.Sprintf("%slayers.%d.", prefix, i)
		ly := &kimiK31Layer{ID: i, InNormW: l.param(p + "input_layernorm.weight"),
			PostNormW: l.param(p + "post_attention_layernorm.weight")}
		if cfg.AttnResBlockSize > 0 {
			ly.AttnResW = kimiK31ResWeight(l, p+"self_attention_res_norm.weight", p+"self_attention_res_proj.weight")
			ly.MLPResW = kimiK31ResWeight(l, p+"mlp_res_norm.weight", p+"mlp_res_proj.weight")
		}
		a := p + "self_attn."
		if cfg.isKDA(i) {
			ly.KDA = true
			ly.Q, ly.K, ly.V, ly.O = l.linear(a+"q_proj"), l.linear(a+"k_proj"), l.linear(a+"v_proj"), l.linear(a+"o_proj")
			ly.B, ly.FA, ly.FB, ly.G = l.linear(a+"b_proj"), l.linear(a+"f_a_proj"), l.linear(a+"f_b_proj"), l.linear(a+"g_proj")
			ly.DtBias, ly.ONormW = l.param(a+"dt_bias"), l.param(a+"o_norm.weight")
			if ly.ALog = l.host(a + "A_log"); l.err == nil && len(ly.ALog) < cfg.LinearAttnConfig.NumHeads {
				return nil, fmt.Errorf("kimi_k31 layer %d: A_log has %d entries for %d heads", i, len(ly.ALog), cfg.LinearAttnConfig.NumHeads)
			}
			ly.ALog = ly.ALog[:min(len(ly.ALog), cfg.LinearAttnConfig.NumHeads)] // the release pads A_log; vLLM reads the first num_heads
			if ly.ConvW, err = kimiK31Conv(l, a); err != nil {
				return nil, fmt.Errorf("kimi_k31 layer %d: %w", i, err)
			}
		} else {
			ly.QA, ly.QB = l.linear(a+"q_a_proj"), l.linear(a+"q_b_proj")
			ly.KVA, ly.KVB = l.linear(a+"kv_a_proj_with_mqa"), l.linear(a+"kv_b_proj")
			ly.QANormW, ly.KVANormW = l.param(a+"q_a_layernorm.weight"), l.param(a+"kv_a_layernorm.weight")
			ly.MO = l.linear(a + "o_proj")
			if cfg.MLAUseOutputGate {
				ly.MG = l.linear(a + "g_proj")
			}
			if cfg.isDSA(i) {
				ly.IWqB, ly.IWk, ly.IWeights = l.linear(a+"indexer.wq_b"), l.linear(a+"indexer.wk"), l.linear(a+"indexer.weights_proj")
				ly.IKNormW = l.param(a + "indexer.k_norm.weight")
				if cfg.DSAConfig.IndexKNormType == "layernorm" {
					ly.IKNormB = l.param(a + "indexer.k_norm.bias")
				}
			}
		}
		if cfg.isMoE(i) {
			f := p + "block_sparse_moe."
			ly.RouterW, ly.RouterBias = l.param(f+"gate.weight"), l.host(f+"gate.e_score_correction_bias")
			if cfg.NumShared > 0 {
				ly.Shared = mlp(f + "shared_experts")
			}
			if cfg.LatentDim > 0 {
				ly.LatentDown, ly.LatentUp = l.linear(f+"routed_expert_down_proj"), l.linear(f+"routed_expert_up_proj")
				if cfg.LatentMoEUseNorm {
					ly.LatentNormW = l.param(f + "routed_expert_norm.weight")
				}
			}
			for e := range cfg.NumExperts {
				q := fmt.Sprintf("%sexperts.%d.", f, e)
				for _, w := range []string{"w1", "w2", "w3"} {
					kimiK31MXFP4(l, q+w)
				}
				ly.Experts = append(ly.Experts, &kimiK31MLP{Gate: l.linear(q + "w1"), Up: l.linear(q + "w3"), Down: l.linear(q + "w2")})
			}
		} else {
			ly.Dense = mlp(p + "mlp")
		}
		if l.err != nil {
			return nil, fmt.Errorf("kimi_k31 layer %d: %w", i, l.err)
		}
		m.Layers = append(m.Layers, ly)
	}
	if l.err != nil {
		return nil, fmt.Errorf("kimi_k31: %w", l.err)
	}
	return m, nil
}

func kimiK31Check(cfg KimiK31Config) error {
	switch {
	case cfg.HiddenAct != "silu" && cfg.HiddenAct != "situ":
		return fmt.Errorf("kimi_k31: hidden_act %q (want silu or situ)", cfg.HiddenAct)
	case cfg.RouterAct != "sigmoid" && cfg.RouterAct != "softmax":
		return fmt.Errorf("kimi_k31: moe_router_activation_func %q", cfg.RouterAct)
	case cfg.QLoraRank == 0 && cfg.KVLoraRank > 0:
		return fmt.Errorf("kimi_k31: MLA without q_lora_rank (q_proj) is not supported")
	case cfg.KVLoraRank > 0 && !cfg.MLAUseNope:
		return fmt.Errorf("kimi_k31: MLA with RoPE (mla_use_nope false) is not supported (Kimi K3 MLA is NoPE-only)")
	}
	if lc := cfg.LinearAttnConfig; lc != nil && len(lc.KDALayers) > 0 && !lc.UseFullRankGate {
		return fmt.Errorf("kimi_k31: KDA without use_full_rank_gate (Kimi Linear's low-rank output gate) is not supported")
	}
	if dc := cfg.DSAConfig; dc != nil {
		if dc.IndexRopeDim != 0 {
			return fmt.Errorf("kimi_k31: index_rope_dim %d (indexer RoPE) is not supported", dc.IndexRopeDim)
		}
		if dc.IndexTopkFreq > 1 {
			return fmt.Errorf("kimi_k31: index_topk_freq %d (index sharing) is not supported", dc.IndexTopkFreq)
		}
		if t := dc.IndexKNormType; t != "" && t != "layernorm" && t != "rmsnorm" {
			return fmt.Errorf("kimi_k31: index_k_norm_type %q", t)
		}
		if dc.IndexTopk < 1 {
			return fmt.Errorf("kimi_k31: index_topk %d", dc.IndexTopk)
		}
	}
	if g := cfg.NumExpertGroup; cfg.NumExperts > 0 && (g < 1 || cfg.NumExperts%g != 0 || cfg.TopkGroup < 1 || cfg.TopkGroup > g) {
		return fmt.Errorf("kimi_k31: %d experts in %d groups, topk_group %d", cfg.NumExperts, g, cfg.TopkGroup)
	}
	return nil
}

// kimiK31ResWeight folds an AttnRes scorer (RMSNorm weight and the (1, D)
// projection) into the one vector the score uses: qk_w ⊙ norm_w.
func kimiK31ResWeight(l *idolLoader, norm, proj string) *tensor.Tensor {
	nw, pw := l.host(norm), l.host(proj)
	if l.err != nil {
		return nil
	}
	if len(nw) != len(pw) {
		l.err = fmt.Errorf("%s has %d entries, %s %d", norm, len(nw), proj, len(pw))
		return nil
	}
	w := make([]float32, len(nw))
	for i := range w {
		w[i] = nw[i] * pw[i]
	}
	var t *tensor.Tensor
	t, l.err = tensor.New(w, []int{len(w)}, l.dev)
	return t
}

// kimiK31Conv concatenates the separate q / k / v depthwise short
// convolutions (C, 1, K) into one (3C, K) weight.
func kimiK31Conv(l *idolLoader, a string) (*tensor.Tensor, error) {
	if l.err != nil {
		return nil, l.err
	}
	var parts []*tensor.Tensor
	for _, n := range []string{"q", "k", "v"} {
		w, ok := l.hf[a+n+"_conv1d.weight"]
		if !ok {
			return nil, fmt.Errorf("missing %s%s_conv1d.weight", a, n)
		}
		if w.Rank() == 3 {
			c, err := w.Contiguous()
			if err != nil {
				return nil, err
			}
			if w, err = c.Reshape(w.Dim(0), w.Dim(2)); err != nil {
				return nil, err
			}
		}
		parts = append(parts, w)
	}
	c, err := ops.Concat(parts, 0)
	if err != nil {
		return nil, err
	}
	l.hf[a+"conv1d.weight"] = c
	return l.param(a + "conv1d.weight"), l.err
}

// kimiK31MXFP4 widens a compressed-tensors mxfp4-pack-quantized weight
// (<name>.weight_packed U8 (N, K/2), <name>.weight_scale U8 E8M0 (N, K/32))
// into <name>.weight at F32; unquantized weights are left alone.
func kimiK31MXFP4(l *idolLoader, name string) {
	if l.err != nil {
		return
	}
	w, ok := l.hf[name+".weight_packed"]
	if !ok {
		return
	}
	sc, ok := l.hf[name+".weight_scale"]
	if !ok {
		l.err = fmt.Errorf("%s.weight_packed without weight_scale", name)
		return
	}
	scF, err := v41ScaleF32(sc)
	if err == nil {
		l.hf[name+".weight"], err = dequantFP4Host(w, scF)
	}
	if err != nil {
		l.err = fmt.Errorf("%s: %w", name, err)
		return
	}
	delete(l.hf, name+".weight_packed")
	delete(l.hf, name+".weight_scale")
}
