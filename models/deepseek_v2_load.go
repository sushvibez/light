package models

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"

	"light/device"
	"light/dtype"
	"light/hub"
	"light/ops"
	"light/tensor"
)

// deepseekV2HFConfig is a deepseek_v2 config.json.
type deepseekV2HFConfig struct {
	VocabSize           int           `json:"vocab_size"`
	HiddenSize          int           `json:"hidden_size"`
	IntermediateSize    int           `json:"intermediate_size"`
	MoEIntermediateSize int           `json:"moe_intermediate_size"`
	NumHiddenLayers     int           `json:"num_hidden_layers"`
	NumAttentionHeads   int           `json:"num_attention_heads"`
	QLoraRank           *int          `json:"q_lora_rank"`
	KVLoraRank          int           `json:"kv_lora_rank"`
	QKNopeHeadDim       int           `json:"qk_nope_head_dim"`
	QKRopeHeadDim       int           `json:"qk_rope_head_dim"`
	VHeadDim            int           `json:"v_head_dim"`
	NRoutedExperts      *int          `json:"n_routed_experts"`
	NumExpertsPerTok    int           `json:"num_experts_per_tok"`
	NSharedExperts      *int          `json:"n_shared_experts"`
	FirstKDenseReplace  int           `json:"first_k_dense_replace"`
	MoELayerFreq        int           `json:"moe_layer_freq"`
	NormTopKProb        bool          `json:"norm_topk_prob"`
	RoutedScaling       float64       `json:"routed_scaling_factor"`
	TopkMethod          string        `json:"topk_method"`
	NGroup              int           `json:"n_group"`
	TopkGroup           int           `json:"topk_group"`
	ScoringFunc         string        `json:"scoring_func"`
	HiddenAct           string        `json:"hidden_act"`
	AttentionBias       bool          `json:"attention_bias"`
	RopeTheta           float64       `json:"rope_theta"`
	RopeScaling         *hfRopeParams `json:"rope_scaling"`
	RopeParameters      *hfRopeParams `json:"rope_parameters"`
	RmsNormEps          float64       `json:"rms_norm_eps"`
	MaxPosition         int           `json:"max_position_embeddings"`
	TieWordEmbeddings   bool          `json:"tie_word_embeddings"`
}

func fetchDeepseekV2Config(repo string) (DeepseekV2Config, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return DeepseekV2Config{}, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return DeepseekV2Config{}, err
	}
	var hf deepseekV2HFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return DeepseekV2Config{}, fmt.Errorf("deepseek_v2 config: %w", err)
	}
	if a := hf.HiddenAct; a != "" && a != "silu" {
		return DeepseekV2Config{}, fmt.Errorf("deepseek_v2: hidden_act %q not supported", a)
	}
	if hf.AttentionBias {
		return DeepseekV2Config{}, fmt.Errorf("deepseek_v2: attention_bias not supported")
	}
	if s := hf.ScoringFunc; s != "" && s != "softmax" {
		return DeepseekV2Config{}, fmt.Errorf("deepseek_v2: scoring_func %q not supported (softmax)", s)
	}
	switch hf.TopkMethod {
	case "", "greedy":
	case "group_limited_greedy":
		// The group restriction is a no-op when every group may be chosen.
		if hf.NGroup > 1 && hf.TopkGroup < hf.NGroup {
			return DeepseekV2Config{}, fmt.Errorf("deepseek_v2: group_limited_greedy routing (n_group %d, topk_group %d) not supported yet", hf.NGroup, hf.TopkGroup)
		}
	default:
		return DeepseekV2Config{}, fmt.Errorf("deepseek_v2: topk_method %q not supported", hf.TopkMethod)
	}
	rp := hf.RopeParameters
	if rp == nil {
		rp = hf.RopeScaling
	}
	base := hf.RopeTheta
	if rp != nil && rp.RopeTheta > 0 {
		base = rp.RopeTheta
	}
	if base == 0 {
		base = 10000
	}
	scaling, err := rp.scaling(0, hf.MaxPosition)
	if err != nil {
		return DeepseekV2Config{}, fmt.Errorf("deepseek_v2: %w", err)
	}
	cfg := DeepseekV2Config{
		VocabSize: hf.VocabSize, HiddenSize: hf.HiddenSize, IntermediateSize: hf.IntermediateSize,
		NumLayers: hf.NumHiddenLayers, NumHeads: hf.NumAttentionHeads,
		KVLoraRank: hf.KVLoraRank, QKNopeDim: hf.QKNopeHeadDim, QKRopeDim: hf.QKRopeHeadDim, VHeadDim: hf.VHeadDim,
		MoEIntermediate: hf.MoEIntermediateSize, TopK: hf.NumExpertsPerTok, FirstKDense: hf.FirstKDenseReplace,
		MoELayerFreq: hf.MoELayerFreq, NormTopK: hf.NormTopKProb, RoutedScale: hf.RoutedScaling,
		RopeBase: base, RopeScaling: scaling, RmsEps: hf.RmsNormEps, MaxPosition: hf.MaxPosition,
		TieEmbed: hf.TieWordEmbeddings,
	}
	if hf.QLoraRank != nil {
		cfg.QLoraRank = *hf.QLoraRank
	}
	if hf.NRoutedExperts != nil {
		cfg.NumExperts = *hf.NRoutedExperts
	}
	if hf.NSharedExperts != nil {
		cfg.NumShared = *hf.NSharedExperts
	}
	if cfg.RmsEps == 0 {
		cfg.RmsEps = 1e-6
	}
	return cfg, nil
}

// deinterleaveRows reorders, in every group of groupSize rows of w, the n
// rows starting at offset from interleaved-pair order (2i, 2i+1) to
// rotate-half order (i, i + n/2). DeepSeek's checkpoints rotate complex pairs
// (transformers' apply_rotary_emb via view_as_complex); reordering the rope
// rows of the query and of the shared key alike turns that into the
// rotate-half RoPE Light's kernels apply, with identical scores.
func deinterleaveRows(w *tensor.Tensor, groupSize, offset, n int) (*tensor.Tensor, error) {
	rows := w.Dim(0)
	if rows%groupSize != 0 || offset+n > groupSize || n%2 != 0 {
		return nil, fmt.Errorf("deinterleaveRows: %d rows, group %d, rope rows %d..%d", rows, groupSize, offset, offset+n)
	}
	idx := make([]int64, rows)
	for r := range idx {
		idx[r] = int64(r)
	}
	half := n / 2
	for g := 0; g < rows/groupSize; g++ {
		base := g*groupSize + offset
		for i := range half {
			idx[base+i] = int64(base + 2*i)          // new row i      ← old 2i
			idx[base+half+i] = int64(base + 2*i + 1) // new row half+i ← old 2i+1
		}
	}
	it, err := tensor.New(idx, []int{rows}, w.Device())
	if err != nil {
		return nil, err
	}
	return ops.IndexSelect(w, it)
}

// LoadDeepseekV2FromHF loads a deepseek_v2 checkpoint (Hub repo or local
// directory), e.g. deepseek-ai/DeepSeek-V2-Lite-Chat. weightDT is F32 or
// BF16 (activations run at it); the MoE router weight is kept in F32.
func LoadDeepseekV2FromHF(repo string, dev device.Device, weightDT dtype.DType) (*DeepseekV2, error) {
	cfg, err := fetchDeepseekV2Config(repo)
	if err != nil {
		return nil, err
	}
	cfg.WeightDType = weightDT
	hf, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, fmt.Errorf("LoadDeepseekV2FromHF: %w", err)
	}
	m, err := NewDeepseekV2(cfg, dev)
	if err != nil {
		return nil, err
	}
	act := m.Cfg.activationDType()
	dq, R, dr := cfg.qkHeadDim(), cfg.KVLoraRank, cfg.QKRopeDim
	// reorder replaces hf[name] by its rope-row-reordered version.
	reorder := func(name string, groupSize, offset int) error {
		w, ok := hf[name]
		if !ok {
			return fmt.Errorf("LoadDeepseekV2FromHF: missing tensor %q", name)
		}
		r, err := deinterleaveRows(w, groupSize, offset, dr)
		if err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		hf[name] = r
		return nil
	}
	// toDev casts an HF tensor and moves it to dev, freeing the host copy.
	toDev := func(name string, dt dtype.DType) (*tensor.Tensor, error) {
		src, ok := hf[name]
		if !ok {
			return nil, fmt.Errorf("LoadDeepseekV2FromHF: missing tensor %q", name)
		}
		delete(hf, name)
		if src.DType() != dt {
			if src, err = ops.Cast(src, dt); err != nil {
				return nil, fmt.Errorf("%s: %w", name, err)
			}
		}
		return src.To(dev)
	}
	if err := copyExactCast(m.TokenEmb.Weight, hf, "model.embed_tokens.weight"); err != nil {
		return nil, err
	}
	if err := copyExactCast(m.FinalNorm.Weight, hf, "model.norm.weight"); err != nil {
		return nil, err
	}
	if !cfg.TieEmbed {
		if err := copyExactCast(m.LMHead.W, hf, "lm_head.weight"); err != nil {
			return nil, err
		}
	}
	for i, b := range m.Blocks {
		p := "model.layers." + strconv.Itoa(i) + "."
		sa := p + "self_attn."
		qName := sa + "q_proj.weight"
		if cfg.QLoraRank > 0 {
			qName = sa + "q_b_proj.weight"
		}
		if err := reorder(qName, dq, cfg.QKNopeDim); err != nil {
			return nil, err
		}
		if err := reorder(sa+"kv_a_proj_with_mqa.weight", R+dr, R); err != nil {
			return nil, err
		}
		type pair struct {
			dst  *tensor.Tensor
			name string
		}
		pairs := []pair{
			{b.InNorm.Weight, p + "input_layernorm.weight"},
			{b.PostNorm.Weight, p + "post_attention_layernorm.weight"},
			{b.KVA.W, sa + "kv_a_proj_with_mqa.weight"},
			{b.KVANorm.Weight, sa + "kv_a_layernorm.weight"},
			{b.KVB.W, sa + "kv_b_proj.weight"},
			{b.OProj.W, sa + "o_proj.weight"},
		}
		if cfg.QLoraRank > 0 {
			pairs = append(pairs, pair{b.QA.W, sa + "q_a_proj.weight"}, pair{b.QANorm.Weight, sa + "q_a_layernorm.weight"},
				pair{b.QB.W, sa + "q_b_proj.weight"})
		} else {
			pairs = append(pairs, pair{b.QProj.W, sa + "q_proj.weight"})
		}
		mlp := p + "mlp."
		if b.MoE == nil {
			pairs = append(pairs, pair{b.Gate.W, mlp + "gate_proj.weight"}, pair{b.Up.W, mlp + "up_proj.weight"},
				pair{b.Down.W, mlp + "down_proj.weight"})
		} else if b.SGate != nil {
			pairs = append(pairs, pair{b.SGate.W, mlp + "shared_experts.gate_proj.weight"},
				pair{b.SUp.W, mlp + "shared_experts.up_proj.weight"}, pair{b.SDown.W, mlp + "shared_experts.down_proj.weight"})
		}
		for _, pr := range pairs {
			if err := copyExactCast(pr.dst, hf, pr.name); err != nil {
				return nil, err
			}
		}
		if b.MoE == nil {
			continue
		}
		router, err := toDev(mlp+"gate.weight", dtype.F32)
		if err != nil {
			return nil, err
		}
		if err := b.MoE.Gate.SetWeight(router); err != nil {
			return nil, fmt.Errorf("%sgate.weight: %w", mlp, err)
		}
		b.MoE.RouterF32 = true
		E := cfg.NumExperts
		gate, up, down := make([]*tensor.Tensor, E), make([]*tensor.Tensor, E), make([]*tensor.Tensor, E)
		for e := range E {
			ep := mlp + "experts." + strconv.Itoa(e) + "."
			if gate[e], err = toDev(ep+"gate_proj.weight", act); err != nil {
				return nil, err
			}
			if up[e], err = toDev(ep+"up_proj.weight", act); err != nil {
				return nil, err
			}
			if down[e], err = toDev(ep+"down_proj.weight", act); err != nil {
				return nil, err
			}
		}
		if err := b.MoE.StackExperts(gate, up, down); err != nil {
			return nil, fmt.Errorf("layer %d experts: %w", i, err)
		}
	}
	return m, nil
}
