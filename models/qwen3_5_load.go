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

// qwen35HFConfig is the text part of a qwen3_5 / qwen3_5_text config.json.
// Multimodal checkpoints (Qwen3_5ForConditionalGeneration, the published
// Qwen3.5 / 3.8 repos) nest it under text_config.
type qwen35HFConfig struct {
	ModelType            string   `json:"model_type"`
	VocabSize            int      `json:"vocab_size"`
	HiddenSize           int      `json:"hidden_size"`
	IntermediateSize     int      `json:"intermediate_size"`
	NumHiddenLayers      int      `json:"num_hidden_layers"`
	LayerTypes           []string `json:"layer_types"`
	FullAttnInterval     int      `json:"full_attention_interval"`
	NumAttentionHeads    int      `json:"num_attention_heads"`
	NumKeyValueHeads     int      `json:"num_key_value_heads"`
	HeadDim              int      `json:"head_dim"`
	AttnOutputGate       *bool    `json:"attn_output_gate"`
	LinearNumKeyHeads    int      `json:"linear_num_key_heads"`
	LinearNumValueHeads  int      `json:"linear_num_value_heads"`
	LinearKeyHeadDim     int      `json:"linear_key_head_dim"`
	LinearValueHeadDim   int      `json:"linear_value_head_dim"`
	LinearConvKernelDim  int      `json:"linear_conv_kernel_dim"`
	RmsNormEps           float64  `json:"rms_norm_eps"`
	MaxPositionEmbedding int      `json:"max_position_embeddings"`
	TieWordEmbeddings    *bool    `json:"tie_word_embeddings"`
	PartialRotaryFactor  float64  `json:"partial_rotary_factor"`
	RopeTheta            float64  `json:"rope_theta"`
	RopeParameters       *struct {
		RopeTheta           float64 `json:"rope_theta"`
		PartialRotaryFactor float64 `json:"partial_rotary_factor"`
		RopeType            string  `json:"rope_type"`
		MRoPESection        []int   `json:"mrope_section"`
		MRoPEInterleaved    bool    `json:"mrope_interleaved"`
	} `json:"rope_parameters"`
	// qwen3_5_moe: routed experts plus a gated shared expert.
	NumExperts                   int             `json:"num_experts"`
	NumExpertsPerTok             int             `json:"num_experts_per_tok"`
	MoEIntermediateSize          int             `json:"moe_intermediate_size"`
	SharedExpertIntermediateSize int             `json:"shared_expert_intermediate_size"`
	ImageTokenID                 int64           `json:"image_token_id"`
	TextConfig                   json.RawMessage `json:"text_config"`
	// Fine-grained FP8 checkpoints (Qwen/Qwen3.5-27B-FP8): F8E4M3 weights
	// with a per-block `weight_scale_inv`; MLX checkpoints: affine
	// quantization (see mlxQuant), also under `quantization`. Top level,
	// not in text_config.
	QuantizationConfig json.RawMessage `json:"quantization_config"`
	Quantization       json.RawMessage `json:"quantization"`
}

// fetchQwen35Config reads config.json (descending into text_config) into a
// Qwen35Config.
func fetchQwen35Config(repo string) (Qwen35Config, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return Qwen35Config{}, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return Qwen35Config{}, err
	}
	var top, hf qwen35HFConfig
	if err := json.Unmarshal(raw, &top); err != nil {
		return Qwen35Config{}, fmt.Errorf("qwen3.5 config: %w", err)
	}
	hf = top
	if len(top.TextConfig) > 0 {
		hf = qwen35HFConfig{}
		if err := json.Unmarshal(top.TextConfig, &hf); err != nil {
			return Qwen35Config{}, fmt.Errorf("qwen3.5 text_config: %w", err)
		}
	}
	tie := false
	switch {
	case hf.TieWordEmbeddings != nil:
		tie = *hf.TieWordEmbeddings
	case top.TieWordEmbeddings != nil:
		tie = *top.TieWordEmbeddings
	}
	theta, partial := hf.RopeTheta, hf.PartialRotaryFactor
	var mropeSection []int
	if rp := hf.RopeParameters; rp != nil {
		if len(rp.MRoPESection) > 0 && !rp.MRoPEInterleaved {
			return Qwen35Config{}, fmt.Errorf("qwen3.5: chunked (non-interleaved) MRoPE not supported")
		}
		mropeSection = rp.MRoPESection
		if rp.RopeType != "" && rp.RopeType != "default" {
			return Qwen35Config{}, fmt.Errorf("qwen3.5: rope_type %q not supported", rp.RopeType)
		}
		if rp.RopeTheta > 0 {
			theta = rp.RopeTheta
		}
		if rp.PartialRotaryFactor > 0 {
			partial = rp.PartialRotaryFactor
		}
	}
	if theta == 0 {
		theta = 10000
	}
	n := hf.NumHiddenLayers
	linear := make([]bool, n)
	switch {
	case len(hf.LayerTypes) == n:
		for i, t := range hf.LayerTypes {
			switch t {
			case "linear_attention":
				linear[i] = true
			case "full_attention":
			default:
				return Qwen35Config{}, fmt.Errorf("qwen3.5: layer %d type %q not supported", i, t)
			}
		}
	case hf.FullAttnInterval > 0:
		for i := range linear {
			linear[i] = (i+1)%hf.FullAttnInterval != 0
		}
	default:
		return Qwen35Config{}, fmt.Errorf("qwen3.5: config has neither layer_types nor full_attention_interval")
	}
	gate := hf.AttnOutputGate == nil || *hf.AttnOutputGate
	eps := hf.RmsNormEps
	if eps == 0 {
		eps = 1e-6
	}
	conv := hf.LinearConvKernelDim
	if conv == 0 {
		conv = 4
	}
	var fp8Block [2]int
	qraw := top.Quantization
	if len(qraw) == 0 {
		qraw = top.QuantizationConfig
	}
	mlxQ, err := parseMLXQuant(qraw)
	if err != nil {
		return Qwen35Config{}, err
	}
	if len(top.QuantizationConfig) > 0 && mlxQ == nil {
		var q struct {
			QuantMethod     string `json:"quant_method"`
			WeightBlockSize []int  `json:"weight_block_size"`
		}
		if err := json.Unmarshal(top.QuantizationConfig, &q); err != nil {
			return Qwen35Config{}, fmt.Errorf("qwen3.5 quantization_config: %w", err)
		}
		if q.QuantMethod != "fp8" || len(q.WeightBlockSize) != 2 {
			return Qwen35Config{}, fmt.Errorf("qwen3.5: quantization %q (block %v) not supported; fp8 with a 2-D weight_block_size, or MLX affine, is", q.QuantMethod, q.WeightBlockSize)
		}
		fp8Block = [2]int{q.WeightBlockSize[0], q.WeightBlockSize[1]}
	}
	inter := hf.IntermediateSize
	if hf.NumExperts > 0 {
		inter = hf.SharedExpertIntermediateSize
	}
	return Qwen35Config{
		VocabSize: hf.VocabSize, HiddenSize: hf.HiddenSize, IntermediateSize: inter,
		NumExperts: hf.NumExperts, MoETopK: hf.NumExpertsPerTok, MoEIntermediate: hf.MoEIntermediateSize, mlx: mlxQ,
		NumLayers: n, LinearLayers: linear,
		NumHeads: hf.NumAttentionHeads, NumKVHeads: hf.NumKeyValueHeads, HeadDim: hf.HeadDim,
		PartialRotary: partial, RopeBase: theta, AttnOutputGate: gate,
		LinearKHeads: hf.LinearNumKeyHeads, LinearVHeads: hf.LinearNumValueHeads,
		LinearKDim: hf.LinearKeyHeadDim, LinearVDim: hf.LinearValueHeadDim, ConvKernel: conv,
		RmsEps: eps, MaxPosition: hf.MaxPositionEmbedding, TieEmbed: tie,
		fp8Block: fp8Block, MRoPESection: mropeSection, ImageTokenID: top.ImageTokenID,
	}, nil
}

// LoadQwen35FromHF loads the text model of a qwen3_5 or qwen3_5_moe
// checkpoint (Hub repo or local directory), e.g. `Qwen/Qwen3.5-0.8B`. The
// vision tower (`model.visual.*`) and the multi-token-prediction head
// (`mtp.*`) are not loaded. weightDT is dtype.F32 or dtype.BF16: the
// activation dtype, and the storage dtype of every weight except those of
// a fine-grained FP8 checkpoint (`quant_method: fp8`, e.g.
// Qwen/Qwen3.5-27B-FP8), whose projections stay F8E4M3 with their
// per-block scales (nn.Linear.WBlockScale) and are expanded to weightDT one
// matmul at a time, and those of an MLX affine-quantized checkpoint
// (`language_model.*` tensors, e.g. Edge0/Edge0-35B-A3B-preview), whose
// 4-bit modules stay packed (quant.Int4) and whose other widths and the
// embedding are expanded at load. LIGHT_EXPERT_SLOTS streams the routed
// experts; a LoRA adapter next to the weights (see findLoRA) is applied as
// a side path.
func LoadQwen35FromHF(repo string, dev device.Device, weightDT dtype.DType) (*Qwen35, error) {
	cfg, err := fetchQwen35Config(repo)
	if err != nil {
		return nil, err
	}
	cfg.WeightDType = weightDT
	cfg.lazyWeights = true // Linears and the embedding are installed below, at their final dtype
	cfg.ExpertSlots = envExpertSlots()
	paths, err := hub.DownloadModelSafetensors(repo, "main")
	if err != nil {
		return nil, fmt.Errorf("LoadQwen35FromHF: %w", err)
	}
	hf, meta, err := safetensors.LoadFiles(paths, device.CPU())
	if err != nil {
		return nil, fmt.Errorf("LoadQwen35FromHF: %w", err)
	}
	m, err := NewQwen35(cfg, dev)
	if err != nil {
		return nil, err
	}
	prefix := ""
	for _, p := range []string{"model.language_model.", "language_model.model.", "model."} {
		if _, ok := hf[p+"embed_tokens.weight"]; ok {
			prefix = p
			break
		}
	}
	if prefix == "" {
		return nil, fmt.Errorf("LoadQwen35FromHF: no embed_tokens.weight under model., model.language_model. or language_model.model.")
	}
	headMod := "lm_head"
	if prefix == "language_model.model." {
		headMod = "language_model.lm_head"
	}
	// mlx-lm saves the zero-centred norms with the 1 already added.
	normsShifted := meta["format"] == "mlx"
	act := cfg.activationDType()
	// toDev casts an HF tensor to dt and moves it to dev, freeing the host copy.
	toDev := func(name string, dt dtype.DType) (*tensor.Tensor, error) {
		src, ok := hf[name]
		if !ok {
			return nil, fmt.Errorf("LoadQwen35FromHF: missing tensor %q", name)
		}
		delete(hf, name)
		if src.DType() != dt {
			var err error
			if src, err = ops.Cast(src, dt); err != nil {
				return nil, fmt.Errorf("%s: %w", name, err)
			}
		}
		return src.To(dev)
	}
	var emb *tensor.Tensor
	if embMod := prefix + "embed_tokens"; mlxQuantized(hf, embMod) {
		if cfg.mlx == nil {
			return nil, fmt.Errorf("LoadQwen35FromHF: %s is quantized but config.json has no quantization", embMod)
		}
		w, s, b, err := takeMLX(hf, embMod)
		if err != nil {
			return nil, err
		}
		g, bits := cfg.mlx.params(embMod)
		if emb, err = mlxDense(w, s, b, g, bits, nn.WeightStorageDType(act, cfg.HiddenSize), dev); err != nil {
			return nil, fmt.Errorf("%s: %w", embMod, err)
		}
	} else if emb, err = toDev(embMod+".weight", nn.WeightStorageDType(act, cfg.HiddenSize)); err != nil {
		return nil, err
	}
	if m.TokenEmb, err = nn.NewEmbeddingFromWeight(emb); err != nil {
		return nil, err
	}
	m.TokenEmb.WidenF32 = emb.DType() != act
	// setLinear installs module mod's weight: MLX-quantized (see
	// installMLXLinear), block-scaled FP8 as stored when the checkpoint has
	// its weight_scale_inv, else at the activation dtype.
	setLinear := func(l *nn.Linear, mod string) error {
		if mlxQuantized(hf, mod) {
			if cfg.mlx == nil {
				return fmt.Errorf("LoadQwen35FromHF: %s is quantized but config.json has no quantization", mod)
			}
			return installMLXLinear(l, hf, mod, cfg.mlx, nn.WeightStorageDType(act, l.InFeatures()), dev)
		}
		name := mod + ".weight"
		if sName := name + "_scale_inv"; cfg.fp8Block[0] > 0 && hf[sName] != nil {
			if hf[name] == nil || hf[name].DType() != dtype.F8E4M3 {
				return fmt.Errorf("LoadQwen35FromHF: %s has a block scale but is not F8E4M3", name)
			}
			w, err := toDev(name, dtype.F8E4M3)
			if err != nil {
				return err
			}
			sc, err := toDev(sName, dtype.F32)
			if err != nil {
				return err
			}
			if err := l.SetWeight(w); err != nil {
				return fmt.Errorf("%s: %w", name, err)
			}
			l.WBlockScale, l.WBlock = sc, cfg.fp8Block
			return nil
		}
		w, err := toDev(name, nn.WeightStorageDType(act, l.InFeatures()))
		if err != nil {
			return err
		}
		if err := l.SetWeight(w); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		return nil
	}
	// Zero-centred RMSNorm: store 1 + w, at the activation dtype.
	normPlusOne := func(dst **tensor.Tensor, name string) error {
		src, ok := hf[name]
		if !ok {
			return fmt.Errorf("LoadQwen35FromHF: missing tensor %q", name)
		}
		w, err := ops.Cast(src, dtype.F32)
		if err != nil {
			return err
		}
		if !normsShifted {
			one, err := tensor.New([]float32{1}, []int{1}, w.Device())
			if err != nil {
				return err
			}
			if w, err = ops.Add(w, one); err != nil {
				return err
			}
		}
		if act != dtype.F32 {
			if w, err = ops.Cast(w, act); err != nil {
				return err
			}
		}
		if !w.Shape().Equal((*dst).Shape()) {
			return fmt.Errorf("%s: shape %v, model %v", name, w.Shape(), (*dst).Shape())
		}
		if *dst, err = w.To(dev); err != nil {
			return err
		}
		return nil
	}
	// F32 parameters of the linear layers: conv weight (C, 1, K) → (C, K),
	// −exp(A_log), dt_bias, gated-norm weight.
	f32 := func(dst **tensor.Tensor, name string, fn func(*tensor.Tensor) (*tensor.Tensor, error)) error {
		src, ok := hf[name]
		if !ok {
			return fmt.Errorf("LoadQwen35FromHF: missing tensor %q", name)
		}
		w, err := ops.Cast(src, dtype.F32)
		if err != nil {
			return err
		}
		if fn != nil {
			if w, err = fn(w); err != nil {
				return err
			}
		}
		if w.Shape().NumElements() != (*dst).Shape().NumElements() {
			return fmt.Errorf("%s: shape %v, model %v", name, w.Shape(), (*dst).Shape())
		}
		if w, err = w.Reshape((*dst).Shape().Dims()...); err != nil {
			return err
		}
		*dst, err = w.To(dev)
		return err
	}
	negExp := func(x *tensor.Tensor) (*tensor.Tensor, error) {
		e, err := ops.Exp(x)
		if err != nil {
			return nil, err
		}
		return ops.Neg(e)
	}
	for i, b := range m.Blocks {
		p := prefix + "layers." + strconv.Itoa(i) + "."
		if err := normPlusOne(&b.InNorm.Weight, p+"input_layernorm.weight"); err != nil {
			return nil, err
		}
		if err := normPlusOne(&b.PostNorm.Weight, p+"post_attention_layernorm.weight"); err != nil {
			return nil, err
		}
		type pair struct {
			l    *nn.Linear
			name string
		}
		mlp := p + "mlp."
		if b.MoE != nil {
			mlp = p + "mlp.shared_expert."
		}
		pairs := []pair{
			{b.Gate, mlp + "gate_proj"},
			{b.Up, mlp + "up_proj"},
			{b.Down, mlp + "down_proj"},
		}
		if b.MoE != nil {
			pairs = append(pairs, pair{b.MoE.Gate, p + "mlp.gate"}, pair{b.SharedGate, p + "mlp.shared_expert_gate"})
		}
		if b.linear {
			la := p + "linear_attn."
			pairs = append(pairs,
				pair{b.InQKV, la + "in_proj_qkv"}, pair{b.InZ, la + "in_proj_z"},
				pair{b.InB, la + "in_proj_b"}, pair{b.InA, la + "in_proj_a"},
				pair{b.OutProj, la + "out_proj"})
			if err := f32(&b.ConvW, la+"conv1d.weight", nil); err != nil {
				return nil, err
			}
			if err := f32(&b.NegA, la+"A_log", negExp); err != nil {
				return nil, err
			}
			if err := f32(&b.DtBias, la+"dt_bias", nil); err != nil {
				return nil, err
			}
			if err := f32(&b.GNorm.Weight, la+"norm.weight", nil); err != nil {
				return nil, err
			}
		} else {
			sa := p + "self_attn."
			pairs = append(pairs,
				pair{b.QProj, sa + "q_proj"}, pair{b.KProj, sa + "k_proj"},
				pair{b.VProj, sa + "v_proj"}, pair{b.OProj, sa + "o_proj"})
			if err := normPlusOne(&b.QNorm.Weight, sa+"q_norm.weight"); err != nil {
				return nil, err
			}
			if err := normPlusOne(&b.KNorm.Weight, sa+"k_norm.weight"); err != nil {
				return nil, err
			}
		}
		for _, pr := range pairs {
			if err := setLinear(pr.l, pr.name); err != nil {
				return nil, err
			}
		}
		if b.MoE != nil {
			if err := loadQwen35Experts(b.MoE, hf, p+"mlp.", cfg.mlx, act, dev); err != nil {
				return nil, err
			}
		}
	}
	linkQwen35Streams(m)
	if err := normPlusOne(&m.FinalNorm.Weight, prefix+"norm.weight"); err != nil {
		return nil, err
	}
	if cfg.TieEmbed {
		if err := m.LMHead.SetWeight(m.TokenEmb.Weight); err != nil {
			return nil, err
		}
	} else if err := setLinear(m.LMHead, headMod); err != nil {
		return nil, err
	}
	if err := applyQwen35LoRA(m, repo, dev); err != nil {
		return nil, err
	}
	return m, nil
}

// loadQwen35Experts fills a routed-expert block under p ("…mlp."): MLX's
// stacked switch_mlp (quantized or not), transformers' fused
// experts.gate_up_proj / down_proj, or per-expert experts.{e}.*_proj;
// streamed when StreamSlots is set.
func loadQwen35Experts(moe *nn.SparseMoE, hf map[string]*tensor.Tensor, p string, q *mlxQuant, act dtype.DType, dev device.Device) error {
	if mlxQuantized(hf, p+"switch_mlp.gate_proj") {
		if q == nil {
			return fmt.Errorf("LoadQwen35FromHF: %sswitch_mlp is quantized but config.json has no quantization", p)
		}
		return installMLXExperts(moe, hf, p, q, act, dev)
	}
	if _, ok := hf[p+"switch_mlp.gate_proj.weight"]; ok { // unquantized MLX: the fused layout
		gu, err := ops.Concat([]*tensor.Tensor{hf[p+"switch_mlp.gate_proj.weight"], hf[p+"switch_mlp.up_proj.weight"]}, 1)
		if err != nil {
			return fmt.Errorf("%sswitch_mlp: %w", p, err)
		}
		hf[p+"experts.gate_up_proj"], hf[p+"experts.down_proj"] = gu, hf[p+"switch_mlp.down_proj.weight"]
		for _, n := range []string{"gate_proj", "up_proj", "down_proj"} {
			delete(hf, p+"switch_mlp."+n+".weight")
		}
	}
	if moe.StreamSlots > 0 {
		return streamHFMoE(moe, hf, p)
	}
	cast := func(name string) (*tensor.Tensor, error) {
		t, ok := hf[name]
		if !ok {
			return nil, fmt.Errorf("LoadQwen35FromHF: missing tensor %q", name)
		}
		delete(hf, name)
		var err error
		if t.DType() != act {
			if t, err = ops.Cast(t, act); err != nil {
				return nil, err
			}
		}
		return t.To(dev)
	}
	var err error
	if _, ok := hf[p+"experts.gate_up_proj"]; ok {
		if moe.GateUp, err = cast(p + "experts.gate_up_proj"); err != nil {
			return err
		}
		moe.Down, err = cast(p + "experts.down_proj")
		return err
	}
	E := moe.Gate.OutFeatures()
	g, u, d := make([]*tensor.Tensor, E), make([]*tensor.Tensor, E), make([]*tensor.Tensor, E)
	for e := range E {
		pe := p + "experts." + strconv.Itoa(e) + "."
		if g[e], err = cast(pe + "gate_proj.weight"); err != nil {
			return err
		}
		if u[e], err = cast(pe + "up_proj.weight"); err != nil {
			return err
		}
		if d[e], err = cast(pe + "down_proj.weight"); err != nil {
			return err
		}
	}
	return moe.StackExperts(g, u, d)
}

// linkQwen35Streams points each streamed MoE layer at the next one, so a
// decode step prefetches the next layer's predicted experts.
func linkQwen35Streams(m *Qwen35) {
	var prev *nn.SparseMoE
	for _, b := range m.Blocks {
		if b.MoE == nil || b.MoE.Stream == nil {
			continue
		}
		if prev != nil {
			prev.Next = b.MoE
		}
		prev = b.MoE
	}
}

// applyQwen35LoRA applies the repo's LoRA adapter, if any (findLoRA), and
// the routed width it was trained for (edge0 runs its 35B-A3B at top-4
// instead of the base model's 8).
func applyQwen35LoRA(m *Qwen35, repo string, dev device.Device) error {
	path, err := findLoRA(repo)
	if err != nil || path == "" {
		return err
	}
	ad, err := loadLoRA(path)
	if err != nil {
		return err
	}
	lin := map[string]*nn.Linear{}
	if !m.Cfg.TieEmbed {
		lin["lm_head"] = m.LMHead
	}
	for i, b := range m.Blocks {
		p := "layers." + strconv.Itoa(i) + "."
		mlp := p + "mlp."
		if b.MoE != nil {
			mlp = p + "mlp.shared_expert."
			lin[p+"mlp.gate"], lin[p+"mlp.shared_expert_gate"] = b.MoE.Gate, b.SharedGate
		}
		lin[mlp+"gate_proj"], lin[mlp+"up_proj"], lin[mlp+"down_proj"] = b.Gate, b.Up, b.Down
		if b.linear {
			la := p + "linear_attn."
			lin[la+"in_proj_qkv"], lin[la+"in_proj_z"], lin[la+"in_proj_b"], lin[la+"in_proj_a"], lin[la+"out_proj"] =
				b.InQKV, b.InZ, b.InB, b.InA, b.OutProj
		} else {
			sa := p + "self_attn."
			lin[sa+"q_proj"], lin[sa+"k_proj"], lin[sa+"v_proj"], lin[sa+"o_proj"] = b.QProj, b.KProj, b.VProj, b.OProj
		}
	}
	if err := ad.apply(lin, dev); err != nil {
		return fmt.Errorf("%s: %w", ad.Path, err)
	}
	if ad.TopK > 0 {
		for _, b := range m.Blocks {
			if b.MoE != nil {
				b.MoE.TopK = ad.TopK
			}
		}
	}
	return nil
}
