package models

import (
	"encoding/json"
	"fmt"
	"math"
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

type gemma4HFRope struct {
	RopeTheta           float64 `json:"rope_theta"`
	RopeType            string  `json:"rope_type"`
	PartialRotaryFactor float64 `json:"partial_rotary_factor"`
}

type gemma4HFConfig struct {
	ModelType               string          `json:"model_type"`
	TextConfig              *gemma4HFConfig `json:"text_config"`
	VocabSize               int             `json:"vocab_size"`
	HiddenSize              int             `json:"hidden_size"`
	IntermediateSize        int             `json:"intermediate_size"`
	NumHiddenLayers         int             `json:"num_hidden_layers"`
	NumAttentionHeads       int             `json:"num_attention_heads"`
	NumKeyValueHeads        int             `json:"num_key_value_heads"`
	NumGlobalKeyValueHeads  int             `json:"num_global_key_value_heads"`
	HeadDim                 int             `json:"head_dim"`
	GlobalHeadDim           int             `json:"global_head_dim"`
	MaxPositionEmbed        int             `json:"max_position_embeddings"`
	RmsNormEps              float64         `json:"rms_norm_eps"`
	SlidingWindow           int             `json:"sliding_window"`
	LayerTypes              []string        `json:"layer_types"`
	FinalLogitSoftcap       *float64        `json:"final_logit_softcapping"`
	HiddenSizePerLayerInput int             `json:"hidden_size_per_layer_input"`
	VocabSizePerLayerInput  int             `json:"vocab_size_per_layer_input"`
	NumKVSharedLayers       int             `json:"num_kv_shared_layers"`
	UseDoubleWideMLP        bool            `json:"use_double_wide_mlp"`
	AttentionKEqV           bool            `json:"attention_k_eq_v"`
	EnableMoEBlock          bool            `json:"enable_moe_block"`
	UseBidirectional        *string         `json:"use_bidirectional_attention"`
	RopeParameters          *struct {
		FullAttention    *gemma4HFRope `json:"full_attention"`
		SlidingAttention *gemma4HFRope `json:"sliding_attention"`
	} `json:"rope_parameters"`
	// PerLayerConfig is how newer transformers saves the full-attention
	// head (instead of global_head_dim / num_global_key_value_heads).
	PerLayerConfig map[string]struct {
		HeadDim          int `json:"head_dim"`
		NumKeyValueHeads int `json:"num_key_value_heads"`
	} `json:"per_layer_config"`
}

func fetchGemma4Config(repo string) (Gemma4Config, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return Gemma4Config{}, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return Gemma4Config{}, err
	}
	var hf gemma4HFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return Gemma4Config{}, fmt.Errorf("fetchGemma4Config: %w", err)
	}
	if hf.ModelType == "gemma4" && hf.TextConfig != nil {
		hf = *hf.TextConfig // Gemma4ForConditionalGeneration: text-only load
		hf.ModelType = "gemma4_text"
	}
	if hf.ModelType != "gemma4_text" {
		return Gemma4Config{}, fmt.Errorf("fetchGemma4Config: model_type %q not supported (gemma4, gemma4_text)", hf.ModelType)
	}
	if hf.EnableMoEBlock {
		return Gemma4Config{}, fmt.Errorf("fetchGemma4Config: MoE blocks (26B-A4B) not supported yet")
	}
	if hf.UseBidirectional != nil && *hf.UseBidirectional != "" && *hf.UseBidirectional != "vision" {
		return Gemma4Config{}, fmt.Errorf("fetchGemma4Config: use_bidirectional_attention=%q not supported", *hf.UseBidirectional)
	}
	if hf.HeadDim == 0 {
		hf.HeadDim = 256
	}
	if hf.GlobalHeadDim == 0 {
		for _, pl := range hf.PerLayerConfig { // full-attention layers' overrides
			if pl.HeadDim != 0 {
				hf.GlobalHeadDim = pl.HeadDim
				if hf.NumGlobalKeyValueHeads == 0 {
					hf.NumGlobalKeyValueHeads = pl.NumKeyValueHeads
				}
				break
			}
		}
	}
	if hf.GlobalHeadDim == 0 {
		hf.GlobalHeadDim = 512
	}
	if hf.NumKeyValueHeads == 0 {
		hf.NumKeyValueHeads = hf.NumAttentionHeads
	}
	if hf.RmsNormEps == 0 {
		hf.RmsNormEps = 1e-6
	}
	if hf.MaxPositionEmbed == 0 {
		hf.MaxPositionEmbed = 131072
	}
	if hf.SlidingWindow == 0 {
		hf.SlidingWindow = 512
	}
	cfg := Gemma4Config{
		VocabSize: hf.VocabSize, HiddenSize: hf.HiddenSize, IntermediateSize: hf.IntermediateSize,
		NumLayers: hf.NumHiddenLayers, NumHeads: hf.NumAttentionHeads, NumKVHeads: hf.NumKeyValueHeads,
		NumGlobalKVHeads: hf.NumGlobalKeyValueHeads, HeadDim: hf.HeadDim, GlobalHeadDim: hf.GlobalHeadDim,
		MaxPosition: hf.MaxPositionEmbed, RopeBase: 1e6, RopeLocalBase: 1e4, PartialRotary: 0.25,
		RmsEps: hf.RmsNormEps, SlidingWindow: hf.SlidingWindow,
		VisionBidirectional: hf.UseBidirectional != nil && *hf.UseBidirectional == "vision",
		PLEDim:              hf.HiddenSizePerLayerInput, PLEVocab: hf.VocabSizePerLayerInput,
		NumKVSharedLayers: hf.NumKVSharedLayers, DoubleWideMLP: hf.UseDoubleWideMLP, AttentionKEqV: hf.AttentionKEqV,
	}
	if cfg.PLEVocab == 0 {
		cfg.PLEVocab = cfg.VocabSize
	}
	if hf.FinalLogitSoftcap != nil {
		cfg.FinalLogitSoftcap = *hf.FinalLogitSoftcap
	}
	if rp := hf.RopeParameters; rp != nil {
		if fa := rp.FullAttention; fa != nil {
			if fa.RopeTheta != 0 {
				cfg.RopeBase = fa.RopeTheta
			}
			if fa.PartialRotaryFactor != 0 {
				cfg.PartialRotary = fa.PartialRotaryFactor
			}
			if fa.RopeType != "" && fa.RopeType != "proportional" && fa.RopeType != "default" {
				return Gemma4Config{}, fmt.Errorf("fetchGemma4Config: full-attention rope_type %q not supported", fa.RopeType)
			}
		}
		if sa := rp.SlidingAttention; sa != nil && sa.RopeTheta != 0 {
			cfg.RopeLocalBase = sa.RopeTheta
		}
	}
	cfg.SlidingLayers = make([]bool, cfg.NumLayers)
	if len(hf.LayerTypes) == cfg.NumLayers {
		for i, t := range hf.LayerTypes {
			cfg.SlidingLayers[i] = t == "sliding_attention"
		}
	} else {
		for i := range cfg.SlidingLayers { // default pattern: every 5th layer global
			cfg.SlidingLayers[i] = (i+1)%5 != 0
		}
	}
	return cfg, nil
}

// safetensorsIndex maps tensor names to the shard holding them and loads
// one tensor at a time straight onto the target device, so a 10 GB
// checkpoint never sits in host memory as a whole.
type safetensorsIndex struct {
	files map[string]*os.File
	where map[string]*os.File
}

func openSafetensorsIndex(paths []string) (*safetensorsIndex, error) {
	idx := &safetensorsIndex{files: map[string]*os.File{}, where: map[string]*os.File{}}
	for _, p := range paths {
		f, err := os.Open(p)
		if err != nil {
			idx.Close()
			return nil, err
		}
		idx.files[p] = f
		infos, _, err := safetensors.Header(f)
		if err != nil {
			idx.Close()
			return nil, fmt.Errorf("%s: %w", p, err)
		}
		for name := range infos {
			idx.where[name] = f
		}
	}
	return idx, nil
}

func (idx *safetensorsIndex) Close() {
	for _, f := range idx.files {
		f.Close()
	}
}

func (idx *safetensorsIndex) has(name string) bool { _, ok := idx.where[name]; return ok }

// load reads a tensor onto dev, cast to dt.
func (idx *safetensorsIndex) load(name string, dt dtype.DType, dev device.Device) (*tensor.Tensor, error) {
	f, ok := idx.where[name]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", name)
	}
	t, err := safetensors.LoadTensor(f, name, dev)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	if t.DType() != dt {
		if t, err = ops.Cast(t, dt); err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
	}
	return t, nil
}

// copyInto loads name and copies it into dst (shape-checked).
func (idx *safetensorsIndex) copyInto(dst *tensor.Tensor, name string) error {
	t, err := idx.load(name, dst.DType(), dst.Device())
	if err != nil {
		return err
	}
	if !dst.Shape().Equal(t.Shape()) {
		return fmt.Errorf("%s: shape mismatch (model %v, HF %v)", name, dst.Shape(), t.Shape())
	}
	return dst.CopyFrom(t)
}

// loadRaw reads a tensor onto the CPU in its stored dtype.
func (idx *safetensorsIndex) loadRaw(name string) (*tensor.Tensor, error) {
	f, ok := idx.where[name]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", name)
	}
	return safetensors.LoadTensor(f, name, device.CPU())
}

// loadLinear fills l from module mod: its 4-bit GPTQ / AWQ tensors when
// the checkpoint quantized it (q set and mod.qweight present), else
// mod.weight at dt (copied in place, or installed into a placeholder).
func (idx *safetensorsIndex) loadLinear(l *nn.Linear, mod string, q *int4Config, dt dtype.DType, dev device.Device) error {
	if q != nil && idx.has(mod+".qweight") {
		hf := map[string]*tensor.Tensor{}
		for _, s := range []string{".qweight", ".qzeros", ".scales", ".g_idx"} {
			if !idx.has(mod + s) {
				continue
			}
			t, err := idx.loadRaw(mod + s)
			if err != nil {
				return err
			}
			hf[mod+s] = t
		}
		w, err := takeInt4(hf, mod, q)
		if err != nil {
			return err
		}
		if w, err = int4To(w, dev); err != nil {
			return err
		}
		if err := l.SetInt4(w); err != nil {
			return fmt.Errorf("%s: %w", mod, err)
		}
		return nil
	}
	if l.W != nil {
		return idx.copyInto(l.W, mod+".weight")
	}
	t, err := idx.load(mod+".weight", dt, dev)
	if err != nil {
		return err
	}
	return l.SetWeight(t)
}

// LoadGemma4FromHF loads a Gemma 4 checkpoint text-only (the vision and
// audio towers of Gemma4ForConditionalGeneration are skipped). weightDT is
// dtype.F32 or dtype.BF16.
func LoadGemma4FromHF(repo string, dev device.Device, weightDT dtype.DType) (*Gemma4, error) {
	cfg, err := fetchGemma4Config(repo)
	if err != nil {
		return nil, err
	}
	cfg.WeightDType = weightDT
	// GPTQ / AWQ: projections load as 4-bit (or plain where the checkpoint
	// left them unquantized), so none is allocated at full size first.
	q4, err := fetchInt4Config(repo)
	if err != nil {
		return nil, err
	}
	cfg.placeholderLinears = q4 != nil
	paths, err := hub.DownloadModelSafetensors(repo, "main")
	if err != nil {
		return nil, err
	}
	idx, err := openSafetensorsIndex(paths)
	if err != nil {
		return nil, err
	}
	defer idx.Close()
	m, err := NewGemma4(cfg, dev)
	if err != nil {
		return nil, err
	}
	prefix := "model."
	for _, p := range []string{"model.language_model.", "language_model.model.", "model."} {
		if idx.has(p + "embed_tokens.weight") {
			prefix = p
			break
		}
	}
	if err := idx.copyInto(m.TokenEmb.Weight, prefix+"embed_tokens.weight"); err != nil {
		return nil, err
	}
	if m.PLEEmb != nil {
		if q4 != nil {
			return nil, fmt.Errorf("gemma4: 4-bit checkpoints with per-layer embeddings (E2B / E4B) are not supported")
		}
		if err := idx.copyInto(m.PLEEmb.Weight, prefix+"embed_tokens_per_layer.weight"); err != nil {
			return nil, err
		}
		// per_layer_model_projection(h) · D^-½: fold the scale into the weight
		// (exact for a linear map, and avoids a BF16-rounded scalar).
		w, err := idx.load(prefix+"per_layer_model_projection.weight", dtype.F32, dev)
		if err != nil {
			return nil, err
		}
		s, err := tensor.New([]float32{float32(1 / math.Sqrt(float64(cfg.HiddenSize)))}, []int{1}, dev)
		if err != nil {
			return nil, err
		}
		if w, err = ops.Mul(w, s); err != nil {
			return nil, err
		}
		if w.DType() != m.PLEProj.W.DType() {
			if w, err = ops.Cast(w, m.PLEProj.W.DType()); err != nil {
				return nil, err
			}
		}
		if err := m.PLEProj.W.CopyFrom(w); err != nil {
			return nil, fmt.Errorf("per_layer_model_projection: %w", err)
		}
		if err := idx.copyInto(m.PLENorm.Weight, prefix+"per_layer_projection_norm.weight"); err != nil {
			return nil, err
		}
	}
	for i, b := range m.Blocks {
		p := prefix + "layers." + strconv.Itoa(i) + "."
		pairs := []struct {
			dst  *tensor.Tensor
			name string
		}{
			{b.InNorm.Weight, p + "input_layernorm.weight"},
			{b.PostAttnNorm.Weight, p + "post_attention_layernorm.weight"},
			{b.PreFFNorm.Weight, p + "pre_feedforward_layernorm.weight"},
			{b.PostFFNorm.Weight, p + "post_feedforward_layernorm.weight"},
			{b.QNorm.Weight, p + "self_attn.q_norm.weight"},
			{b.LayerScalar, p + "layer_scalar"},
		}
		lins := map[string]*nn.Linear{"self_attn.q_proj": b.QProj, "self_attn.o_proj": b.OProj,
			"mlp.gate_proj": b.GateProj, "mlp.up_proj": b.UpProj, "mlp.down_proj": b.DownProj}
		if b.KProj != nil {
			lins["self_attn.k_proj"] = b.KProj
			pairs = append(pairs, struct {
				dst  *tensor.Tensor
				name string
			}{b.KNorm.Weight, p + "self_attn.k_norm.weight"})
		}
		if b.VProj != nil {
			lins["self_attn.v_proj"] = b.VProj
		}
		if b.PLEGate != nil {
			lins["per_layer_input_gate"], lins["per_layer_projection"] = b.PLEGate, b.PLEProj
			pairs = append(pairs, struct {
				dst  *tensor.Tensor
				name string
			}{b.PostPLENorm.Weight, p + "post_per_layer_input_norm.weight"})
		}
		for name, l := range lins {
			if err := idx.loadLinear(l, p+name, q4, cfg.activationDType(), dev); err != nil {
				return nil, err
			}
		}
		for _, pr := range pairs {
			if pr.name == p+"layer_scalar" && !idx.has(pr.name) {
				one, err := tensor.New([]float32{1}, []int{1}, dev)
				if err != nil {
					return nil, err
				}
				if one.DType() != pr.dst.DType() {
					if one, err = ops.Cast(one, pr.dst.DType()); err != nil {
						return nil, err
					}
				}
				if err := pr.dst.CopyFrom(one); err != nil {
					return nil, err
				}
				continue
			}
			if err := idx.copyInto(pr.dst, pr.name); err != nil {
				return nil, err
			}
		}
	}
	if err := idx.copyInto(m.FinalNorm.Weight, prefix+"norm.weight"); err != nil {
		return nil, err
	}
	return m, nil
}
