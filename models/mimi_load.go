package models

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"

	"light/device"
	"light/hub"
	"light/tensor"
)

type mimiHFConfig struct {
	ModelType          string  `json:"model_type"`
	AudioChannels      int     `json:"audio_channels"`
	NumFilters         int     `json:"num_filters"`
	HiddenSize         int     `json:"hidden_size"`
	KernelSize         int     `json:"kernel_size"`
	LastKernelSize     int     `json:"last_kernel_size"`
	ResidualKernelSize int     `json:"residual_kernel_size"`
	Compress           int     `json:"compress"`
	DilationGrowthRate int     `json:"dilation_growth_rate"`
	NumResidualLayers  int     `json:"num_residual_layers"`
	UpsamplingRatios   []int   `json:"upsampling_ratios"`
	NumHiddenLayers    int     `json:"num_hidden_layers"`
	NumAttentionHeads  int     `json:"num_attention_heads"`
	HeadDim            int     `json:"head_dim"`
	IntermediateSize   int     `json:"intermediate_size"`
	NormEps            float64 `json:"norm_eps"`
	RopeTheta          float64 `json:"rope_theta"`
	SlidingWindow      int     `json:"sliding_window"`
	CodebookSize       int     `json:"codebook_size"`
	CodebookDim        int     `json:"codebook_dim"`
	NumQuantizers      int     `json:"num_quantizers"`
	NumSemantic        int     `json:"num_semantic_quantizers"`
	UpsampleGroups     int     `json:"upsample_groups"`
	SamplingRate       int     `json:"sampling_rate"`
	FrameRate          float64 `json:"frame_rate"`
	UseCausalConv      *bool   `json:"use_causal_conv"`
	UseConvShortcut    bool    `json:"use_conv_shortcut"`
	PadMode            string  `json:"pad_mode"`
	TrimRightRatio     float64 `json:"trim_right_ratio"`
	LayerScaleInit     float64 `json:"layer_scale_initial_scale"`
	HiddenAct          string  `json:"hidden_act"`
}

func fetchMimiConfig(repo string) (MimiConfig, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return MimiConfig{}, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return MimiConfig{}, err
	}
	var hf mimiHFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return MimiConfig{}, fmt.Errorf("fetchMimiConfig: %w", err)
	}
	if hf.ModelType != "mimi" {
		return MimiConfig{}, fmt.Errorf("fetchMimiConfig: model_type %q not supported (mimi)", hf.ModelType)
	}
	if (hf.UseCausalConv != nil && !*hf.UseCausalConv) || hf.UseConvShortcut || (hf.PadMode != "" && hf.PadMode != "constant") ||
		(hf.TrimRightRatio != 0 && hf.TrimRightRatio != 1) || (hf.HiddenAct != "" && hf.HiddenAct != "gelu") {
		return MimiConfig{}, fmt.Errorf("fetchMimiConfig: only causal, constant-padded, gelu Mimi configs are supported")
	}
	cfg := DefaultMimiConfig()
	set := func(dst *int, v int) {
		if v != 0 {
			*dst = v
		}
	}
	set(&cfg.Channels, hf.AudioChannels)
	set(&cfg.NumFilters, hf.NumFilters)
	set(&cfg.HiddenSize, hf.HiddenSize)
	set(&cfg.KernelSize, hf.KernelSize)
	set(&cfg.LastKernelSize, hf.LastKernelSize)
	set(&cfg.ResidualKernelSize, hf.ResidualKernelSize)
	set(&cfg.Compress, hf.Compress)
	set(&cfg.DilationGrowth, hf.DilationGrowthRate)
	set(&cfg.NumResidualLayers, hf.NumResidualLayers)
	set(&cfg.NumLayers, hf.NumHiddenLayers)
	set(&cfg.NumHeads, hf.NumAttentionHeads)
	set(&cfg.HeadDim, hf.HeadDim)
	set(&cfg.IntermediateSize, hf.IntermediateSize)
	set(&cfg.SlidingWindow, hf.SlidingWindow)
	set(&cfg.CodebookSize, hf.CodebookSize)
	set(&cfg.CodebookDim, hf.CodebookDim)
	set(&cfg.NumQuantizers, hf.NumQuantizers)
	set(&cfg.NumSemantic, hf.NumSemantic)
	set(&cfg.UpsampleGroups, hf.UpsampleGroups)
	set(&cfg.SampleRate, hf.SamplingRate)
	if len(hf.UpsamplingRatios) > 0 {
		cfg.UpsamplingRatios = hf.UpsamplingRatios
	}
	if hf.NormEps != 0 {
		cfg.NormEps = hf.NormEps
	}
	if hf.RopeTheta != 0 {
		cfg.RopeBase = hf.RopeTheta
	}
	if hf.FrameRate != 0 {
		cfg.FrameRate = hf.FrameRate
	}
	// latent_stride = encodec frame rate / frame rate.
	hop := 1
	for _, r := range cfg.UpsamplingRatios {
		hop *= r
	}
	encodecRate := float64(cfg.SampleRate) / float64(hop)
	cfg.LatentStride = int(encodecRate/cfg.FrameRate + 0.5)
	if cfg.LatentStride <= 1 {
		cfg.LatentStride = 0
	}
	return cfg, nil
}

// LoadMimiFromHF loads kyutai/mimi (F32 single safetensors).
func LoadMimiFromHF(repo string, dev device.Device) (*Mimi, error) {
	cfg, err := fetchMimiConfig(repo)
	if err != nil {
		return nil, err
	}
	paths, err := hub.DownloadModelSafetensors(repo, "main")
	if err != nil {
		return nil, err
	}
	idx, err := openSafetensorsIndex(paths)
	if err != nil {
		return nil, err
	}
	defer idx.Close()
	m, err := NewMimi(cfg, dev)
	if err != nil {
		return nil, err
	}
	cp := func(dst *tensor.Tensor, name string) error { return idx.copyInto(dst, name) }
	conv := func(c *mimiConv, name string) error {
		if err := cp(c.W, name+".conv.weight"); err != nil {
			return err
		}
		if c.B != nil {
			return cp(c.B, name+".conv.bias")
		}
		return nil
	}
	seanet := func(prefix string, ls []*mimiSeanetLayer) error {
		for i, l := range ls {
			p := prefix + ".layers." + strconv.Itoa(i)
			switch {
			case l.conv != nil:
				if err := conv(l.conv, p); err != nil {
					return err
				}
			case l.convT != nil:
				if err := cp(l.convT.W, p+".conv.weight"); err != nil {
					return err
				}
				if l.convT.B != nil {
					if err := cp(l.convT.B, p+".conv.bias"); err != nil {
						return err
					}
				}
			case l.resnet != nil:
				if err := conv(l.resnet.Conv1, p+".block.1"); err != nil {
					return err
				}
				if err := conv(l.resnet.Conv2, p+".block.3"); err != nil {
					return err
				}
			}
		}
		return nil
	}
	tf := func(prefix string, t *mimiTransformer) error {
		for i, l := range t.Layers {
			p := prefix + ".layers." + strconv.Itoa(i) + "."
			for _, pr := range []struct {
				dst  *tensor.Tensor
				name string
			}{
				{l.InNorm.Gamma, "input_layernorm.weight"}, {l.InNorm.Beta, "input_layernorm.bias"},
				{l.PostAttnNorm.Gamma, "post_attention_layernorm.weight"}, {l.PostAttnNorm.Beta, "post_attention_layernorm.bias"},
				{l.QProj.W, "self_attn.q_proj.weight"}, {l.KProj.W, "self_attn.k_proj.weight"},
				{l.VProj.W, "self_attn.v_proj.weight"}, {l.OProj.W, "self_attn.o_proj.weight"},
				{l.FC1.W, "mlp.fc1.weight"}, {l.FC2.W, "mlp.fc2.weight"},
				{l.AttnScale, "self_attn_layer_scale.scale"}, {l.MLPScale, "mlp_layer_scale.scale"},
			} {
				if err := cp(pr.dst, p+pr.name); err != nil {
					return err
				}
			}
		}
		return nil
	}
	rvq := func(prefix string, q *mimiRVQ) error {
		if q.InProj != nil {
			if err := cp(q.InProj, prefix+".input_proj.weight"); err != nil {
				return err
			}
			if err := cp(q.OutProj, prefix+".output_proj.weight"); err != nil {
				return err
			}
		}
		for i, cb := range q.Layers {
			p := prefix + ".layers." + strconv.Itoa(i) + ".codebook."
			if err := cp(cb.EmbedSum, p+"embed_sum"); err != nil {
				return err
			}
			if err := cp(cb.ClusterUsage, p+"cluster_usage"); err != nil {
				return err
			}
		}
		return nil
	}
	if err := seanet("encoder", m.Encoder); err != nil {
		return nil, err
	}
	if err := tf("encoder_transformer", m.EncoderTF); err != nil {
		return nil, err
	}
	if m.Downsample != nil {
		if err := cp(m.Downsample.W, "downsample.conv.weight"); err != nil {
			return nil, err
		}
		if err := cp(m.Upsample.W, "upsample.conv.weight"); err != nil {
			return nil, err
		}
	}
	if err := rvq("quantizer.semantic_residual_vector_quantizer", m.Semantic); err != nil {
		return nil, err
	}
	if err := rvq("quantizer.acoustic_residual_vector_quantizer", m.Acoustic); err != nil {
		return nil, err
	}
	if err := tf("decoder_transformer", m.DecoderTF); err != nil {
		return nil, err
	}
	if err := seanet("decoder", m.Decoder); err != nil {
		return nil, err
	}
	return m, nil
}
