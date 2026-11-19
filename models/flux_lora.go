package models

import (
	"fmt"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/safetensors"
	"light/tensor"
)

// FluxLoRA is a parsed LoRA adapter for the Flux transformer: per
// diffusers module name (as LoadFluxTransformer names them, e.g.
// "transformer_blocks.0.attn.to_q"), A (r, in), B (out, r) and the alpha
// the file carries (0: none, scale is the strength alone). Files in the
// diffusers / PEFT layout ("transformer.<module>.lora_A.weight"), the
// kohya layout ("lora_unet_<module with _>.lora_down.weight" + ".alpha")
// and the BFL module names either uses (double_blocks.N.img_attn.qkv,
// single_blocks.N.linear1, ...) are read; fused qkv / linear1 adapters
// split by output rows onto the separate projections. Keys for other
// parts (text encoders) are ignored.
type FluxLoRA struct {
	Modules map[string]*fluxLoRAModule
	Skipped []string // keys of other parts, for the record
	applied map[string]bool
}

type fluxLoRAModule struct {
	A, B  *tensor.Tensor
	Alpha float64
}

// LoadFluxLoRA reads a LoRA safetensors file.
func LoadFluxLoRA(path string) (*FluxLoRA, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	ts, _, err := safetensors.Load(f, device.CPU())
	if err != nil {
		return nil, fmt.Errorf("lora %s: %w", path, err)
	}
	return ParseFluxLoRA(ts)
}

var (
	reDouble = regexp.MustCompile(`^double_blocks[._](\d+)[._](img|txt)_(attn[._]qkv|attn[._]proj|mlp[._]0|mlp[._]2|mod[._]lin)$`)
	reSingle = regexp.MustCompile(`^single_blocks[._](\d+)[._](linear1|linear2|modulation[._]lin)$`)
	// diffusers module names with "_" for "." (kohya "lora_transformer_" keys)
	reDiffU = regexp.MustCompile(`^(transformer_blocks|single_transformer_blocks)_(\d+)_(.*)$`)
)

// ParseFluxLoRA groups a LoRA state dict by module.
func ParseFluxLoRA(ts map[string]*tensor.Tensor) (*FluxLoRA, error) {
	type raw struct {
		a, b  *tensor.Tensor
		alpha float64
	}
	byMod := map[string]*raw{}
	ad := &FluxLoRA{Modules: map[string]*fluxLoRAModule{}, applied: map[string]bool{}}
	for key, t := range ts {
		mod, part, ok := splitLoRAKey(key)
		if !ok {
			ad.Skipped = append(ad.Skipped, key)
			continue
		}
		r := byMod[mod]
		if r == nil {
			r = &raw{}
			byMod[mod] = r
		}
		switch part {
		case "A":
			r.a = t
		case "B":
			r.b = t
		case "alpha":
			v, err := scalarF64(t)
			if err != nil {
				return nil, fmt.Errorf("lora %s: %w", key, err)
			}
			r.alpha = v
		}
	}
	for mod, r := range byMod {
		if r.a == nil || r.b == nil {
			return nil, fmt.Errorf("lora: module %s has A %v and B %v", mod, r.a != nil, r.b != nil)
		}
		if r.a.Rank() != 2 || r.b.Rank() != 2 || r.a.Dim(0) != r.b.Dim(1) {
			return nil, fmt.Errorf("lora: module %s: A %v, B %v", mod, r.a.Shape(), r.b.Shape())
		}
		targets, splits := fluxLoRATargets(mod)
		if targets == nil {
			return nil, fmt.Errorf("lora: module %s is not a Flux transformer linear", mod)
		}
		if len(targets) == 1 {
			ad.Modules[targets[0]] = &fluxLoRAModule{A: r.a, B: r.b, Alpha: r.alpha}
			continue
		}
		// Fused projection: B's rows split onto the targets, A shared.
		// qkv is three equal parts; single linear1 is (D, D, D, 4D).
		out := r.b.Dim(0)
		switch {
		case splits[0] == -1 && out%3 == 0:
			splits = []int{out / 3, out / 3, out / 3}
		case splits[0] == -2 && out%7 == 0:
			D := out / 7
			splits = []int{D, D, D, 4 * D}
		default:
			return nil, fmt.Errorf("lora: module %s: B has %d rows, not a fused projection", mod, out)
		}
		off := 0
		for i, tgt := range targets {
			b, err := r.b.Narrow(0, off, splits[i])
			if err != nil {
				return nil, err
			}
			if b, err = b.Contiguous(); err != nil {
				return nil, err
			}
			ad.Modules[tgt] = &fluxLoRAModule{A: r.a, B: b, Alpha: r.alpha}
			off += splits[i]
		}
	}
	sort.Strings(ad.Skipped)
	return ad, nil
}

func scalarF64(t *tensor.Tensor) (float64, error) {
	c, err := t.To(device.CPU())
	if err != nil {
		return 0, err
	}
	if c.DType() != dtype.F32 {
		if c, err = ops.Cast(c, dtype.F32); err != nil {
			return 0, err
		}
	}
	v := tensor.MustSlice[float32](c)
	if len(v) != 1 {
		return 0, fmt.Errorf("alpha %v is not a scalar", t.Shape())
	}
	return float64(v[0]), nil
}

// splitLoRAKey returns the module name (BFL or diffusers form, dotted) and
// which part of the adapter the key holds: "A" (lora_A / lora_down /
// lora.down), "B", or "alpha". Keys of other parts are not ok.
func splitLoRAKey(key string) (mod, part string, ok bool) {
	k := key
	for _, pfx := range []string{"transformer.", "diffusion_model.", "model.diffusion_model.", "base_model.model."} {
		k = strings.TrimPrefix(k, pfx)
	}
	kohya := false
	switch {
	case strings.HasPrefix(k, "lora_unet_"):
		k, kohya = strings.TrimPrefix(k, "lora_unet_"), true
	case strings.HasPrefix(k, "lora_transformer_"):
		k, kohya = strings.TrimPrefix(k, "lora_transformer_"), true
	case strings.HasPrefix(k, "lora_te") || strings.HasPrefix(k, "text_encoder"):
		return "", "", false
	}
	suffixes := []struct{ s, part string }{
		{".lora_A.weight", "A"}, {".lora_B.weight", "B"},
		{".lora_down.weight", "A"}, {".lora_up.weight", "B"},
		{".lora.down.weight", "A"}, {".lora.up.weight", "B"},
		{".lora_A.default.weight", "A"}, {".lora_B.default.weight", "B"},
		{".alpha", "alpha"},
	}
	for _, s := range suffixes {
		if strings.HasSuffix(k, s.s) {
			mod = strings.TrimSuffix(k, s.s)
			if kohya {
				mod = kohyaToDotted(mod)
			}
			return mod, s.part, true
		}
	}
	return "", "", false
}

// kohyaToDotted undoes kohya's "." → "_" on a module name, for the names
// the regexes above know.
func kohyaToDotted(m string) string {
	if reDouble.MatchString(m) || reSingle.MatchString(m) {
		return m // the regexes accept either separator
	}
	if sm := reDiffU.FindStringSubmatch(m); sm != nil {
		rest := sm[3]
		// attn_to_q → attn.to_q, ff_net_0_proj → ff.net.0.proj, norm1_linear → norm1.linear, ...
		rest = strings.NewReplacer("attn_to_out_0", "attn.to_out.0", "attn_to_add_out", "attn.to_add_out",
			"attn_add_q_proj", "attn.add_q_proj", "attn_add_k_proj", "attn.add_k_proj", "attn_add_v_proj", "attn.add_v_proj",
			"attn_to_q", "attn.to_q", "attn_to_k", "attn.to_k", "attn_to_v", "attn.to_v",
			"ff_context_net_0_proj", "ff_context.net.0.proj", "ff_context_net_2", "ff_context.net.2",
			"ff_net_0_proj", "ff.net.0.proj", "ff_net_2", "ff.net.2",
			"norm1_context_linear", "norm1_context.linear", "norm1_linear", "norm1.linear", "norm_linear", "norm.linear").Replace(rest)
		return sm[1] + "." + sm[2] + "." + rest
	}
	return m
}

// fluxLoRATargets maps a module name to the diffusers linears it adapts
// and, for fused ones, the output rows per target (-1: equal thirds).
func fluxLoRATargets(mod string) (targets []string, splits []int) {
	if sm := reDouble.FindStringSubmatch(mod); sm != nil {
		p := "transformer_blocks." + sm[1] + "."
		img := sm[2] == "img"
		switch strings.NewReplacer("_", ".").Replace(sm[3]) {
		case "attn.qkv":
			if img {
				return []string{p + "attn.to_q", p + "attn.to_k", p + "attn.to_v"}, []int{-1}
			}
			return []string{p + "attn.add_q_proj", p + "attn.add_k_proj", p + "attn.add_v_proj"}, []int{-1}
		case "attn.proj":
			if img {
				return []string{p + "attn.to_out.0"}, nil
			}
			return []string{p + "attn.to_add_out"}, nil
		case "mlp.0":
			if img {
				return []string{p + "ff.net.0.proj"}, nil
			}
			return []string{p + "ff_context.net.0.proj"}, nil
		case "mlp.2":
			if img {
				return []string{p + "ff.net.2"}, nil
			}
			return []string{p + "ff_context.net.2"}, nil
		case "mod.lin":
			if img {
				return []string{p + "norm1.linear"}, nil
			}
			return []string{p + "norm1_context.linear"}, nil
		}
	}
	if sm := reSingle.FindStringSubmatch(mod); sm != nil {
		p := "single_transformer_blocks." + sm[1] + "."
		switch strings.NewReplacer("_", ".").Replace(sm[2]) {
		case "linear1":
			return []string{p + "attn.to_q", p + "attn.to_k", p + "attn.to_v", p + "proj_mlp"}, []int{-2}
		case "linear2":
			return []string{p + "proj_out"}, nil
		case "modulation.lin":
			return []string{p + "norm.linear"}, nil
		}
	}
	switch mod {
	case "img_in":
		return []string{"x_embedder"}, nil
	case "txt_in":
		return []string{"context_embedder"}, nil
	case "time_in.in_layer":
		return []string{"time_text_embed.timestep_embedder.linear_1"}, nil
	case "time_in.out_layer":
		return []string{"time_text_embed.timestep_embedder.linear_2"}, nil
	case "vector_in.in_layer":
		return []string{"time_text_embed.text_embedder.linear_1"}, nil
	case "vector_in.out_layer":
		return []string{"time_text_embed.text_embedder.linear_2"}, nil
	case "guidance_in.in_layer":
		return []string{"time_text_embed.guidance_embedder.linear_1"}, nil
	case "guidance_in.out_layer":
		return []string{"time_text_embed.guidance_embedder.linear_2"}, nil
	case "final_layer.linear":
		return []string{"proj_out"}, nil
	case "final_layer.adaLN_modulation.1":
		return []string{"norm_out.linear"}, nil
	}
	if strings.HasPrefix(mod, "transformer_blocks.") || strings.HasPrefix(mod, "single_transformer_blocks.") || fluxTopLinears[mod] {
		return []string{mod}, nil
	}
	return nil, nil
}

var fluxTopLinears = map[string]bool{"x_embedder": true, "context_embedder": true, "proj_out": true, "norm_out.linear": true,
	"time_text_embed.timestep_embedder.linear_1": true, "time_text_embed.timestep_embedder.linear_2": true,
	"time_text_embed.text_embedder.linear_1": true, "time_text_embed.text_embedder.linear_2": true,
	"time_text_embed.guidance_embedder.linear_1": true, "time_text_embed.guidance_embedder.linear_2": true}

// linears names the block's linears as the checkpoint does.
func (b *fluxDoubleBlock) linears(i int) map[string]*nn.Linear {
	p := "transformer_blocks." + strconv.Itoa(i) + "."
	return map[string]*nn.Linear{
		p + "norm1.linear": b.Norm1, p + "norm1_context.linear": b.Norm1Ctx,
		p + "attn.to_q": b.Img.Q, p + "attn.to_k": b.Img.K, p + "attn.to_v": b.Img.V,
		p + "attn.add_q_proj": b.Ctx.Q, p + "attn.add_k_proj": b.Ctx.K, p + "attn.add_v_proj": b.Ctx.V,
		p + "attn.to_out.0": b.Out, p + "attn.to_add_out": b.CtxOut,
		p + "ff.net.0.proj": b.FF1, p + "ff.net.2": b.FF2, p + "ff_context.net.0.proj": b.CtxFF1, p + "ff_context.net.2": b.CtxFF2,
	}
}

func (b *fluxSingleBlock) linears(i int) map[string]*nn.Linear {
	p := "single_transformer_blocks." + strconv.Itoa(i) + "."
	return map[string]*nn.Linear{
		p + "norm.linear": b.Norm, p + "attn.to_q": b.Attn.Q, p + "attn.to_k": b.Attn.K, p + "attn.to_v": b.Attn.V,
		p + "proj_mlp": b.ProjMLP, p + "proj_out": b.ProjOut,
	}
}

func (m *FluxTransformer) topLinears() map[string]*nn.Linear {
	ls := map[string]*nn.Linear{"x_embedder": m.XEmbed, "context_embedder": m.ContextEmbed, "proj_out": m.ProjOut, "norm_out.linear": m.NormOut,
		"time_text_embed.timestep_embedder.linear_1": m.TimeL1, "time_text_embed.timestep_embedder.linear_2": m.TimeL2,
		"time_text_embed.text_embedder.linear_1": m.TextL1, "time_text_embed.text_embedder.linear_2": m.TextL2}
	if m.GuideL1 != nil {
		ls["time_text_embed.guidance_embedder.linear_1"], ls["time_text_embed.guidance_embedder.linear_2"] = m.GuideL1, m.GuideL2
	}
	return ls
}

// fluxLoRAUse is an adapter at a strength, applied to blocks as they load.
type fluxLoRAUse struct {
	ad       *FluxLoRA
	strength float32
}

// ApplyLoRA adds ad at strength (1 = as trained; the file's alpha / rank
// folds in) to the transformer's linears. A streamed transformer applies
// it to each block as the block is built. Modules the model lacks are an
// error (for a streamed model, at the first forward pass).
func (m *FluxTransformer) ApplyLoRA(ad *FluxLoRA, strength float32) error {
	if err := applyFluxLoRA(ad, strength, m.topLinears(), m.XEmbed.W.Device()); err != nil {
		return err
	}
	if m.stream != nil {
		m.stream.loras = append(m.stream.loras, fluxLoRAUse{ad, strength})
		return nil
	}
	for i, b := range m.Double {
		if err := applyFluxLoRA(ad, strength, b.linears(i), m.XEmbed.W.Device()); err != nil {
			return err
		}
	}
	for i, b := range m.Single {
		if err := applyFluxLoRA(ad, strength, b.linears(i), m.XEmbed.W.Device()); err != nil {
			return err
		}
	}
	return ad.unused(m)
}

func applyFluxLoRA(ad *FluxLoRA, strength float32, lins map[string]*nn.Linear, dev device.Device) error {
	for name, l := range lins {
		mod := ad.Modules[name]
		if mod == nil {
			continue
		}
		if mod.A.Dim(1) != l.InFeatures() || mod.B.Dim(0) != l.OutFeatures() {
			return fmt.Errorf("lora: %s: A %v, B %v for a (%d → %d) layer", name, mod.A.Shape(), mod.B.Shape(), l.InFeatures(), l.OutFeatures())
		}
		scale := float64(strength)
		if mod.Alpha != 0 {
			scale *= mod.Alpha / float64(mod.A.Dim(0))
		}
		lr, err := nn.NewLoRA(mod.A, mod.B, scale, dev)
		if err != nil {
			return fmt.Errorf("lora: %s: %w", name, err)
		}
		if err := l.AddLoRA(lr); err != nil {
			return fmt.Errorf("lora: %s: %w", name, err)
		}
		ad.applied[name] = true
	}
	return nil
}

// unused reports adapter modules the model never matched (a resident
// model: after all blocks; names from a different architecture).
func (ad *FluxLoRA) unused(m *FluxTransformer) error {
	var missing []string
	for name := range ad.Modules {
		if !ad.applied[name] {
			missing = append(missing, name)
		}
	}
	if len(missing) == 0 {
		return nil
	}
	sort.Strings(missing)
	if len(missing) > 4 {
		missing = append(missing[:4], fmt.Sprintf("… %d more", len(missing)-4))
	}
	return fmt.Errorf("lora: %d modules not in the model (%d double, %d single blocks): %s", len(missing), m.Cfg.NumLayers, m.Cfg.NumSingleLayers, strings.Join(missing, ", "))
}
