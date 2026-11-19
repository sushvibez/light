package models

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/safetensors"
	"light/tensor"
)

// Single-file Flux transformers: flux1-schnell.safetensors and the
// community checkpoints (FP8, ComfyUI "model.diffusion_model." wrapped)
// use Black Forest Labs' module names (double_blocks.N.img_attn.qkv,
// single_blocks.N.linear1, final_layer.adaLN_modulation.1, ...). They are
// renamed to the diffusers names LoadFluxTransformer expects, as
// diffusers' convert_flux_transformer_checkpoint_to_diffusers does: fused
// qkv / linear1 become row views onto the separate projections (no copy),
// the final modulation's halves swap from [shift, scale] to [scale,
// shift]. A file in diffusers names loads as is. The config comes from a
// sidecar (<file>.config.json or config.json beside it) or the shapes.

// FluxConfigAt is the transformer config for a diffusers folder or a
// single-file checkpoint (sidecar, else derived from the tensor shapes:
// FLUX.1's rope axes 16 / 56 / 56 for 128-wide heads).
func FluxConfigAt(path string) (FluxConfig, error) {
	var cfg FluxConfig
	st, err := os.Stat(path)
	if err != nil {
		return cfg, err
	}
	if st.IsDir() {
		return readFluxConfig(filepath.Join(path, "config.json"))
	}
	for _, side := range []string{path + ".config.json", filepath.Join(filepath.Dir(path), "config.json")} {
		if _, err := os.Stat(side); err == nil {
			return readFluxConfig(side)
		}
	}
	if strings.HasSuffix(strings.ToLower(path), ".gguf") {
		shapes, err := ggufShapes(path)
		if err != nil {
			return cfg, fmt.Errorf("flux %s: %w", filepath.Base(path), err)
		}
		return fluxConfigFromShapes(shapes)
	}
	f, err := os.Open(path)
	if err != nil {
		return cfg, err
	}
	defer f.Close()
	infos, _, err := safetensors.Header(f)
	if err != nil {
		return cfg, fmt.Errorf("flux %s: %w", filepath.Base(path), err)
	}
	shapes := map[string][]int{}
	for k, ti := range infos {
		shapes[strings.TrimPrefix(k, "model.diffusion_model.")] = ti.Shape
	}
	return fluxConfigFromShapes(shapes)
}

func readFluxConfig(path string) (FluxConfig, error) {
	var cfg FluxConfig
	raw, err := os.ReadFile(path)
	if err != nil {
		return cfg, err
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return cfg, fmt.Errorf("flux config %s: %w", path, err)
	}
	return cfg, nil
}

// fluxConfigFromShapes derives the config from a BFL or diffusers state
// dict's shapes.
func fluxConfigFromShapes(shapes map[string][]int) (FluxConfig, error) {
	var cfg FluxConfig
	get := func(names ...string) []int {
		for _, n := range names {
			if s, ok := shapes[n]; ok {
				return s
			}
		}
		return nil
	}
	in := get("img_in.weight", "x_embedder.weight")
	txt := get("txt_in.weight", "context_embedder.weight")
	vec := get("vector_in.in_layer.weight", "time_text_embed.text_embedder.linear_1.weight")
	qn := get("double_blocks.0.img_attn.norm.query_norm.scale", "transformer_blocks.0.attn.norm_q.weight")
	if in == nil || txt == nil || vec == nil || qn == nil {
		return cfg, fmt.Errorf("flux: not a Flux transformer checkpoint (no img_in / txt_in / vector_in / query norm)")
	}
	D, hd := in[0], qn[0]
	cfg.InChannels, cfg.JointDim, cfg.PooledDim = in[1], txt[1], vec[1]
	cfg.HeadDim, cfg.NumHeads, cfg.PatchSize = hd, D/hd, 1
	cfg.GuidanceEmbeds = get("guidance_in.in_layer.weight", "time_text_embed.guidance_embedder.linear_1.weight") != nil
	for k := range shapes {
		var pfx string
		var n *int
		switch {
		case strings.HasPrefix(k, "double_blocks."), strings.HasPrefix(k, "transformer_blocks."):
			pfx, n = "", &cfg.NumLayers
		case strings.HasPrefix(k, "single_blocks."), strings.HasPrefix(k, "single_transformer_blocks."):
			pfx, n = "", &cfg.NumSingleLayers
		default:
			continue
		}
		_ = pfx
		parts := strings.SplitN(k, ".", 3)
		if i, err := strconv.Atoi(parts[1]); err == nil && i+1 > *n {
			*n = i + 1
		}
	}
	if hd == 128 {
		cfg.AxesDims = []int{16, 56, 56}
	} else {
		return cfg, fmt.Errorf("flux: %d-wide heads: put the rope axes in a sidecar config.json (axes_dims_rope)", hd)
	}
	return cfg, nil
}

// loadFluxSingleFile maps the file and returns its tensors under diffusers
// names.
func loadFluxSingleFile(path string) (map[string]*tensor.Tensor, error) {
	raw, _, err := safetensors.LoadFiles([]string{path}, device.CPU())
	if err != nil {
		return nil, err
	}
	hf := make(map[string]*tensor.Tensor, len(raw))
	bfl := false
	for k, t := range raw {
		k = strings.TrimPrefix(k, "model.diffusion_model.")
		bfl = bfl || strings.HasPrefix(k, "double_blocks.")
		hf[k] = t
	}
	if !bfl {
		return hf, nil
	}
	return fluxFromBFL(hf)
}

// fluxFromBFL renames a BFL state dict to diffusers names.
func fluxFromBFL(src map[string]*tensor.Tensor) (map[string]*tensor.Tensor, error) {
	out := map[string]*tensor.Tensor{}
	var err error
	mv := func(from, to string) {
		if t, ok := src[from]; ok {
			out[to] = t
			delete(src, from)
		}
	}
	both := func(from, to string) { mv(from+".weight", to+".weight"); mv(from+".bias", to+".bias") }
	// rows splits t's leading dim into parts of the given sizes (views, or
	// byte ranges of a quantized tensor).
	rows := func(name string, sizes []int, tos []string) {
		t, ok := src[name]
		if !ok || err != nil {
			if err == nil {
				err = fmt.Errorf("flux: missing %s", name)
			}
			return
		}
		delete(src, name)
		parts, e := splitRows(t, sizes)
		if e != nil {
			err = fmt.Errorf("flux: %s: %w", name, e)
			return
		}
		for i, v := range parts {
			out[tos[i]] = v
		}
	}
	D := 0
	if t, ok := src["img_in.weight"]; ok {
		D = t.Dim(0)
	} else {
		return nil, fmt.Errorf("flux: no img_in.weight")
	}
	both("time_in.in_layer", "time_text_embed.timestep_embedder.linear_1")
	both("time_in.out_layer", "time_text_embed.timestep_embedder.linear_2")
	both("vector_in.in_layer", "time_text_embed.text_embedder.linear_1")
	both("vector_in.out_layer", "time_text_embed.text_embedder.linear_2")
	both("guidance_in.in_layer", "time_text_embed.guidance_embedder.linear_1")
	both("guidance_in.out_layer", "time_text_embed.guidance_embedder.linear_2")
	both("txt_in", "context_embedder")
	both("img_in", "x_embedder")
	both("final_layer.linear", "proj_out")
	for _, s := range []string{"weight", "bias"} { // [shift, scale] → [scale, shift]
		t, ok := src["final_layer.adaLN_modulation.1."+s]
		if !ok {
			return nil, fmt.Errorf("flux: missing final_layer.adaLN_modulation.1.%s", s)
		}
		delete(src, "final_layer.adaLN_modulation.1."+s)
		if t, err = widen(t); err != nil { // rearranged on the host in F32 (2D x D, small)
			return nil, err
		}
		shift, e1 := t.Narrow(0, 0, D)
		scale, e2 := t.Narrow(0, D, D)
		if e1 != nil || e2 != nil {
			return nil, fmt.Errorf("flux: final modulation %s: %v %v", s, e1, e2)
		}
		if out["norm_out.linear."+s], err = ops.Concat([]*tensor.Tensor{scale, shift}, 0); err != nil {
			return nil, err
		}
	}
	for i := 0; ; i++ {
		q := "double_blocks." + strconv.Itoa(i) + "."
		if _, ok := src[q+"img_mod.lin.weight"]; !ok {
			break
		}
		p := "transformer_blocks." + strconv.Itoa(i) + "."
		both(q+"img_mod.lin", p+"norm1.linear")
		both(q+"txt_mod.lin", p+"norm1_context.linear")
		for _, s := range []string{"weight", "bias"} {
			rows(q+"img_attn.qkv."+s, []int{D, D, D}, []string{p + "attn.to_q." + s, p + "attn.to_k." + s, p + "attn.to_v." + s})
			rows(q+"txt_attn.qkv."+s, []int{D, D, D}, []string{p + "attn.add_q_proj." + s, p + "attn.add_k_proj." + s, p + "attn.add_v_proj." + s})
		}
		mv(q+"img_attn.norm.query_norm.scale", p+"attn.norm_q.weight")
		mv(q+"img_attn.norm.key_norm.scale", p+"attn.norm_k.weight")
		mv(q+"txt_attn.norm.query_norm.scale", p+"attn.norm_added_q.weight")
		mv(q+"txt_attn.norm.key_norm.scale", p+"attn.norm_added_k.weight")
		both(q+"img_attn.proj", p+"attn.to_out.0")
		both(q+"txt_attn.proj", p+"attn.to_add_out")
		both(q+"img_mlp.0", p+"ff.net.0.proj")
		both(q+"img_mlp.2", p+"ff.net.2")
		both(q+"txt_mlp.0", p+"ff_context.net.0.proj")
		both(q+"txt_mlp.2", p+"ff_context.net.2")
	}
	for i := 0; ; i++ {
		q := "single_blocks." + strconv.Itoa(i) + "."
		if _, ok := src[q+"modulation.lin.weight"]; !ok {
			break
		}
		p := "single_transformer_blocks." + strconv.Itoa(i) + "."
		both(q+"modulation.lin", p+"norm.linear")
		for _, s := range []string{"weight", "bias"} {
			rows(q+"linear1."+s, []int{D, D, D, 4 * D}, []string{p + "attn.to_q." + s, p + "attn.to_k." + s, p + "attn.to_v." + s, p + "proj_mlp." + s})
		}
		both(q+"linear2", p+"proj_out")
		mv(q+"norm.query_norm.scale", p+"attn.norm_q.weight")
		mv(q+"norm.key_norm.scale", p+"attn.norm_k.weight")
	}
	if err != nil {
		return nil, err
	}
	if len(src) > 0 {
		var left []string
		for k := range src {
			left = append(left, k)
		}
		if len(left) > 3 {
			left = left[:3]
		}
		return nil, fmt.Errorf("flux: %d unknown tensors in the single-file checkpoint (%s …)", len(src), strings.Join(left, ", "))
	}
	return out, nil
}

func castCPU(t *tensor.Tensor, dt dtype.DType) (*tensor.Tensor, error) { return ops.Cast(t, dt) }
