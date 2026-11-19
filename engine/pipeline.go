package engine

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"light/diffusion"
	"light/gguf"
	"light/safetensors"
)

// PipelineSlot names one component of an image pipeline: a folder of a
// Hub repo (Repo + Sub, e.g. "black-forest-labs/FLUX.1-schnell" +
// "transformer"), or a local folder / file (Path). Empty means the
// pipeline's own component (ImageOptions.Repo).
type PipelineSlot struct {
	Repo string `json:"repo,omitempty"`
	Sub  string `json:"sub,omitempty"`
	Path string `json:"path,omitempty"`
}

// LoraSpec is a LoRA adapter file and its strength (1 = as trained).
type LoraSpec struct {
	Path  string  `json:"path"`
	Scale float32 `json:"scale"`
}

// PipelineSpec composes an image pipeline from separately chosen parts, as
// ComfyUI does: the transformer / UNet, the VAE, the text encoders in the
// family's order (FLUX.1: CLIP-L, T5-XXL) and any LoRAs. Slots left empty
// come from ImageOptions.Repo.
type PipelineSpec struct {
	Family       string         `json:"family"` // "flux", "flux2", "sdxl", "sd15", "zimage"
	Transformer  PipelineSlot   `json:"transformer"`
	VAE          PipelineSlot   `json:"vae"`
	TextEncoders []PipelineSlot `json:"text_encoders"`
	Loras        []LoraSpec     `json:"loras"`
	Scheduler    PipelineSlot   `json:"scheduler,omitempty"`
	// Run is where the pipeline runs; Stages overrides it per stage
	// ("text_encode", "denoise", "decode").
	Run    RunTarget            `json:"run,omitempty"`
	Stages map[string]RunTarget `json:"stages,omitempty"`
}

// RunTarget is where a pipeline (or one of its stages) executes.
type RunTarget struct {
	// Target is "local/cpu", "local/gpu", "remote/api", "remote/colab" or
	// "remote/ssh" ("" = local/gpu when one exists, else local/cpu).
	Target string `json:"target,omitempty"`
	// remote/api: the Light server's base URL (its /v1/images/generations)
	// and the environment variable holding its key.
	Endpoint string `json:"endpoint,omitempty"`
	KeyEnv   string `json:"key_env,omitempty"`
	// remote/colab: the kept session's name and the GPU to ask for.
	Session string `json:"session,omitempty"`
	GPU     string `json:"gpu,omitempty"`
	// remote/ssh: user@host and the light binary there.
	Host  string `json:"host,omitempty"`
	Light string `json:"light,omitempty"`
}

// PipelineStages are the stages a RunTarget can be set for.
var PipelineStages = []string{"text_encode", "denoise", "decode"}

// Validate checks the targets; remote ones are recognised but not run yet.
func (p *PipelineSpec) Validate() error {
	check := func(stage string, r RunTarget) error {
		switch r.Target {
		case "", "local/cpu", "local/gpu":
			return nil
		case "remote/api":
			if r.Endpoint == "" {
				return fmt.Errorf("pipeline %s: remote/api needs an endpoint", stage)
			}
		case "remote/colab":
			if r.Session == "" {
				return fmt.Errorf("pipeline %s: remote/colab needs a session name", stage)
			}
		case "remote/ssh":
			if r.Host == "" {
				return fmt.Errorf("pipeline %s: remote/ssh needs a host", stage)
			}
		default:
			return fmt.Errorf("pipeline %s: unknown target %q", stage, r.Target)
		}
		return fmt.Errorf("pipeline %s: %s is not wired yet (local/cpu and local/gpu run today)", stage, r.Target)
	}
	if err := check("run", p.Run); err != nil {
		return err
	}
	for stage, r := range p.Stages {
		known := false
		for _, s := range PipelineStages {
			known = known || s == stage
		}
		if !known {
			return fmt.Errorf("pipeline: unknown stage %q, want one of %v", stage, PipelineStages)
		}
		if err := check(stage, r); err != nil {
			return err
		}
	}
	if d, ok := p.Stages["decode"]; ok && d.Target != "" && d.Target != p.stageTarget("denoise") {
		return fmt.Errorf("pipeline: decode must run where denoise runs for now")
	}
	return nil
}

// stageTarget is the effective target of a stage.
func (p *PipelineSpec) stageTarget(stage string) string {
	if r, ok := p.Stages[stage]; ok && r.Target != "" {
		return r.Target
	}
	return p.Run.Target
}

// LoadPipelineSpec reads a pipeline JSON file.
func LoadPipelineSpec(path string) (*PipelineSpec, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var p PipelineSpec
	if err := json.Unmarshal(raw, &p); err != nil {
		return nil, fmt.Errorf("pipeline %s: %w", path, err)
	}
	return &p, p.Validate()
}

// LibraryFile is one file of the local model library (LibraryFiles).
type LibraryFile struct {
	Name   string `json:"name"`
	Path   string `json:"path"`
	Bytes  int64  `json:"bytes"`
	Format string `json:"format"` // "safetensors" or "gguf"
	DType  string `json:"dtype"`  // the dominant weight dtype: "bf16", "f16", "f32", "fp8_e4m3", "q4_k", ...
	Family string `json:"family"` // what the keys say: "flux", "flux2", "sdxl", "sd15", "lora", "vae", "clip", "t5", "qwen3", or ""
	Layout string `json:"layout"` // "diffusers", "bfl" (single-file FLUX), "comfy" ("model.diffusion_model." prefix), "kohya" (LoRA), "peft" (LoRA), or ""
}

// LibraryKinds are the library's folders, named as ComfyUI names them so
// an existing library can be symlinked in.
var LibraryKinds = []string{"checkpoints", "loras", "vae", "clip"}

// LibraryDir is the local model library: LIGHT_MODELS_DIR or
// ~/.cache/light/models, holding one folder per LibraryKinds.
func LibraryDir() string {
	if d := os.Getenv("LIGHT_MODELS_DIR"); d != "" {
		return d
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache", "light", "models")
}

// LibraryFiles lists the safetensors and GGUF files under the library's
// kind folder (one of LibraryKinds), following symlinks one level, with
// what their headers say about them. A missing folder is an empty list.
func LibraryFiles(kind string) ([]LibraryFile, error) {
	ok := false
	for _, k := range LibraryKinds {
		ok = ok || k == kind
	}
	if !ok {
		return nil, fmt.Errorf("library: kind %q, want one of %v", kind, LibraryKinds)
	}
	dir := filepath.Join(LibraryDir(), kind)
	ents, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return []LibraryFile{}, nil
	}
	if err != nil {
		return nil, err
	}
	var out []LibraryFile
	for _, e := range ents {
		p := filepath.Join(dir, e.Name())
		st, err := os.Stat(p) // through symlinks
		if err != nil || st.IsDir() {
			continue
		}
		f := LibraryFile{Name: e.Name(), Path: p, Bytes: st.Size()}
		switch strings.ToLower(filepath.Ext(e.Name())) {
		case ".safetensors":
			f.Format = "safetensors"
		case ".gguf":
			f.Format = "gguf"
		default:
			continue
		}
		probeLibraryFile(&f)
		out = append(out, f)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	if out == nil {
		out = []LibraryFile{}
	}
	return out, nil
}

// probeLibraryFile reads the header for the dtype, family and layout;
// a file it cannot parse keeps empty fields.
func probeLibraryFile(f *LibraryFile) {
	r, err := os.Open(f.Path)
	if err != nil {
		return
	}
	defer r.Close()
	bytesBy := map[string]int64{}
	var keys []string
	switch f.Format {
	case "safetensors":
		infos, _, err := safetensors.Header(r)
		if err != nil {
			return
		}
		for name, ti := range infos {
			keys = append(keys, name)
			bytesBy[strings.ToLower(ti.DType)] += int64(ti.DataOffsets[1] - ti.DataOffsets[0])
		}
	case "gguf":
		g, err := gguf.Open(r)
		if err != nil {
			return
		}
		for name, ti := range g.Tensors {
			keys = append(keys, name)
			n := int64(1)
			for _, d := range ti.Dims {
				n *= int64(d)
			}
			bytesBy[strings.ToLower(ti.DType.String())] += n
		}
		if arch, ok := g.String("general.architecture"); ok {
			f.Family = ggufFamily(arch)
		}
	}
	var best string
	for dt, n := range bytesBy {
		if best == "" || n > bytesBy[best] {
			best = dt
		}
	}
	f.DType = normDType(best)
	fam, layout := keysFamily(keys)
	if f.Family == "" {
		f.Family = fam
	}
	f.Layout = layout
}

func normDType(s string) string {
	switch s {
	case "f8_e4m3", "f8e4m3", "float8_e4m3fn":
		return "fp8_e4m3"
	case "f8_e5m2", "f8e5m2", "float8_e5m2":
		return "fp8_e5m2"
	case "bf16", "bfloat16":
		return "bf16"
	case "f16", "float16":
		return "f16"
	case "f32", "float32":
		return "f32"
	}
	return s
}

func ggufFamily(arch string) string {
	switch strings.ToLower(arch) {
	case "flux":
		return "flux"
	case "flux2":
		return "flux2"
	case "sdxl":
		return "sdxl"
	case "sd1":
		return "sd15"
	case "t5", "t5encoder":
		return "t5"
	case "clip":
		return "clip"
	case "qwen3":
		return "qwen3"
	}
	return strings.ToLower(arch)
}

// keysFamily recognises a checkpoint by its tensor names.
func keysFamily(keys []string) (family, layout string) {
	has := func(pfx string) bool {
		for _, k := range keys {
			if strings.HasPrefix(k, pfx) || strings.Contains(k, "."+pfx) {
				return true
			}
		}
		return false
	}
	any := func(subs ...string) bool {
		for _, k := range keys {
			for _, s := range subs {
				if strings.Contains(k, s) {
					return true
				}
			}
		}
		return false
	}
	switch {
	case any(".lora_A.", ".lora_B.", ".lora_down.", ".lora_up."), has("lora_unet_"), has("lora_te"):
		layout = "peft"
		if has("lora_unet_") || has("lora_te") {
			layout = "kohya"
		}
		return "lora", layout
	case has("model.diffusion_model."):
		layout = "comfy"
	case has("double_blocks.") && has("single_blocks."):
		layout = "bfl"
	case has("transformer_blocks.") || has("down_blocks.") || has("input_blocks."):
		layout = "diffusers"
	}
	switch {
	case has("double_blocks.") || (has("transformer_blocks.") && any("add_q_proj")):
		if any("txt_mod", "norm_added_q") && !any("time_text_embed.text_embedder", "vector_in") {
			return "flux2", layout
		}
		return "flux", layout
	case has("input_blocks.") || has("down_blocks.") && any("attentions.0.transformer_blocks.0.attn1"):
		if any("label_emb", "add_embedding") {
			return "sdxl", layout
		}
		return "sd15", layout
	case has("decoder.up.") || has("decoder.up_blocks.") || has("encoder.down."):
		return "vae", layout
	case has("text_model.encoder.layers."):
		return "clip", layout
	case has("encoder.block.") && any("SelfAttention"):
		return "t5", layout
	case has("model.layers.") && any("q_norm", "k_norm"):
		return "qwen3", layout
	}
	return "", layout
}

// resolveSlot turns a slot into a local folder: the Hub component folder
// (downloaded on demand) or the path as given. Single-file checkpoints are
// reported as such for the caller to handle.
func resolveSlot(s PipelineSlot, fallbackRepo, sub string) (dir string, file string, err error) {
	switch {
	case s.Path != "":
		st, err := os.Stat(s.Path)
		if err != nil {
			return "", "", err
		}
		if st.IsDir() {
			return s.Path, "", nil
		}
		return "", s.Path, nil
	case s.Repo != "":
		root, err := diffusion.Snapshot(s.Repo)
		if err != nil {
			return "", "", err
		}
		if s.Sub == "" {
			s.Sub = sub
		}
		return filepath.Join(root, s.Sub), "", nil
	}
	if fallbackRepo == "" {
		return "", "", fmt.Errorf("pipeline: the %s slot is empty and no repo is set", sub)
	}
	root, err := diffusion.Snapshot(fallbackRepo)
	if err != nil {
		return "", "", err
	}
	return filepath.Join(root, sub), "", nil
}

// fluxComponents maps a FLUX.1 spec to component folder overrides.
func fluxComponents(o ImageOptions) (map[string]string, error) {
	return familyComponents(o, []string{"text_encoder", "text_encoder_2"})
}

// flux2Components maps a FLUX.2 spec (one text encoder) to overrides.
func flux2Components(o ImageOptions) (map[string]string, error) {
	return familyComponents(o, []string{"text_encoder"})
}

// familyComponents resolves the transformer, VAE, the family's text
// encoders (in order) and the scheduler slot.
func familyComponents(o ImageOptions, encoders []string) (map[string]string, error) {
	sp := o.Pipeline
	comp := map[string]string{}
	set := func(slot PipelineSlot, sub string) error {
		dir, file, err := resolveSlot(slot, o.Repo, sub)
		if err != nil {
			return err
		}
		if file != "" {
			if sub != "transformer" {
				return fmt.Errorf("pipeline: %s: single-file checkpoints (%s) are not loadable yet for this part; use a diffusers folder or a Hub repo", sub, filepath.Base(file))
			}
			comp[sub] = file // BFL / diffusers-named single file, see models.FluxConfigAt
			return nil
		}
		comp[sub] = dir
		return nil
	}
	if err := set(sp.Transformer, "transformer"); err != nil {
		return nil, err
	}
	if err := set(sp.VAE, "vae"); err != nil {
		return nil, err
	}
	encs := sp.TextEncoders
	for len(encs) < len(encoders) {
		encs = append(encs, PipelineSlot{})
	}
	if len(encs) > len(encoders) {
		return nil, fmt.Errorf("pipeline: %d text encoders given, the family has %d (%v)", len(encs), len(encoders), encoders)
	}
	for i, sub := range encoders {
		if err := set(encs[i], sub); err != nil {
			return nil, err
		}
	}
	if sp.Scheduler != (PipelineSlot{}) {
		if err := set(sp.Scheduler, "scheduler"); err != nil {
			return nil, err
		}
	}
	return comp, nil
}

// fluxLoras turns the spec's LoRAs into load options, checking the files.
func fluxLoras(sp *PipelineSpec) ([]diffusion.FluxLoRA, error) {
	var out []diffusion.FluxLoRA
	for _, l := range sp.Loras {
		if _, err := os.Stat(l.Path); err != nil {
			return nil, fmt.Errorf("pipeline lora: %w", err)
		}
		sc := l.Scale
		if sc == 0 {
			sc = 1
		}
		out = append(out, diffusion.FluxLoRA{Path: l.Path, Scale: sc})
	}
	return out, nil
}
