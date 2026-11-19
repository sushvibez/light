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
	"light/hub"
	"light/nn"
	"light/ops"
	"light/quant"
	"light/safetensors"
	"light/tensor"
)

// mlxQuant is the `quantization` block of an MLX checkpoint's config.json
// (mlx-lm / mlx-vlm convert -q): a default group size and bit width for
// affine quantization, and per-module overrides keyed by the module path
// (e.g. routers kept at 8 bits), or false for a module left unquantized.
type mlxQuant struct {
	Group, Bits int
	over        map[string][2]int
}

// parseMLXQuant reads raw (the `quantization` or `quantization_config`
// object); nil when it is not MLX affine quantization (it has a
// quant_method, as GPTQ / AWQ / FP8 configs do).
func parseMLXQuant(raw json.RawMessage) (*mlxQuant, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil, nil
	}
	if _, ok := fields["quant_method"]; ok {
		return nil, nil
	}
	q := &mlxQuant{over: map[string][2]int{}}
	if json.Unmarshal(fields["group_size"], &q.Group) != nil || json.Unmarshal(fields["bits"], &q.Bits) != nil || q.Group <= 0 || q.Bits <= 0 {
		return nil, nil
	}
	if m, ok := fields["mode"]; ok {
		var mode string
		if json.Unmarshal(m, &mode) == nil && mode != "affine" {
			return nil, fmt.Errorf("mlx: quantization mode %q not supported (affine)", mode)
		}
	}
	for k, v := range fields {
		var o struct {
			Group int `json:"group_size"`
			Bits  int `json:"bits"`
		}
		if k == "group_size" || k == "bits" || k == "mode" || json.Unmarshal(v, &o) != nil || o.Bits == 0 {
			continue
		}
		if o.Group == 0 {
			o.Group = q.Group
		}
		q.over[k] = [2]int{o.Group, o.Bits}
	}
	return q, nil
}

// params returns module mod's group size and bits.
func (q *mlxQuant) params(mod string) (int, int) {
	if o, ok := q.over[mod]; ok {
		return o[0], o[1]
	}
	return q.Group, q.Bits
}

// mlxQuantized reports whether module mod is stored quantized in hf.
func mlxQuantized(hf map[string]*tensor.Tensor, mod string) bool {
	_, ok := hf[mod+".scales"]
	return ok
}

// takeMLX removes module mod's weight / scales / biases from hf.
func takeMLX(hf map[string]*tensor.Tensor, mod string) (w, s, b *tensor.Tensor, err error) {
	w, s, b = hf[mod+".weight"], hf[mod+".scales"], hf[mod+".biases"]
	if w == nil || s == nil || b == nil {
		return nil, nil, nil, fmt.Errorf("mlx: %s: weight / scales / biases missing", mod)
	}
	delete(hf, mod+".weight")
	delete(hf, mod+".scales")
	delete(hf, mod+".biases")
	return w, s, b, nil
}

// installMLXLinear installs the quantized module mod into l: 4-bit as
// quant.Int4 (the codes stay packed, ops.MatMulInt4), other widths
// expanded to dt.
func installMLXLinear(l *nn.Linear, hf map[string]*tensor.Tensor, mod string, q *mlxQuant, dt dtype.DType, dev device.Device) error {
	w, s, b, err := takeMLX(hf, mod)
	if err != nil {
		return err
	}
	group, bits := q.params(mod)
	if bits == 4 {
		w4, err := quant.FromMLX(w, s, b, group)
		if err != nil {
			return fmt.Errorf("%s: %w", mod, err)
		}
		if w4, err = int4To(w4, dev); err != nil {
			return err
		}
		return l.SetInt4(w4)
	}
	dense, err := mlxDense(w, s, b, group, bits, dt, dev)
	if err != nil {
		return fmt.Errorf("%s: %w", mod, err)
	}
	return l.SetWeight(dense)
}

// mlxDense expands an MLX weight to dt on dev.
func mlxDense(w, s, b *tensor.Tensor, group, bits int, dt dtype.DType, dev device.Device) (*tensor.Tensor, error) {
	dense, err := quant.DequantMLX(w, s, b, group, bits)
	if err != nil {
		return nil, err
	}
	if dt != dtype.F32 {
		if dense, err = ops.Cast(dense, dt); err != nil {
			return nil, err
		}
	}
	return dense.To(dev)
}

// mlxExpert returns expert e of a stacked MLX module (E, N, ·) as rank-2
// weight / scales / biases views.
func mlxExpert(w, s, b *tensor.Tensor, e int) (we, se, be *tensor.Tensor, err error) {
	pick := func(t *tensor.Tensor) (*tensor.Tensor, error) {
		v, err := t.Narrow(0, e, 1)
		if err != nil {
			return nil, err
		}
		if v, err = v.Contiguous(); err != nil {
			return nil, err
		}
		return v.Reshape(t.Dim(1), t.Dim(2))
	}
	if we, err = pick(w); err != nil {
		return
	}
	if se, err = pick(s); err != nil {
		return
	}
	be, err = pick(b)
	return
}

// installMLXExperts fills moe from MLX's stacked experts under p
// (`switch_mlp.{gate,up,down}_proj`, each (E, N, ·)): with StreamSlots an
// ExpertStream reading each expert when the router first picks it (the
// packed stacks stay memory-mapped) into 4-bit slots, or slots expanded to
// dt for other widths; else 4-bit experts kept packed (GateUpQ4 /
// DownQ4), else expanded stacks at dt.
func installMLXExperts(moe *nn.SparseMoE, hf map[string]*tensor.Tensor, p string, q *mlxQuant, dt dtype.DType, dev device.Device) error {
	type stack struct {
		w, s, b     *tensor.Tensor
		group, bits int
	}
	var st [3]stack // gate, up, down
	for i, name := range []string{"gate_proj", "up_proj", "down_proj"} {
		mod := p + "switch_mlp." + name
		w, s, b, err := takeMLX(hf, mod)
		if err != nil {
			return err
		}
		if w.Rank() != 3 {
			return fmt.Errorf("mlx: %s.weight %v: want stacked experts (E, N, ·)", mod, w.Shape())
		}
		g, bits := q.params(mod)
		st[i] = stack{w, s, b, g, bits}
	}
	E := st[0].w.Dim(0)
	if E != moe.Gate.OutFeatures() {
		return fmt.Errorf("mlx: %sswitch_mlp has %d experts, router %d", p, E, moe.Gate.OutFeatures())
	}
	// dense expands expert e of stack i to F32.
	dense := func(i, e int) (*tensor.Tensor, error) {
		w, s, b, err := mlxExpert(st[i].w, st[i].s, st[i].b, e)
		if err != nil {
			return nil, err
		}
		return quant.DequantMLX(w, s, b, st[i].group, st[i].bits)
	}
	load := func(e int) (*tensor.Tensor, *tensor.Tensor, error) {
		g, err := dense(0, e)
		if err != nil {
			return nil, nil, err
		}
		u, err := dense(1, e)
		if err != nil {
			return nil, nil, err
		}
		d, err := dense(2, e)
		if err != nil {
			return nil, nil, err
		}
		gu, err := ops.Concat([]*tensor.Tensor{g, u}, 0)
		if err != nil {
			return nil, nil, err
		}
		if dt != dtype.F32 {
			if gu, err = ops.Cast(gu, dt); err != nil {
				return nil, nil, err
			}
			if d, err = ops.Cast(d, dt); err != nil {
				return nil, nil, err
			}
		}
		if gu, err = gu.To(dev); err != nil {
			return nil, nil, err
		}
		d, err = d.To(dev)
		return gu, d, err
	}
	packed := st[0].bits == 4 && st[1].bits == 4 && st[2].bits == 4 && st[0].group == st[1].group
	// load4 returns expert e packed on the host: the codes copied out of
	// the mapping, the affine pair rewritten (quant.FromMLX).
	load4 := func(e int) (quant.Int4, quant.Int4, error) {
		var w4 [3]quant.Int4
		for i := range st {
			w, s, b, err := mlxExpert(st[i].w, st[i].s, st[i].b, e)
			if err != nil {
				return quant.Int4{}, quant.Int4{}, err
			}
			if w4[i], err = quant.FromMLX(w, s, b, st[i].group); err != nil {
				return quant.Int4{}, quant.Int4{}, fmt.Errorf("%sswitch_mlp expert %d: %w", p, e, err)
			}
		}
		gu, err := concatInt4(w4[0], w4[1])
		return gu, w4[2], err
	}
	if moe.StreamSlots > 0 {
		var s *nn.ExpertStream
		var err error
		if packed {
			s, err = nn.NewExpertStreamQ4(moe.StreamSlots, dev, load4)
		} else {
			s, err = nn.NewExpertStream(moe.StreamSlots, dev, load)
		}
		if err != nil {
			return err
		}
		moe.Stream = s
		return nil
	}
	if packed {
		gu, dn := make([]quant.Int4, E), make([]quant.Int4, E)
		for e := range E {
			g, d, err := load4(e)
			if err != nil {
				return err
			}
			if gu[e], err = int4To(g, dev); err != nil {
				return err
			}
			if dn[e], err = int4To(d, dev); err != nil {
				return err
			}
		}
		moe.GateUpQ4, moe.DownQ4 = gu, dn
		return nil
	}
	gus, dns := make([]*tensor.Tensor, E), make([]*tensor.Tensor, E)
	for e := range E {
		var err error
		if gus[e], dns[e], err = load(e); err != nil {
			return err
		}
	}
	var err error
	if moe.GateUp, err = ops.Stack(gus, 0); err != nil {
		return err
	}
	moe.Down, err = ops.Stack(dns, 0)
	return err
}

// loraAdapter is a loaded LoRA adapter: per module path (normalized, see
// loraModule) its A (r, in) and B (out, r), and the scale.
type loraAdapter struct {
	A, B  map[string]*tensor.Tensor
	Scale float64
	// TopK is the routed width the adapter was trained for (edge0's "K"
	// metadata), 0 when unstated.
	TopK int
	Path string
}

// loraModule strips a checkpoint's wrapper prefixes from a module path:
// "language_model.model.layers.3.mlp.gate_proj", "base_model.model.model.
// layers.3…" and "model.layers.3…" all become "layers.3.mlp.gate_proj".
func loraModule(name string) string {
	for {
		trimmed := false
		for _, p := range []string{"base_model.model.", "language_model.", "model."} {
			if strings.HasPrefix(name, p) {
				name, trimmed = name[len(p):], true
			}
		}
		if !trimmed {
			return name
		}
	}
}

// findLoRA locates the LoRA adapter for repo: LIGHT_LORA names a file
// (or "off"); otherwise a top-level adapters.safetensors (mlx-lm),
// adapter_model.safetensors (PEFT) or lora*.safetensors (edge0) in the
// repo or directory. "" when there is none.
func findLoRA(repo string) (string, error) {
	switch v := os.Getenv("LIGHT_LORA"); strings.ToLower(v) {
	case "":
	case "off", "none", "0":
		return "", nil
	default:
		if _, err := os.Stat(v); err == nil {
			return v, nil
		}
		return hub.Download(repo, "main", v)
	}
	var names []string
	if st, err := os.Stat(repo); err == nil && st.IsDir() {
		ents, err := os.ReadDir(repo)
		if err != nil {
			return "", err
		}
		for _, e := range ents {
			names = append(names, e.Name())
		}
	} else {
		files, err := hub.ListFiles(repo, "main")
		if err != nil {
			return "", err
		}
		for _, f := range files {
			names = append(names, f.Name)
		}
	}
	for _, n := range names {
		if n == "adapters.safetensors" || n == "adapter_model.safetensors" ||
			(strings.HasPrefix(n, "lora") && strings.HasSuffix(n, ".safetensors") && !strings.Contains(n, "/")) {
			return hub.Download(repo, "main", n)
		}
	}
	return "", nil
}

// loadLoRA reads an adapter file in any of three layouts:
//
//   - edge0: `<module>.lora_A` (r, in), `<module>.lora_B` (out, r), scale
//     alpha / r from the safetensors metadata
//   - PEFT: `<module>.lora_A.weight`, `<module>.lora_B.weight`, scale
//     lora_alpha / r from the adjacent adapter_config.json
//   - mlx-lm: `<module>.lora_a` (in, r), `<module>.lora_b` (r, out) (its
//     LoRALinear: y + scale · (x lora_a) lora_b), scale from
//     adapter_config.json's lora_parameters (mlx-lm's default 20)
func loadLoRA(path string) (*loraAdapter, error) {
	ts, meta, err := safetensors.LoadFiles([]string{path}, device.CPU())
	if err != nil {
		return nil, err
	}
	var cfg struct {
		R          float64 `json:"r"`
		LoraAlpha  float64 `json:"lora_alpha"`
		LoraParams *struct {
			Rank  float64 `json:"rank"`
			Scale float64 `json:"scale"`
		} `json:"lora_parameters"`
		FineTuneType string `json:"fine_tune_type"`
	}
	if raw, err := os.ReadFile(filepath.Join(filepath.Dir(path), "adapter_config.json")); err == nil {
		if err := json.Unmarshal(raw, &cfg); err != nil {
			return nil, fmt.Errorf("lora: adapter_config.json: %w", err)
		}
	}
	if cfg.FineTuneType == "dora" {
		return nil, fmt.Errorf("lora: DoRA adapters not supported")
	}
	ad := &loraAdapter{A: map[string]*tensor.Tensor{}, B: map[string]*tensor.Tensor{}, Path: path}
	// edge0 stores its metadata as one JSON string under __metadata__.
	if m, ok := meta["__metadata__"]; ok {
		var inner map[string]string
		if json.Unmarshal([]byte(m), &inner) == nil {
			for k, v := range inner {
				meta[k] = v
			}
		}
	}
	mlxLayout := false
	for name, t := range ts {
		var mod string
		var isA bool
		switch {
		case strings.HasSuffix(name, ".lora_A"), strings.HasSuffix(name, ".lora_B"):
			mod, isA = name[:len(name)-7], strings.HasSuffix(name, "A")
		case strings.HasSuffix(name, ".lora_A.weight"), strings.HasSuffix(name, ".lora_B.weight"):
			mod, isA = name[:len(name)-14], name[len(name)-8] == 'A'
		case strings.HasSuffix(name, ".lora_a"), strings.HasSuffix(name, ".lora_b"):
			mod, isA, mlxLayout = name[:len(name)-7], strings.HasSuffix(name, "a"), true
			if t, err = t.Transpose(0, 1); err != nil {
				return nil, err
			}
		default:
			return nil, fmt.Errorf("lora: %s: unexpected tensor %q", filepath.Base(path), name)
		}
		if isA {
			ad.A[loraModule(mod)] = t
		} else {
			ad.B[loraModule(mod)] = t
		}
	}
	for mod := range ad.A {
		if ad.B[mod] == nil {
			return nil, fmt.Errorf("lora: %s has lora_A but no lora_B", mod)
		}
	}
	if len(ad.A) != len(ad.B) || len(ad.A) == 0 {
		return nil, fmt.Errorf("lora: %s: %d A / %d B matrices", filepath.Base(path), len(ad.A), len(ad.B))
	}
	num := func(k string) float64 {
		f, _ := strconv.ParseFloat(meta[k], 64)
		return f
	}
	switch {
	case mlxLayout:
		ad.Scale = 20
		if lp := cfg.LoraParams; lp != nil && lp.Scale != 0 {
			ad.Scale = lp.Scale
		}
	case num("alpha") > 0 && num("r") > 0:
		ad.Scale = num("alpha") / num("r")
	case cfg.LoraAlpha > 0 && cfg.R > 0:
		ad.Scale = cfg.LoraAlpha / cfg.R
	default:
		return nil, fmt.Errorf("lora: %s: no alpha / r in its metadata or adapter_config.json", filepath.Base(path))
	}
	ad.TopK = int(num("K"))
	return ad, nil
}

// apply installs the adapter on the Linears named by lin (normalized
// module paths); every adapter module must exist.
func (ad *loraAdapter) apply(lin map[string]*nn.Linear, dev device.Device) error {
	for mod, a := range ad.A {
		l := lin[mod]
		if l == nil {
			return fmt.Errorf("lora: module %q not in the model", mod)
		}
		b := ad.B[mod]
		if a.Dim(1) != l.InFeatures() || b.Dim(0) != l.OutFeatures() {
			return fmt.Errorf("lora: %s: A %v, B %v for a (%d → %d) layer", mod, a.Shape(), b.Shape(), l.InFeatures(), l.OutFeatures())
		}
		lr, err := nn.NewLoRA(a, b, ad.Scale, dev)
		if err != nil {
			return fmt.Errorf("lora: %s: %w", mod, err)
		}
		l.LoRA = lr
	}
	return nil
}
