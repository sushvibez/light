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
	"light/quant"
	"light/tensor"
)

// int4Config is the quantization_config of a GPTQ or AWQ checkpoint.
type int4Config struct {
	QuantMethod      string `json:"quant_method"`
	Bits             int    `json:"bits"`
	GroupSize        int    `json:"group_size"`
	DescAct          bool   `json:"desc_act"`
	CheckpointFormat string `json:"checkpoint_format"`
	Version          string `json:"version"` // AWQ: "gemm"
}

// fetchInt4Config returns the repo's 4-bit quantization config, nil when
// the checkpoint is not GPTQ / AWQ, or an error for variants Light does not
// read (other bit widths, AWQ GEMV / exllama layouts).
func fetchInt4Config(repo string) (*int4Config, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return nil, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var top struct {
		Q *int4Config `json:"quantization_config"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, err
	}
	q := top.Q
	if q == nil || (q.QuantMethod != "gptq" && q.QuantMethod != "awq") {
		return nil, nil
	}
	if q.Bits != 4 {
		return nil, fmt.Errorf("%s: %d-bit weights not supported (4)", q.QuantMethod, q.Bits)
	}
	switch q.QuantMethod {
	case "gptq":
		if f := q.CheckpointFormat; f != "" && f != "gptq" && f != "gptq_v2" {
			return nil, fmt.Errorf("gptq: checkpoint_format %q not supported", f)
		}
	case "awq":
		if v := q.Version; v != "" && v != "gemm" {
			return nil, fmt.Errorf("awq: version %q not supported (gemm)", v)
		}
	}
	return q, nil
}

// llamaLinears names every Linear of a Llama-family model by its HF module
// path (without ".weight"), e.g. "model.layers.3.self_attn.q_proj".
func llamaLinears(m *Llama, prefix string) map[string]*nn.Linear {
	out := map[string]*nn.Linear{}
	for i, b := range m.Blocks {
		p := prefix + "layers." + strconv.Itoa(i) + "."
		for name, l := range map[string]*nn.Linear{
			"self_attn.q_proj": b.QProj, "self_attn.k_proj": b.KProj, "self_attn.v_proj": b.VProj,
			"self_attn.o_proj": b.OProj, "mlp.gate_proj": b.GateProj, "mlp.up_proj": b.UpProj, "mlp.down_proj": b.DownProj,
		} {
			if l != nil {
				out[p+name] = l
			}
		}
	}
	if !m.Cfg.TieEmbed {
		out["lm_head"] = m.LMHead
	}
	return out
}

// installInt4 converts every 4-bit linear of hf (modules with a
// ".qweight") into quant.Int4 on dev and installs it, removing the packed
// tensors from hf. Unquantized linears are left for the regular mapping.
func installInt4(lin map[string]*nn.Linear, hf map[string]*tensor.Tensor, q *int4Config, dev device.Device) error {
	take := func(name string) (*tensor.Tensor, error) {
		t, ok := hf[name]
		if !ok {
			return nil, fmt.Errorf("%s: missing tensor %q", q.QuantMethod, name)
		}
		delete(hf, name)
		return t, nil
	}
	for mod, l := range lin {
		if _, ok := hf[mod+".qweight"]; !ok {
			continue
		}
		qw, err := take(mod + ".qweight")
		if err != nil {
			return err
		}
		qz, err := take(mod + ".qzeros")
		if err != nil {
			return err
		}
		sc, err := take(mod + ".scales")
		if err != nil {
			return err
		}
		if sc, err = ops.Cast(sc, dtype.F32); err != nil {
			return err
		}
		var w quant.Int4
		if q.QuantMethod == "gptq" {
			gi := hf[mod+".g_idx"]
			delete(hf, mod+".g_idx")
			w, err = quant.FromGPTQ(qw, qz, sc, gi, q.GroupSize, q.CheckpointFormat == "gptq_v2")
		} else {
			w, err = quant.FromAWQ(qw, qz, sc, q.GroupSize)
		}
		if err != nil {
			return fmt.Errorf("%s: %w", mod, err)
		}
		ts := []**tensor.Tensor{&w.Codes, &w.Scales, &w.Zeros}
		if w.Perm != nil {
			ts = append(ts, &w.Perm)
		}
		for _, p := range ts {
			if *p, err = (*p).To(dev); err != nil {
				return err
			}
		}
		if err := l.SetInt4(w); err != nil {
			return fmt.Errorf("%s: %w", mod, err)
		}
		// Old AutoGPTQ checkpoints store a bias for every linear, also where
		// the architecture has none; it must be all zeros then.
		if b, ok := hf[mod+".bias"]; ok && l.B == nil {
			b32, err := ops.Cast(b, dtype.F32)
			if err != nil {
				return err
			}
			for _, v := range tensor.MustSlice[float32](b32) {
				if v != 0 {
					return fmt.Errorf("%s: non-zero bias on a layer without one", mod)
				}
			}
			delete(hf, mod+".bias")
		}
	}
	return nil
}

// copyLinearW installs hf[name] as a Linear's weight: nothing to do for a
// 4-bit layer, a cast-and-move into a placeholder (W nil, the 4-bit loaders'
// lazy layers), or the regular in-place copy.
func copyLinearW(l *nn.Linear, hf map[string]*tensor.Tensor, name string, dt dtype.DType, dev device.Device) error {
	if l.W4 != nil {
		return nil
	}
	if l.W != nil {
		return copyExactCast(l.W, hf, name)
	}
	src, ok := hf[name]
	if !ok {
		return fmt.Errorf("missing tensor %q", name)
	}
	delete(hf, name)
	var err error
	if src.DType() != dt {
		if src, err = ops.Cast(src, dt); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
	}
	if src, err = src.To(dev); err != nil {
		return err
	}
	return l.SetWeight(src)
}

// installInt4Llama installs the 4-bit linears of a Llama-family checkpoint
// (GPTQ / AWQ: attention and MLP projections; the LM head usually stays
// 16-bit). MoE expert stacks are not supported in 4 bits.
func installInt4Llama(m *Llama, hf map[string]*tensor.Tensor, q *int4Config, dev device.Device) error {
	for i, b := range m.Blocks {
		if b.MoE != nil {
			return fmt.Errorf("%s: layer %d is MoE; 4-bit experts are not supported", q.QuantMethod, i)
		}
	}
	return installInt4(llamaLinears(m, "model."), hf, q, dev)
}
