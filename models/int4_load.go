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
	for mod, l := range lin {
		if _, ok := hf[mod+".qweight"]; !ok {
			continue
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
// (GPTQ / AWQ: attention and MLP projections, and MoE experts when the
// checkpoint quantizes them; routers and the LM head usually stay 16-bit).
func installInt4Llama(m *Llama, hf map[string]*tensor.Tensor, q *int4Config, dev device.Device) error {
	for i, b := range m.Blocks {
		if b.MoE == nil {
			continue
		}
		p := "model.layers." + strconv.Itoa(i) + "."
		if _, ok := hf[p+"mlp.experts.0.gate_proj.qweight"]; !ok {
			return fmt.Errorf("%s: layer %d experts are not 4-bit; mixed MoE layouts are not supported", q.QuantMethod, i)
		}
		if err := installInt4Experts(b.MoE, hf, p, m.Cfg.NumExperts, q, dev); err != nil {
			return err
		}
	}
	return installInt4(llamaLinears(m, "model."), hf, q, dev)
}

// takeInt4 converts the 4-bit module `mod` of hf (qweight, qzeros, scales,
// GPTQ's g_idx) to quant.Int4 on the CPU, removing those tensors from hf.
func takeInt4(hf map[string]*tensor.Tensor, mod string, q *int4Config) (quant.Int4, error) {
	take := func(name string) (*tensor.Tensor, error) {
		t, ok := hf[name]
		if !ok {
			return nil, fmt.Errorf("%s: missing tensor %q", q.QuantMethod, name)
		}
		delete(hf, name)
		return t, nil
	}
	qw, err := take(mod + ".qweight")
	if err != nil {
		return quant.Int4{}, err
	}
	qz, err := take(mod + ".qzeros")
	if err != nil {
		return quant.Int4{}, err
	}
	sc, err := take(mod + ".scales")
	if err != nil {
		return quant.Int4{}, err
	}
	if sc, err = ops.Cast(sc, dtype.F32); err != nil {
		return quant.Int4{}, err
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
		return quant.Int4{}, fmt.Errorf("%s: %w", mod, err)
	}
	return w, nil
}

// int4To moves a 4-bit weight to dev.
func int4To(w quant.Int4, dev device.Device) (quant.Int4, error) {
	ts := []**tensor.Tensor{&w.Codes, &w.Scales, &w.Zeros}
	if w.Perm != nil {
		ts = append(ts, &w.Perm)
	}
	for _, p := range ts {
		var err error
		if *p, err = (*p).To(dev); err != nil {
			return quant.Int4{}, err
		}
	}
	return w, nil
}

// concatInt4 stacks a and b along N (gate rows, then up rows).
func concatInt4(a, b quant.Int4) (quant.Int4, error) {
	if a.Group != b.Group || a.K() != b.K() || a.Perm != nil || b.Perm != nil {
		return quant.Int4{}, fmt.Errorf("concatInt4: incompatible gate / up (group %d / %d, act-order %v / %v)",
			a.Group, b.Group, a.Perm != nil, b.Perm != nil)
	}
	out := quant.Int4{Group: a.Group}
	var err error
	if out.Codes, err = ops.Concat([]*tensor.Tensor{a.Codes, b.Codes}, 0); err != nil {
		return quant.Int4{}, err
	}
	if out.Scales, err = ops.Concat([]*tensor.Tensor{a.Scales, b.Scales}, 0); err != nil {
		return quant.Int4{}, err
	}
	if out.Zeros, err = ops.Concat([]*tensor.Tensor{a.Zeros, b.Zeros}, 0); err != nil {
		return quant.Int4{}, err
	}
	return out, nil
}

// installInt4Experts installs the 4-bit experts of one MoE block
// (prefix p + "mlp.experts.{e}.{gate,up,down}_proj").
func installInt4Experts(moe *nn.SparseMoE, hf map[string]*tensor.Tensor, p string, numExperts int, q *int4Config, dev device.Device) error {
	gu := make([]quant.Int4, numExperts)
	dn := make([]quant.Int4, numExperts)
	for e := range numExperts {
		pe := p + "mlp.experts." + strconv.Itoa(e) + "."
		g, err := takeInt4(hf, pe+"gate_proj", q)
		if err != nil {
			return err
		}
		u, err := takeInt4(hf, pe+"up_proj", q)
		if err != nil {
			return err
		}
		d, err := takeInt4(hf, pe+"down_proj", q)
		if err != nil {
			return err
		}
		c, err := concatInt4(g, u)
		if err != nil {
			return fmt.Errorf("%s: %w", pe, err)
		}
		if gu[e], err = int4To(c, dev); err != nil {
			return err
		}
		if dn[e], err = int4To(d, dev); err != nil {
			return err
		}
	}
	moe.GateUpQ4, moe.DownQ4 = gu, dn
	return nil
}
