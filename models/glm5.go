package models

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/safetensors"
	"light/tensor"
)

// GLM5Config is GLM-5.3's text config (transformers' Glm5NextTextConfig).
type GLM5Config struct {
	VocabSize        int      `json:"vocab_size"`
	HiddenSize       int      `json:"hidden_size"`
	IntermediateSize int      `json:"intermediate_size"`
	MoEIntermediate  int      `json:"moe_intermediate_size"`
	NumLayers        int      `json:"num_hidden_layers"`
	NumHeads         int      `json:"num_attention_heads"`
	NSharedExperts   int      `json:"n_shared_experts"`
	NRoutedExperts   int      `json:"n_routed_experts"`
	ExpertsPerTok    int      `json:"num_experts_per_tok"`
	RoutedScaling    float64  `json:"routed_scaling_factor"`
	KVLoraRank       int      `json:"kv_lora_rank"`
	QLoraRank        int      `json:"q_lora_rank"`
	QKRopeHeadDim    int      `json:"qk_rope_head_dim"`
	QKNopeHeadDim    int      `json:"qk_nope_head_dim"`
	VHeadDim         int      `json:"v_head_dim"`
	NGroup           int      `json:"n_group"`
	TopkGroup        int      `json:"topk_group"`
	NormTopkProb     bool     `json:"norm_topk_prob"`
	RMSNormEps       float64  `json:"rms_norm_eps"`
	TieEmbeddings    bool     `json:"tie_word_embeddings"`
	LayerTypes       []string `json:"layer_types"`
	MLPLayerTypes    []string `json:"mlp_layer_types"`
	IndexTopk        int      `json:"index_topk"`
	IndexHeadDim     int      `json:"index_head_dim"`
	IndexNHeads      int      `json:"index_n_heads"`
	IndexKpool       int      `json:"index_kpool"`
	IndexAlwaysTail  bool     `json:"index_kpool_always_select_tail"`
	IndexerTypes     []string `json:"indexer_types"`
	SwiGLULimit      float64  `json:"swiglu_limit"`
	LinearHeadDim    int      `json:"linear_head_dim"`
	LinearNumHeads   int      `json:"linear_num_heads"`
	LinearConvKernel int      `json:"linear_conv_kernel_dim"`
	LinearLowerBound *float64 `json:"linear_lower_bound"`
	HCMult           int      `json:"hc_mult"`
	HCEps            float64  `json:"hc_eps"`
	HCSinkhornIters  int      `json:"hc_sinkhorn_iters"`
	LinearAttnConfig *struct {
		NumHeads       int      `json:"num_heads"`
		HeadDim        int      `json:"head_dim"`
		GateLowerBound *float64 `json:"gate_lower_bound"`
		ConvKernel     int      `json:"short_conv_kernel_size"`
	} `json:"linear_attn_config"`
}

// GLM5 is GLM-5.3 / GLM-5.3-Flash's language model (Glm5NextTextModel +
// LM head): Kimi delta-attention layers and multi-head latent attention
// layers (no RoPE: mla_use_nope), dense then sparse MLPs (sigmoid router
// with a correction bias, a shared expert, SwiGLU clamped at
// swiglu_limit), and manifold-constrained hyper-connections: hc_mult
// residual streams, collapsed into each block's input and expanded from
// its output by per-token weights, mixed by a Sinkhorn-normalised matrix.
//
// The DeepSeek sparse-attention indexer is not ported: up to index_topk
// tokens it selects every visible token, so dense causal attention is
// exact there; longer sequences return an error.
type GLM5 struct {
	Cfg    GLM5Config
	Embed  *tensor.Tensor // (V, D)
	Layers []*glm5Layer
	NormW  *tensor.Tensor
	LMHead *nn.Linear
	ones   map[int]*tensor.Tensor
	dev    device.Device
}

type glm5HC struct {
	Fn    *tensor.Tensor // (24, N·D) F32, transposed view taken at use
	Base  []float32      // (2 + N)·N
	Scale [3]float32
	fnT   *tensor.Tensor
}

type glm5Expert struct{ Gate, Up, Down *nn.Linear }

type glm5Layer struct {
	AttnHC, FFNHC      glm5HC
	InNormW, PostNormW *tensor.Tensor
	Linear             bool
	// Kimi delta attention.
	Q, K, V, O, B, FA, FB, GA, GB *nn.Linear
	ConvW                         *tensor.Tensor // (3·qkv, kernel)
	DtBias, ONormW                *tensor.Tensor
	ALog                          []float32
	// MLA.
	QA, QB, KVA, KVB, MO *nn.Linear
	QANormW, KVANormW    *tensor.Tensor
	// DSA indexer (nil IdxWqB: none; IdxShared: reuse the previous layer's
	// selection).
	IdxWqB, IdxWk, IdxW, IdxGate *nn.Linear
	IdxKNormW, IdxKNormB, IdxAPE *tensor.Tensor
	IdxShared                    bool
	// MLP: dense, or routed experts + shared expert.
	Dense      *glm5Expert
	Shared     *glm5Expert
	Experts    []*glm5Expert
	RouterW    *tensor.Tensor // (E, D) F32
	RouterBias []float32
	// Stream holds the routed experts instead of Experts when they are
	// streamed from the checkpoint (LIGHT_EXPERT_SLOTS); next is the
	// following streamed layer, whose picks a decode step prefetches.
	Stream *nn.ExpertCache[*glm5Expert]
	next   *glm5Layer
}

// LoadGLM5 loads GLM-5.3 (a folder with config.json and safetensors) onto
// dev at F32 (16-bit linear weights under nn.WithWeightStorage16). With
// LIGHT_EXPERT_SLOTS set, each sparse layer keeps only that many routed
// experts loaded, read from the (memory-mapped) checkpoint when picked.
func LoadGLM5(dir string, dev device.Device) (*GLM5, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var top struct {
		ModelType  string          `json:"model_type"`
		TextConfig json.RawMessage `json:"text_config"`
		Quant      *struct {
			Method    string `json:"quant_method"`
			BlockSize []int  `json:"weight_block_size"`
		} `json:"quantization_config"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, err
	}
	cfg := GLM5Config{RMSNormEps: 1e-5, HCEps: 1e-6, HCSinkhornIters: 20, NGroup: 1, TopkGroup: 1}
	src := raw
	if len(top.TextConfig) > 0 {
		src = top.TextConfig
	}
	if err := json.Unmarshal(src, &cfg); err != nil {
		return nil, fmt.Errorf("glm5 config: %w", err)
	}
	if lc := cfg.LinearAttnConfig; lc != nil {
		if cfg.LinearNumHeads == 0 {
			cfg.LinearNumHeads = lc.NumHeads
		}
		if cfg.LinearHeadDim == 0 {
			cfg.LinearHeadDim = lc.HeadDim
		}
		if cfg.LinearConvKernel == 0 {
			cfg.LinearConvKernel = lc.ConvKernel
		}
		if cfg.LinearLowerBound == nil {
			cfg.LinearLowerBound = lc.GateLowerBound
		}
	}
	if cfg.QKRopeHeadDim != 0 {
		return nil, fmt.Errorf("glm5: qk_rope_head_dim %d (RoPE in the MLA layers) is not supported yet", cfg.QKRopeHeadDim)
	}
	if cfg.NGroup > 1 {
		return nil, fmt.Errorf("glm5: grouped routing (n_group %d) is not supported yet", cfg.NGroup)
	}
	if len(cfg.LayerTypes) != cfg.NumLayers || len(cfg.MLPLayerTypes) != cfg.NumLayers {
		return nil, fmt.Errorf("glm5: %d layer types and %d MLP types for %d layers", len(cfg.LayerTypes), len(cfg.MLPLayerTypes), cfg.NumLayers)
	}
	paths, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	hf, _, err := safetensors.LoadFiles(paths, device.CPU())
	if err != nil {
		return nil, err
	}
	if err := glm5Rename(hf); err != nil {
		return nil, err
	}
	l := &idolLoader{hf: hf, dev: dev}
	if q := top.Quant; q != nil && q.Method == "fp8" && len(q.BlockSize) == 2 {
		l.fp8Block = [2]int{q.BlockSize[0], q.BlockSize[1]} // FP8 weights stay FP8, block-scaled
	}
	prefix := "model.language_model."
	if _, ok := hf[prefix+"embed_tokens.weight"]; !ok {
		prefix = "model."
	}
	m := &GLM5{Cfg: cfg, dev: dev, ones: map[int]*tensor.Tensor{}}
	m.Embed = l.param(prefix + "embed_tokens.weight")
	m.NormW = l.param(prefix + "norm.weight")
	if cfg.TieEmbeddings {
		hf["lm_head.weight"] = hf[prefix+"embed_tokens.weight"]
	}
	m.LMHead = l.linear("lm_head")
	hc := func(p string) glm5HC {
		h := glm5HC{Fn: l.param(p + ".fn"), Base: l.host(p + ".base")}
		if s := l.host(p + ".scale"); len(s) == 3 {
			copy(h.Scale[:], s)
		}
		return h
	}
	mlp := func(p string) *glm5Expert {
		return &glm5Expert{Gate: l.linear(p + ".gate_proj"), Up: l.linear(p + ".up_proj"), Down: l.linear(p + ".down_proj")}
	}
	for i := range cfg.NumLayers {
		p := fmt.Sprintf("%slayers.%d.", prefix, i)
		ly := &glm5Layer{AttnHC: hc(p + "attn_hc"), FFNHC: hc(p + "ffn_hc"),
			InNormW: l.param(p + "input_layernorm.weight"), PostNormW: l.param(p + "post_attention_layernorm.weight")}
		a := p + "self_attn."
		switch cfg.LayerTypes[i] {
		case "linear_attention":
			ly.Linear = true
			ly.Q, ly.K, ly.V, ly.O = l.linear(a+"q_proj"), l.linear(a+"k_proj"), l.linear(a+"v_proj"), l.linear(a+"o_proj")
			ly.B = l.linear(a + "b_proj")
			ly.FA, ly.FB = l.linear(a+"forget_gate.f_a_proj"), l.linear(a+"forget_gate.f_b_proj")
			ly.GA, ly.GB = l.linear(a+"g_a_proj"), l.linear(a+"g_b_proj")
			ly.DtBias, ly.ALog = l.param(a+"forget_gate.dt_bias"), l.host(a+"forget_gate.A_log")
			ly.ONormW = l.param(a + "o_norm.weight")
			if w, ok := hf[a+"conv1d.weight"]; ok { // (C, 1, K) → (C, K)
				if hf[a+"conv1d.weight"], err = w.Reshape(w.Dim(0), w.Dim(2)); err != nil {
					return nil, err
				}
			}
			ly.ConvW = l.param(a + "conv1d.weight")
		default:
			ly.QA, ly.QB = l.linear(a+"q_a_proj"), l.linear(a+"q_b_proj")
			ly.KVA, ly.KVB = l.linear(a+"kv_a_proj_with_mqa"), l.linear(a+"kv_b_proj")
			ly.QANormW, ly.KVANormW = l.param(a+"q_a_layernorm.weight"), l.param(a+"kv_a_layernorm.weight")
			ly.MO = l.linear(a + "o_proj")
			if i < len(cfg.IndexerTypes) && cfg.IndexerTypes[i] == "shared" {
				ly.IdxShared = true
			} else if _, ok := hf[a+"indexer.wq_b.weight"]; ok {
				x := a + "indexer."
				if g, ok := hf[x+"index_kpool_compress_gate"]; ok {
					hf[x+"compress_gate.weight"] = g
					delete(hf, x+"index_kpool_compress_gate")
				}
				ly.IdxWqB, ly.IdxWk = l.linear(x+"wq_b"), l.linear(x+"wk")
				ly.IdxW, ly.IdxGate = l.linear(x+"weights_proj"), l.linear(x+"compress_gate")
				ly.IdxKNormW, ly.IdxKNormB = l.param(x+"k_norm.weight"), l.param(x+"k_norm.bias")
				ly.IdxAPE = l.param(x + "index_kpool_compress_ape")
			}
		}
		if cfg.MLPLayerTypes[i] == "sparse" {
			ly.RouterW = l.param(p + "mlp.gate.weight")
			ly.RouterBias = l.host(p + "mlp.gate.e_score_correction_bias")
			ly.Shared = mlp(p + "mlp.shared_experts")
			load, err := glm5ExpertLoader(l, p+"mlp.experts", cfg)
			if err != nil {
				return nil, err
			}
			if slots := envExpertSlots(); slots > 0 {
				if ly.Stream, err = nn.NewExpertCache(min(slots, cfg.NRoutedExperts), load); err != nil {
					return nil, err
				}
			} else {
				for e := range cfg.NRoutedExperts {
					ex, err := load(e)
					if err != nil {
						return nil, err
					}
					ly.Experts = append(ly.Experts, ex)
				}
				delete(hf, p+"mlp.experts.gate_up_proj")
				delete(hf, p+"mlp.experts.down_proj")
			}
		} else {
			ly.Dense = mlp(p + "mlp")
		}
		if l.err != nil {
			return nil, fmt.Errorf("glm5 layer %d: %w", i, l.err)
		}
		m.Layers = append(m.Layers, ly)
	}
	if l.err != nil {
		return nil, fmt.Errorf("glm5: %w", l.err)
	}
	var prev *glm5Layer
	for _, ly := range m.Layers {
		if ly.Stream != nil {
			if prev != nil {
				prev.next = ly
			}
			prev = ly
		}
	}
	return m, nil
}

// glm5Rename maps the released checkpoint's names onto transformers'
// (its conversion_mapping for glm5_next): hc_{attn,ffn}_{fn,base,scale} →
// {attn,ffn}_hc.*, the forget-gate tensors under forget_gate., and the
// separate q / k / v short convolutions concatenated into conv1d.
func glm5Rename(hf map[string]*tensor.Tensor) error {
	repl := []struct{ from, to string }{
		{".hc_attn_fn", ".attn_hc.fn"}, {".hc_attn_base", ".attn_hc.base"}, {".hc_attn_scale", ".attn_hc.scale"},
		{".hc_ffn_fn", ".ffn_hc.fn"}, {".hc_ffn_base", ".ffn_hc.base"}, {".hc_ffn_scale", ".ffn_hc.scale"},
		{"self_attn.f_a_proj.", "self_attn.forget_gate.f_a_proj."}, {"self_attn.f_b_proj.", "self_attn.forget_gate.f_b_proj."},
		{"self_attn.dt_bias", "self_attn.forget_gate.dt_bias"}, {"self_attn.A_log", "self_attn.forget_gate.A_log"},
	}
	for k, v := range hf {
		for _, r := range repl {
			if i := strings.Index(k, r.from); i >= 0 {
				nk := k[:i] + r.to + k[i+len(r.from):]
				delete(hf, k)
				hf[nk] = v
				break
			}
		}
	}
	for k := range hf {
		p, ok := strings.CutSuffix(k, "self_attn.q_conv1d.weight")
		if !ok {
			continue
		}
		parts := []*tensor.Tensor{hf[k], hf[p+"self_attn.k_conv1d.weight"], hf[p+"self_attn.v_conv1d.weight"]}
		if parts[1] == nil || parts[2] == nil {
			return fmt.Errorf("glm5: %s without its k / v convolutions", k)
		}
		c, err := ops.Concat(parts, 0)
		if err != nil {
			return err
		}
		hf[p+"self_attn.conv1d.weight"] = c
		delete(hf, k)
		delete(hf, p+"self_attn.k_conv1d.weight")
		delete(hf, p+"self_attn.v_conv1d.weight")
	}
	return nil
}

// glm5ExpertLoader returns a reader of routed expert e from the
// checkpoint, per expert (experts.e.*_proj) or sliced from transformers'
// fused (E, 2I, D) gate_up_proj and (E, D, I) down_proj. It never writes
// into l's map, so loads may run concurrently.
func glm5ExpertLoader(l *idolLoader, p string, cfg GLM5Config) (func(e int) (*glm5Expert, error), error) {
	I := cfg.MoEIntermediate
	gu, fused := l.hf[p+".gate_up_proj"]
	down := l.hf[p+".down_proj"]
	if fused && down == nil {
		return nil, fmt.Errorf("%s: gate_up_proj without down_proj", p)
	}
	if !fused {
		for e := range cfg.NRoutedExperts {
			for _, w := range []string{".gate_proj.weight", ".up_proj.weight", ".down_proj.weight"} {
				if _, ok := l.hf[fmt.Sprintf("%s.%d%s", p, e, w)]; !ok {
					return nil, fmt.Errorf("glm5: missing %s.%d%s", p, e, w)
				}
			}
		}
	}
	return func(e int) (*glm5Expert, error) {
		q := fmt.Sprintf("%s.%d", p, e)
		var s *idolLoader
		if fused {
			s = l.sub()
			ge, err := gu.Narrow(0, e, 1)
			if err != nil {
				return nil, err
			}
			de, err := down.Narrow(0, e, 1)
			if err != nil {
				return nil, err
			}
			g, _ := ge.Narrow(1, 0, I)
			u, _ := ge.Narrow(1, I, I)
			for name, t := range map[string]*tensor.Tensor{".gate_proj.weight": g, ".up_proj.weight": u, ".down_proj.weight": de} {
				c, err := t.Contiguous()
				if err != nil {
					return nil, err
				}
				if s.hf[q+name], err = c.Reshape(c.Dim(1), c.Dim(2)); err != nil {
					return nil, err
				}
			}
		} else {
			s = l.sub(q+".gate_proj", q+".up_proj", q+".down_proj")
		}
		ex := &glm5Expert{Gate: s.linear(q + ".gate_proj"), Up: s.linear(q + ".up_proj"), Down: s.linear(q + ".down_proj")}
		if s.err != nil {
			return nil, fmt.Errorf("glm5 %s: %w", q, s.err)
		}
		return ex, nil
	}, nil
}

func (m *GLM5) onesOf(n int) (*tensor.Tensor, error) {
	if t, ok := m.ones[n]; ok {
		return t, nil
	}
	t, err := tensor.Ones(dtype.F32, []int{n}, m.dev)
	if err == nil {
		m.ones[n] = t
	}
	return t, err
}

func (m *GLM5) rms(o *iop, x, w *tensor.Tensor) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) { return ops.RMSNorm(x, w, m.Cfg.RMSNormEps) })
}

// swiglu is down(silu(clamp(gate, max=L)) · clamp(up, ±L)).
func (m *GLM5) swiglu(o *iop, e *glm5Expert, x *tensor.Tensor) *tensor.Tensor {
	L := m.Cfg.SwiGLULimit
	g, u := o.lin(e.Gate, x), o.lin(e.Up, x)
	if L > 0 {
		g = o.do(func() (*tensor.Tensor, error) { return ops.Clamp(g, math.Inf(-1), L) })
		u = o.do(func() (*tensor.Tensor, error) { return ops.Clamp(u, -L, L) })
	}
	return o.lin(e.Down, o.mul(o.un(ops.SiLU, g), u))
}

// hcWeights computes a hyper-connection's per-token weights for streams
// (B, S, N, D): pre (B, S, N, 1), post (B, S, N, 1) and combᵀ (B·S, N, N),
// Sinkhorn-normalised (the small per-token math runs on the host).
func (m *GLM5) hcWeights(o *iop, h *glm5HC, streams *tensor.Tensor) (pre, post, combT *tensor.Tensor) {
	if o.err != nil {
		return nil, nil, nil
	}
	B, S, N, D := streams.Dim(0), streams.Dim(1), streams.Dim(2), streams.Dim(3)
	ones, err := m.onesOf(N * D)
	if err != nil {
		o.err = err
		return nil, nil, nil
	}
	flat := o.reshape(streams, B, S, N*D)
	normed := o.do(func() (*tensor.Tensor, error) { return ops.RMSNorm(flat, ones, m.Cfg.RMSNormEps) })
	if h.fnT == nil {
		h.fnT = o.do(func() (*tensor.Tensor, error) { return h.Fn.Transpose(0, 1) })
	}
	w := o.do(func() (*tensor.Tensor, error) {
		r, err := normed.Reshape(B*S, N*D)
		if err != nil {
			return nil, err
		}
		return ops.MatMul(r, h.fnT)
	})
	if o.err != nil {
		return nil, nil, nil
	}
	wc, err := w.To(device.CPU())
	if err != nil {
		o.err = err
		return nil, nil, nil
	}
	wv := tensor.MustSlice[float32](wc)
	W := (2 + N) * N
	eps := float32(m.Cfg.HCEps)
	preV, postV, combV := make([]float32, B*S*N), make([]float32, B*S*N), make([]float32, B*S*N*N)
	sig := func(x float32) float32 { return float32(1 / (1 + math.Exp(-float64(x)))) }
	c := make([]float32, N*N)
	for t := range B * S {
		r := wv[t*W : (t+1)*W]
		for i := range N {
			preV[t*N+i] = sig(r[i]*h.Scale[0]+h.Base[i]) + eps
			postV[t*N+i] = 2 * sig(r[N+i]*h.Scale[1]+h.Base[N+i])
		}
		for i := range N { // softmax over each row, + eps
			mx := float32(math.Inf(-1))
			for j := range N {
				c[i*N+j] = r[2*N+i*N+j]*h.Scale[2] + h.Base[2*N+i*N+j]
				mx = max(mx, c[i*N+j])
			}
			var s float32
			for j := range N {
				c[i*N+j] = float32(math.Exp(float64(c[i*N+j] - mx)))
				s += c[i*N+j]
			}
			for j := range N {
				c[i*N+j] = c[i*N+j]/s + eps
			}
		}
		colNorm := func() {
			for j := range N {
				var s float32
				for i := range N {
					s += c[i*N+j]
				}
				for i := range N {
					c[i*N+j] /= s + eps
				}
			}
		}
		rowNorm := func() {
			for i := range N {
				var s float32
				for j := range N {
					s += c[i*N+j]
				}
				for j := range N {
					c[i*N+j] /= s + eps
				}
			}
		}
		colNorm()
		for range m.Cfg.HCSinkhornIters - 1 {
			rowNorm()
			colNorm()
		}
		for i := range N { // store combᵀ: out_j = Σ_i comb[i][j]·residual_i
			for j := range N {
				combV[t*N*N+j*N+i] = c[i*N+j]
			}
		}
	}
	pre = o.do(func() (*tensor.Tensor, error) { return tensor.New(preV, []int{B, S, N, 1}, m.dev) })
	post = o.do(func() (*tensor.Tensor, error) { return tensor.New(postV, []int{B, S, N, 1}, m.dev) })
	combT = o.do(func() (*tensor.Tensor, error) { return tensor.New(combV, []int{B * S, N, N}, m.dev) })
	return pre, post, combT
}

// hcCombine is post ⊗ y + combᵀ·residual over the streams.
func (m *GLM5) hcCombine(o *iop, y, residual, post, combT *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	B, S, N, D := residual.Dim(0), residual.Dim(1), residual.Dim(2), residual.Dim(3)
	exp := o.mul(post, o.reshape(y, B, S, 1, D))
	mix := o.do(func() (*tensor.Tensor, error) {
		r, err := residual.Contiguous()
		if err != nil {
			return nil, err
		}
		if r, err = r.Reshape(B*S, N, D); err != nil {
			return nil, err
		}
		return ops.MatMul(combT, r)
	})
	return o.add(exp, o.reshape(mix, B, S, N, D))
}

// l2norm is x / √(Σx² + eps) over the last axis (fla's l2norm).
func (m *GLM5) l2norm(o *iop, x *tensor.Tensor, eps float64) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	D := x.Dim(-1)
	w := o.do(func() (*tensor.Tensor, error) {
		v := make([]float32, D)
		for i := range v {
			v[i] = float32(1 / math.Sqrt(float64(D)))
		}
		return tensor.New(v, []int{D}, m.dev)
	})
	// RMSNorm(x)·(1/√D) with eps/D is x·√D/√(Σx² + eps)·(1/√D).
	return o.do(func() (*tensor.Tensor, error) { return ops.RMSNorm(x, w, eps/float64(D)) })
}

func (m *GLM5) kda(o *iop, ly *glm5Layer, x *tensor.Tensor, c *GLM5Cache, li int) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	cfg := m.Cfg
	B, S := x.Dim(0), x.Dim(1)
	H, hd := cfg.LinearNumHeads, cfg.LinearHeadDim
	qkv := H * hd
	mixed := o.cat(2, o.lin(ly.Q, x), o.lin(ly.K, x), o.lin(ly.V, x)) // (B, S, 3qkv)
	conv := o.do(func() (*tensor.Tensor, error) {
		t, err := mixed.Permute(0, 2, 1)
		if err != nil {
			return nil, err
		}
		if t, err = t.Contiguous(); err != nil {
			return nil, err
		}
		var state *tensor.Tensor
		if c != nil {
			state = c.conv[li]
		}
		out, ns, err := ops.CausalConv1dDepthwise(t, ly.ConvW, state, true)
		if err == nil && c != nil {
			c.conv[li] = ns
		}
		return out, err
	})
	conv = o.perm(conv, 0, 2, 1) // (B, S, 3qkv)
	part := func(i int) *tensor.Tensor { return o.reshape(o.narrow(conv, 2, i*qkv, qkv), B, S, H, hd) }
	q := o.affine(m.l2norm(o, part(0), 1e-6), 1/math.Sqrt(float64(hd)), 0)
	k := m.l2norm(o, part(1), 1e-6)
	v := part(2)
	f := o.add(o.lin(ly.FB, o.lin(ly.FA, x)), ly.DtBias)
	f = o.reshape(f, B, S, H, hd)
	a := o.do(func() (*tensor.Tensor, error) {
		e := make([]float32, H)
		for i, al := range ly.ALog {
			e[i] = float32(math.Exp(float64(al)))
		}
		return tensor.New(e, []int{H, 1}, m.dev)
	})
	var g *tensor.Tensor
	if lb := cfg.LinearLowerBound; lb != nil {
		g = o.affine(o.un(ops.Sigmoid, o.mul(f, a)), *lb, 0)
	} else {
		g = o.mul(o.un(ops.Neg, a), o.un(ops.Softplus, f))
	}
	beta := o.un(ops.Sigmoid, o.lin(ly.B, x))
	if o.err != nil {
		return nil
	}
	var rec *tensor.Tensor
	if c != nil {
		rec = c.rec[li]
	}
	core, final, err := ops.KimiDeltaRule(q, k, v, g, beta, rec)
	if err != nil {
		o.err = err
		return nil
	}
	if c != nil {
		c.rec[li] = final
	}
	gate := o.reshape(o.lin(ly.GB, o.lin(ly.GA, x)), B, S, H, hd)
	out := o.mul(m.rms(o, core, ly.ONormW), o.un(ops.Sigmoid, gate))
	return o.lin(ly.O, o.reshape(out, B, S, qkv))
}

func (m *GLM5) mla(o *iop, ly *glm5Layer, x *tensor.Tensor, c *GLM5Cache, li int, sel *glm5Selection) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	cfg := m.Cfg
	B, S := x.Dim(0), x.Dim(1)
	H, dk, dv := cfg.NumHeads, cfg.QKNopeHeadDim+cfg.QKRopeHeadDim, cfg.VHeadDim
	P := 0 // cached positions before these tokens
	if c != nil {
		P = c.Len
	}
	qResid := m.rms(o, o.lin(ly.QA, x), ly.QANormW)
	q := o.lin(ly.QB, qResid)
	// DSA: past index_topk visible tokens each query attends only to the
	// pools its indexer selects (and its own incomplete tail); up to it the
	// selection is every visible token, the dense causal mask.
	var allowed glm5Selection
	if ly.IdxWqB != nil {
		allowed = m.indexerSelect(o, ly, x, qResid, c, li, P)
		*sel = allowed
	} else if ly.IdxShared {
		allowed = *sel
	} else if cfg.IndexTopk > 0 && P+S > cfg.IndexTopk {
		o.err = fmt.Errorf("glm5: layer %d has no indexer for %d tokens past index_topk %d", li, P+S, cfg.IndexTopk)
		return nil
	}
	q = o.perm(o.reshape(q, B, S, H, dk), 0, 2, 1, 3)
	kv := m.rms(o, o.narrow(o.lin(ly.KVA, x), 2, 0, cfg.KVLoraRank), ly.KVANormW)
	kv = o.reshape(o.lin(ly.KVB, kv), B, S, H, cfg.QKNopeHeadDim+dv)
	k := o.perm(o.narrow(kv, 3, 0, cfg.QKNopeHeadDim), 0, 2, 1, 3)
	v := o.perm(o.narrow(kv, 3, cfg.QKNopeHeadDim, dv), 0, 2, 1, 3)
	if c != nil { // append to the cached keys / values (no RoPE: positions need no fix-up)
		k, v = o.reshape(k, B, H, S, cfg.QKNopeHeadDim), o.reshape(v, B, H, S, dv)
		if c.k[li] != nil {
			k, v = o.cat(2, c.k[li], k), o.cat(2, c.v[li], v)
		}
		c.k[li], c.v[li] = k, v
	}
	T := P + S
	mask := o.do(func() (*tensor.Tensor, error) {
		if allowed != nil { // per batch row: selected tokens only
			mv := make([]float32, B*S*T)
			for b := range B {
				for i := range S {
					row := mv[(b*S+i)*T : (b*S+i+1)*T]
					for j := range row {
						if j > P+i || !allowed[b][i][j] {
							row[j] = float32(math.Inf(-1))
						}
					}
				}
			}
			return tensor.New(mv, []int{B, 1, S, T}, m.dev)
		}
		mv := make([]float32, S*T)
		for i := range S {
			for j := P + i + 1; j < T; j++ {
				mv[i*T+j] = float32(math.Inf(-1))
			}
		}
		return tensor.New(mv, []int{1, 1, S, T}, m.dev)
	})
	a := o.do(func() (*tensor.Tensor, error) {
		qc, err := q.Contiguous()
		if err != nil {
			return nil, err
		}
		kc, err := k.Contiguous()
		if err != nil {
			return nil, err
		}
		vc, err := v.Contiguous()
		if err != nil {
			return nil, err
		}
		return nn.SDPAScaled(qc, kc, vc, mask, 1/math.Sqrt(float64(dk)))
	})
	return o.lin(ly.MO, o.reshape(o.perm(a, 0, 2, 1, 3), B, S, H*dv))
}

// glm5Selection is a DSA selection: [batch][query][key] allowed, or nil for
// every visible key.
type glm5Selection [][][]bool

// indexerSelect is Glm5NextTextIndexer for the new tokens x (B, S, D) after
// P cached ones: index keys LayerNorm(wk·x) pooled kpool at a time by a
// softmax gate (compress_gate·x + ape), pool scores Σ_h w_h·ReLU(q_h·pool /
// √d) with q = wq_b(q_resid) and w = weights_proj(x) / √heads, then per
// query the top index_topk / kpool causally visible complete pools plus its
// incomplete tail. nil while every visible token would be selected. Index
// keys and gate scores are kept in the cache for later tokens.
func (m *GLM5) indexerSelect(o *iop, ly *glm5Layer, x, qResid *tensor.Tensor, c *GLM5Cache, li, P int) glm5Selection {
	if o.err != nil {
		return nil
	}
	cfg := m.Cfg
	B, S := x.Dim(0), x.Dim(1)
	T := P + S
	nh, hd, kp := cfg.IndexNHeads, cfg.IndexHeadDim, cfg.IndexKpool
	kNew := o.do(func() (*tensor.Tensor, error) {
		return ops.LayerNorm(o.lin(ly.IdxWk, x), ly.IdxKNormW, ly.IdxKNormB, 1e-6)
	})
	gNew := o.lin(ly.IdxGate, x)
	kAll, gAll := kNew, gNew
	if c != nil {
		if c.ik[li] != nil {
			kAll, gAll = o.cat(1, c.ik[li], kNew), o.cat(1, c.ig[li], gNew)
		}
		c.ik[li], c.ig[li] = kAll, gAll
	}
	if o.err != nil || cfg.IndexTopk <= 0 || T <= cfg.IndexTopk || kp <= 0 {
		return nil
	}
	Pc := T / kp // complete pools
	selectK := min(cfg.IndexTopk/kp, Pc)
	var scores []float32
	if Pc > 0 {
		kq := o.reshape(o.narrow(kAll, 1, 0, Pc*kp), B, Pc, kp, hd)
		gq := o.add(o.reshape(o.narrow(gAll, 1, 0, Pc*kp), B, Pc, kp, hd), ly.IdxAPE)
		probs := o.do(func() (*tensor.Tensor, error) { return ops.Softmax(gq, 2) })
		weighted := o.mul(probs, kq)
		pool := o.do(func() (*tensor.Tensor, error) { return ops.Sum(weighted, ops.ReduceOptions{Axes: []int{2}}) }) // (B, Pc, hd)
		q := o.reshape(o.lin(ly.IdxWqB, qResid), B, S*nh, hd)
		poolT := o.reshape(o.perm(pool, 0, 2, 1), B, hd, Pc)
		sc := o.do(func() (*tensor.Tensor, error) { return ops.MatMul(q, poolT) }) // (B, S·nh, Pc)
		sc = o.affine(o.un(ops.Relu, sc), 1/math.Sqrt(float64(hd)), 0)
		w := o.affine(o.lin(ly.IdxW, x), 1/math.Sqrt(float64(nh)), 0) // (B, S, nh)
		r := o.mul(o.reshape(sc, B, S, nh, Pc), o.reshape(w, B, S, nh, 1))
		idx := o.do(func() (*tensor.Tensor, error) { return ops.Sum(r, ops.ReduceOptions{Axes: []int{2}}) })
		if o.err != nil {
			return nil
		}
		h, err := idx.To(device.CPU())
		if err != nil {
			o.err = err
			return nil
		}
		scores = tensor.MustSlice[float32](h)
	}
	out := make(glm5Selection, B)
	for b := range B {
		out[b] = make([][]bool, S)
		for i := range S {
			qpos := P + i
			row := make([]bool, T)
			var cand []int
			for p := range Pc {
				if p*kp+kp-1 <= qpos { // the pool's last token is visible
					cand = append(cand, p)
				}
			}
			if len(cand) > selectK {
				s := scores[(b*S+i)*Pc : (b*S+i+1)*Pc]
				sort.SliceStable(cand, func(x, y int) bool { return s[cand[x]] > s[cand[y]] })
				cand = cand[:selectK]
			}
			for _, p := range cand {
				for j := p * kp; j < p*kp+kp; j++ {
					row[j] = true
				}
			}
			if cfg.IndexAlwaysTail { // the incomplete pool the query sits in
				vis := qpos + 1
				for j := vis - vis%kp; j < vis; j++ {
					row[j] = true
				}
			}
			out[b][i] = row
		}
	}
	return out
}

// moe is the routed experts plus the shared expert on x (B, S, D).
func (m *GLM5) moe(o *iop, ly *glm5Layer, x *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	B, S, D := x.Dim(0), x.Dim(1), x.Dim(2)
	flat := o.reshape(x, B*S, D)
	if o.err != nil {
		return nil
	}
	if nx := ly.next; ly.Stream != nil && nx != nil {
		moePrefetch(nx.Stream, flat, func(o *iop, x *tensor.Tensor) [][]moeHit { return m.route(o, nx, x) })
	}
	byExpert := m.route(o, ly, flat)
	out := m.swiglu(o, ly.Shared, flat)
	out = moeRun(o, m.dev, flat, byExpert, out, ly.Stream, ly.Experts, func(e *glm5Expert, xs, wt *tensor.Tensor) *tensor.Tensor {
		return o.mul(m.swiglu(o, e, xs), wt)
	})
	return o.reshape(out, B, S, D)
}

// route is the sigmoid router on flat (T, D): the correction bias picks
// the top k, the unbiased scores (renormalised, scaled) weigh them.
func (m *GLM5) route(o *iop, ly *glm5Layer, flat *tensor.Tensor) [][]moeHit {
	if o.err != nil {
		return nil
	}
	cfg := m.Cfg
	T := flat.Dim(0)
	logits := o.do(func() (*tensor.Tensor, error) {
		wt, err := ly.RouterW.Transpose(0, 1)
		if err != nil {
			return nil, err
		}
		return ops.MatMul(flat, wt)
	})
	if o.err != nil {
		return nil
	}
	lc, err := logits.To(device.CPU())
	if err != nil {
		o.err = err
		return nil
	}
	lv := tensor.MustSlice[float32](lc)
	E, K := cfg.NRoutedExperts, cfg.ExpertsPerTok
	byExpert := make([][]moeHit, E)
	idx := make([]int, E)
	for t := range T {
		sc := make([]float32, E)
		for e := range E {
			sc[e] = float32(1 / (1 + math.Exp(-float64(lv[t*E+e]))))
			idx[e] = e
		}
		sort.SliceStable(idx, func(a, b int) bool { return sc[idx[a]]+ly.RouterBias[idx[a]] > sc[idx[b]]+ly.RouterBias[idx[b]] })
		var sum float32
		for _, e := range idx[:K] {
			sum += sc[e]
		}
		for _, e := range idx[:K] {
			w := sc[e]
			if cfg.NormTopkProb {
				w /= sum + 1e-20
			}
			byExpert[e] = append(byExpert[e], moeHit{t, w * float32(cfg.RoutedScaling)})
		}
	}
	return byExpert
}

// GLM5Cache holds a sequence's state between ForwardWithCache calls: each
// delta-attention layer's short-convolution tail and recurrent state, and
// each MLA layer's keys and values.
type GLM5Cache struct {
	Len       int // positions already processed
	conv, rec []*tensor.Tensor
	ik, ig    []*tensor.Tensor // DSA index keys / gate scores per layer
	k, v      []*tensor.Tensor
}

// NewCache returns an empty cache for m.
func (m *GLM5) NewCache() *GLM5Cache {
	n := len(m.Layers)
	return &GLM5Cache{conv: make([]*tensor.Tensor, n), rec: make([]*tensor.Tensor, n),
		k: make([]*tensor.Tensor, n), v: make([]*tensor.Tensor, n),
		ik: make([]*tensor.Tensor, n), ig: make([]*tensor.Tensor, n)}
}

// Forward returns the logits (B, S, V) of every position for tokens (B, S).
func (m *GLM5) Forward(tokens *tensor.Tensor) (*tensor.Tensor, error) {
	return m.ForwardWithCache(tokens, nil)
}

// ForwardWithCache runs tokens (B, S) after the positions already in cache
// (nil: none, nothing kept) and returns their logits (B, S, V).
func (m *GLM5) ForwardWithCache(tokens *tensor.Tensor, cache *GLM5Cache) (*tensor.Tensor, error) {
	o := &iop{}
	cfg := m.Cfg
	B, S := tokens.Dim(0), tokens.Dim(1)
	D, N := cfg.HiddenSize, cfg.HCMult
	flatTok := o.do(func() (*tensor.Tensor, error) { return tokens.Reshape(B * S) })
	h := o.reshape(o.do(func() (*tensor.Tensor, error) { return ops.IndexSelect(m.Embed, flatTok) }), B, S, 1, D)
	copies := make([]*tensor.Tensor, N)
	for i := range copies {
		copies[i] = h
	}
	streams := o.cat(2, copies...) // (B, S, N, D)
	var sel glm5Selection          // the last full indexer's selection, for shared layers
	for li, ly := range m.Layers {
		pre, post, combT := m.hcWeights(o, &ly.AttnHC, streams)
		x := o.do(func() (*tensor.Tensor, error) {
			p, err := ops.Mul(streams, pre)
			if err != nil {
				return nil, err
			}
			return ops.Sum(p, ops.ReduceOptions{Axes: []int{2}})
		})
		x = m.rms(o, x, ly.InNormW)
		if ly.Linear {
			x = m.kda(o, ly, x, cache, li)
		} else {
			x = m.mla(o, ly, x, cache, li, &sel)
		}
		streams = m.hcCombine(o, x, streams, post, combT)
		pre, post, combT = m.hcWeights(o, &ly.FFNHC, streams)
		x = o.do(func() (*tensor.Tensor, error) {
			p, err := ops.Mul(streams, pre)
			if err != nil {
				return nil, err
			}
			return ops.Sum(p, ops.ReduceOptions{Axes: []int{2}})
		})
		x = m.rms(o, x, ly.PostNormW)
		if ly.Dense != nil {
			x = m.swiglu(o, ly.Dense, x)
		} else {
			x = m.moe(o, ly, x)
		}
		streams = m.hcCombine(o, x, streams, post, combT)
	}
	mean := o.do(func() (*tensor.Tensor, error) { return ops.Mean(streams, ops.ReduceOptions{Axes: []int{2}}) })
	logits := o.lin(m.LMHead, m.rms(o, mean, m.NormW))
	if cache != nil && o.err == nil {
		cache.Len += S
	}
	return logits, o.err
}
