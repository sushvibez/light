package models

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"sort"
	"strings"

	"light/device"
	"light/dtype"
	"light/hub"
	"light/ops"
	"light/tensor"
)

// Masked-diffusion language models: LLaDA (GSAI-ML/LLaDA-8B-Instruct,
// model_type llada) and Dream (Dream-org/Dream-v0-Instruct-7B, model_type
// Dream). Both are ordinary Llama / Qwen2 decoders run with full
// bidirectional attention over the prompt plus a canvas of mask tokens;
// each sampling step predicts every masked position and commits the most
// confident ones. LLaDA predicts position i from its own logits; Dream from
// position i-1's (its logits are shifted right by one).

// LoadLLaDA loads a LLaDA checkpoint (its own OLMo-style names:
// model.transformer.blocks.N.{q,k,v}_proj, attn_out, ff_proj (gate),
// up_proj, ff_out (down), attn_norm, ff_norm; wte, ln_f, ff_out) as a Llama.
func LoadLLaDA(repo string, dev device.Device, weightDT dtype.DType) (*Llama, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return nil, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c struct {
		DModel    int     `json:"d_model"`
		NHeads    int     `json:"n_heads"`
		NKVHeads  *int    `json:"n_kv_heads"`
		NLayers   int     `json:"n_layers"`
		MLPHidden int     `json:"mlp_hidden_size"`
		Vocab     int     `json:"vocab_size"`
		EmbSize   int     `json:"embedding_size"`
		MaxSeq    int     `json:"max_sequence_length"`
		Theta     float64 `json:"rope_theta"`
		Eps       float64 `json:"rms_norm_eps"`
		Tie       bool    `json:"weight_tying"`
		Block     string  `json:"block_type"`
		Norm      string  `json:"layer_norm_type"`
		QKVBias   bool    `json:"include_qkv_bias"`
		Bias      bool    `json:"include_bias"`
		Alibi     bool    `json:"alibi"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("llada config: %w", err)
	}
	if c.Block != "llama" || c.Norm != "rms" || c.Bias || c.QKVBias || c.Alibi {
		return nil, fmt.Errorf("llada: block_type %q / layer_norm_type %q / bias / alibi not supported", c.Block, c.Norm)
	}
	kv := c.NHeads
	if c.NKVHeads != nil && *c.NKVHeads > 0 {
		kv = *c.NKVHeads
	}
	V := c.EmbSize
	if V == 0 {
		V = c.Vocab
	}
	if c.Eps == 0 {
		c.Eps = 1e-5
	}
	cfg := LlamaConfig{VocabSize: V, HiddenSize: c.DModel, IntermediateSize: c.MLPHidden, NumLayers: c.NLayers,
		NumHeads: c.NHeads, NumKVHeads: kv, HeadDim: c.DModel / c.NHeads, MaxPosition: c.MaxSeq,
		RopeBase: c.Theta, RmsEps: c.Eps, TieEmbed: c.Tie, WeightDType: weightDT, SkipInit: true}
	src, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, fmt.Errorf("LoadLLaDA: %w", err)
	}
	hf := map[string]*tensor.Tensor{}
	ren := map[string]string{"attn_norm": "input_layernorm", "q_proj": "self_attn.q_proj", "k_proj": "self_attn.k_proj",
		"v_proj": "self_attn.v_proj", "attn_out": "self_attn.o_proj", "ff_norm": "post_attention_layernorm",
		"ff_proj": "mlp.gate_proj", "up_proj": "mlp.up_proj", "ff_out": "mlp.down_proj"}
	for k, t := range src {
		n := strings.TrimPrefix(k, "model.transformer.")
		switch {
		case n == "wte.weight":
			n = "model.embed_tokens.weight"
		case n == "ln_f.weight":
			n = "model.norm.weight"
		case n == "ff_out.weight":
			n = "lm_head.weight"
		case strings.HasPrefix(n, "blocks."):
			parts := strings.SplitN(n, ".", 4) // blocks, N, name, weight
			if len(parts) != 4 || ren[parts[2]] == "" {
				return nil, fmt.Errorf("LoadLLaDA: unexpected tensor %s", k)
			}
			n = "model.layers." + parts[1] + "." + ren[parts[2]] + "." + parts[3]
		default:
			return nil, fmt.Errorf("LoadLLaDA: unexpected tensor %s", k)
		}
		hf[n] = t
	}
	m, err := NewLlama(cfg, dev)
	if err != nil {
		return nil, err
	}
	if err := mapHFToLlama(m, hf); err != nil {
		return nil, err
	}
	return m, nil
}

// MaskTokenID reads mask_token_id from a repo's config.json.
func MaskTokenID(repo string) (int64, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return 0, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	var c struct {
		Mask *int64 `json:"mask_token_id"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return 0, err
	}
	if c.Mask == nil {
		return 0, fmt.Errorf("%s: config.json has no mask_token_id", repo)
	}
	return *c.Mask, nil
}

// ForwardBidirectional returns the logits (B, T, V) of tokens (B, T) with
// every position attending to every other (no cache).
func (m *Llama) ForwardBidirectional(tokens *tensor.Tensor) (*tensor.Tensor, error) {
	return m.ForwardDiffusion(tokens, nil, tokens.Dim(1), false)
}

// MaskedDiffusionOpts controls LLaDAGenerate and DreamGenerate.
type MaskedDiffusionOpts struct {
	MaskID    int64
	GenLength int // tokens to generate (the mask canvas)
	Steps     int // denoising steps in all
	// LLaDA: BlockLength splits the canvas into semi-autoregressive blocks
	// (Steps divided among them); CFGScale > 0 adds unsupervised
	// classifier-free guidance (the prompt masked out); Remasking
	// "low_confidence" (default) or "random"; LogitsEOSInf bans EOSID.
	BlockLength  int
	CFGScale     float64
	Remasking    string
	LogitsEOSInf bool
	EOSID        int64
	// Dream: Alg "maskgit_plus", "topk_margin", "entropy" (default) or
	// "origin"; Eps is the final timestep (default 1e-3).
	Alg string
	Eps float64
	// Temperature 0 is greedy; otherwise sampling (TopP / TopK for Dream)
	// with Rng.
	Temperature float64
	TopP        float64
	TopK        int
	Rng         *rand.Rand
	// OnStep, when set, sees the canvas (generated part) after each step.
	OnStep func(step int, canvas []int64)
}

func (o *MaskedDiffusionOpts) rng() *rand.Rand {
	if o.Rng == nil {
		o.Rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}
	return o.Rng
}

// bidiLogits runs the canvas rows (B, T) and returns host logits per row.
func (m *Llama) bidiLogits(rows [][]int64) ([][]float32, error) {
	B, T := len(rows), len(rows[0])
	flat := make([]int64, 0, B*T)
	for _, r := range rows {
		flat = append(flat, r...)
	}
	x, err := tensor.New(flat, []int{B, T}, m.TokenEmb.Weight.Device())
	if err != nil {
		return nil, err
	}
	out, err := m.ForwardBidirectional(x)
	if err != nil {
		return nil, err
	}
	if out, err = out.To(device.CPU()); err == nil && out.DType() != dtype.F32 {
		out, err = ops.Cast(out, dtype.F32)
	}
	if err == nil {
		out, err = out.Contiguous()
	}
	if err != nil {
		return nil, err
	}
	all := tensor.MustSlice[float32](out)
	V := out.Dim(-1)
	res := make([][]float32, B)
	for b := range res {
		res[b] = all[b*T*V : (b+1)*T*V]
	}
	return res, nil
}

// LLaDAGenerate is ML-GSAI/LLaDA's generate(): low-confidence (or random)
// remasking over semi-autoregressive blocks, with optional CFG. Returns the
// GenLength generated tokens.
func LLaDAGenerate(m *Llama, prompt []int64, o MaskedDiffusionOpts) ([]int64, error) {
	L, G := len(prompt), o.GenLength
	if o.BlockLength <= 0 {
		o.BlockLength = G
	}
	if G <= 0 || G%o.BlockLength != 0 {
		return nil, fmt.Errorf("llada: gen length %d must be a positive multiple of block length %d", G, o.BlockLength)
	}
	nb := G / o.BlockLength
	if o.Steps <= 0 || o.Steps%nb != 0 {
		return nil, fmt.Errorf("llada: steps %d must be a positive multiple of the %d blocks", o.Steps, nb)
	}
	steps := o.Steps / nb
	x := append(append([]int64(nil), prompt...), make([]int64, G)...)
	for i := L; i < L+G; i++ {
		x[i] = o.MaskID
	}
	for b := range nb {
		start, end := L+b*o.BlockLength, L+(b+1)*o.BlockLength
		masked := 0
		for i := start; i < end; i++ {
			if x[i] == o.MaskID {
				masked++
			}
		}
		for s := range steps {
			// get_num_transfer_tokens: base, the first `remainder` steps one more.
			k := masked / steps
			if s < masked%steps {
				k++
			}
			rows := [][]int64{x}
			if o.CFGScale > 0 {
				un := append([]int64(nil), x...)
				for i := range L {
					un[i] = o.MaskID
				}
				rows = append(rows, un)
			}
			lg, err := m.bidiLogits(rows)
			if err != nil {
				return nil, err
			}
			logits := lg[0]
			V := len(logits) / len(x)
			if o.CFGScale > 0 { // un + (cfg + 1)·(cond − un)
				mix := make([]float32, len(logits))
				for i := range mix {
					mix[i] = lg[1][i] + float32(o.CFGScale+1)*(logits[i]-lg[1][i])
				}
				logits = mix
			}
			type cand struct {
				pos  int
				tok  int64
				conf float64
			}
			var cands []cand
			for i := L; i < L+G; i++ {
				if x[i] != o.MaskID {
					continue
				}
				row := logits[i*V : (i+1)*V]
				if o.LogitsEOSInf {
					row = append([]float32(nil), row...)
					row[o.EOSID] = float32(math.Inf(-1))
				}
				tok := argmaxGumbel(row, o.Temperature, &o)
				conf := math.Inf(-1)
				if i < end {
					switch o.Remasking {
					case "", "low_confidence":
						conf = softmaxAt(row, tok)
					case "random":
						conf = o.rng().Float64()
					default:
						return nil, fmt.Errorf("llada: remasking %q", o.Remasking)
					}
				}
				cands = append(cands, cand{i, int64(tok), conf})
			}
			sort.SliceStable(cands, func(a, b int) bool { return cands[a].conf > cands[b].conf })
			for _, c := range cands[:min(k, len(cands))] {
				x[c.pos] = c.tok
			}
			if o.OnStep != nil {
				o.OnStep(b*steps+s, x[L:])
			}
		}
	}
	return x[L:], nil
}

// argmaxGumbel is LLaDA's add_gumbel_noise + argmax: exp(logits) /
// (−log u)^temperature in F64, greedy at temperature 0.
func argmaxGumbel(row []float32, temp float64, o *MaskedDiffusionOpts) int {
	best, bv := 0, math.Inf(-1)
	for i, l := range row {
		v := float64(l)
		if temp > 0 {
			u := o.rng().Float64()
			v = math.Exp(v) / math.Pow(-math.Log(u), temp)
		}
		if v > bv {
			best, bv = i, v
		}
	}
	return best
}

// softmaxAt is softmax(row)[i].
func softmaxAt(row []float32, i int) float64 {
	mx := float32(math.Inf(-1))
	for _, v := range row {
		mx = max(mx, v)
	}
	var s float64
	for _, v := range row {
		s += math.Exp(float64(v - mx))
	}
	return math.Exp(float64(row[i]-mx)) / s
}

// DreamGenerate is Dream's diffusion_generate (_sample): a canvas of
// GenLength masks after the prompt, Steps denoising steps on the linear
// schedule t = 1 → Eps, committing the most confident masked positions
// (maskgit_plus: top probability; topk_margin: top-1 − top-2; entropy:
// negative entropy) or, for "origin", each masked position with
// probability 1 − s/t.
func DreamGenerate(m *Llama, prompt []int64, o MaskedDiffusionOpts) ([]int64, error) {
	L, G := len(prompt), o.GenLength
	if G <= 0 || o.Steps <= 0 {
		return nil, fmt.Errorf("dream: GenLength and Steps must be > 0")
	}
	if o.Eps == 0 {
		o.Eps = 1e-3
	}
	if o.Alg == "" {
		o.Alg = "entropy"
	}
	x := append(append([]int64(nil), prompt...), make([]int64, G)...)
	for i := L; i < L+G; i++ {
		x[i] = o.MaskID
	}
	ts := make([]float32, o.Steps+1) // torch.linspace(1, eps, steps+1) in F32
	for i := range ts {
		ts[i] = float32(1 + (o.Eps-1)*float64(i)/float64(o.Steps))
	}
	ts[o.Steps] = float32(o.Eps)
	for s := range o.Steps {
		lg, err := m.bidiLogits([][]int64{x})
		if err != nil {
			return nil, err
		}
		logits := lg[0]
		V := len(logits) / len(x)
		row := func(i int) []float32 { // shifted: position i reads i−1 (0 reads 0)
			j := max(i-1, 0)
			return logits[j*V : (j+1)*V]
		}
		t, sNext := ts[s], ts[s+1]
		var masked []int
		for i := range x {
			if x[i] == o.MaskID {
				masked = append(masked, i)
			}
		}
		if o.Alg == "origin" {
			p := 1.0
			if s < o.Steps-1 {
				p = float64(1 - sNext/t)
			}
			for _, i := range masked {
				if o.rng().Float64() < p {
					_, tok := dreamSample(row(i), &o, "")
					x[i] = int64(tok)
				}
			}
		} else {
			type cand struct {
				pos  int
				tok  int
				conf float64
			}
			cands := make([]cand, len(masked))
			for ci, i := range masked {
				conf, tok := dreamSample(row(i), &o, o.Alg)
				cands[ci] = cand{i, tok, conf}
			}
			n := len(masked)
			if s < o.Steps-1 {
				n = int(float32(len(masked)) * (1 - sNext/t))
			}
			sort.SliceStable(cands, func(a, b int) bool { return cands[a].conf > cands[b].conf })
			for _, c := range cands[:min(n, len(cands))] {
				x[c.pos] = int64(c.tok)
			}
		}
		if o.OnStep != nil {
			o.OnStep(s, x[L:])
		}
	}
	return x[L:], nil
}

// dreamSample is Dream's sample_tokens for one position: temperature,
// top-p, top-k, softmax; the token (sampled, or the argmax at temperature
// 0) and its confidence by alg.
func dreamSample(row []float32, o *MaskedDiffusionOpts, alg string) (float64, int) {
	V := len(row)
	l := make([]float64, V)
	for i, v := range row {
		l[i] = float64(v)
		if o.Temperature > 0 {
			l[i] /= o.Temperature
		}
	}
	idx := make([]int, V)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return l[idx[a]] > l[idx[b]] })
	if o.TopP > 0 && o.TopP < 1 { // keep the smallest prefix reaching TopP (the first always)
		probs := softmax64(l)
		cum := 0.0
		for r, i := range idx {
			if r > 0 && cum > o.TopP {
				l[i] = math.Inf(-1)
			}
			cum += probs[i]
		}
	}
	if o.TopK > 0 && o.TopK < V {
		kth := l[idx[o.TopK-1]]
		for i := range l {
			if l[i] < kth {
				l[i] = math.Inf(-1)
			}
		}
	}
	p := softmax64(l)
	tok := idx[0]
	if o.Temperature > 0 {
		u, acc := o.rng().Float64(), 0.0
		for i, pi := range p {
			acc += pi
			if u < acc {
				tok = i
				break
			}
		}
	}
	conf := p[tok]
	switch alg {
	case "topk_margin":
		a, b := 0.0, 0.0
		for _, pi := range p {
			if pi > a {
				a, b = pi, a
			} else if pi > b {
				b = pi
			}
		}
		conf = a - b
	case "entropy":
		conf = 0
		for _, pi := range p {
			conf += pi * math.Log(pi+1e-10)
		}
	}
	return conf, tok
}

func softmax64(l []float64) []float64 {
	mx := math.Inf(-1)
	for _, v := range l {
		mx = max(mx, v)
	}
	p := make([]float64, len(l))
	var s float64
	for i, v := range l {
		p[i] = math.Exp(v - mx)
		s += p[i]
	}
	for i := range p {
		p[i] /= s
	}
	return p
}
