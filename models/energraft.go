package models

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/safetensors"
	"light/tensor"
)

// EnergraftConfig is energy_config.json of an Energraft-EBT release.
type EnergraftConfig struct {
	EnergySteps   int     `json:"energy_steps"`
	Relaxation    float64 `json:"refinement_relaxation"`
	Init          string  `json:"candidate_initialization"`
	InitialStd    float64 `json:"candidate_initial_std"`
	StepSize      float64 `json:"candidate_step_size"`
	HeadWidth     int     `json:"energy_head_width"`
	BaseModel     string  `json:"base_model"`
	ReleaseFormat string  `json:"format"`
}

// Energraft is an Energraft-EBT energy-based language model
// (dimweak/Energraft-EBT-Qwen3-1.7B-Base): a Qwen3 backbone with tied
// embeddings plus a scalar energy head. The next token is not read off an
// LM head: a candidate distribution (Gaussian logits) is embedded as
// softmax(state)·E, run as one extra token through every layer - attending
// to the prompt's cached keys / values and to itself, at the next position
// - and the energy head scores the final normed state; the candidate moves
// against the energy's gradient w.r.t. its embedding, mapped back through
// the tied head (state ← p − step·E·∇), EnergySteps times, the later steps
// relaxed toward the proposal. Greedy decoding takes the final state's
// argmax. The gradient runs through Light's autograd with the backbone's
// linears frozen (input gradients only).
type Energraft struct {
	Cfg      EnergraftConfig
	Backbone *Llama
	// Energy head: f·baseline + last·silu(first·(f − center)).
	Baseline, Center *tensor.Tensor // (D,)
	First, Last      *nn.Linear
	dev              device.Device
}

// LoadEnergraft loads a release folder (or HF repo): the Qwen3 backbone in
// F32, energy_head.safetensors and energy_config.json.
func LoadEnergraft(repo string, dev device.Device) (*Energraft, error) {
	dir, err := snapshotDir(repo)
	if err != nil {
		return nil, err
	}
	m := &Energraft{dev: dev, Cfg: EnergraftConfig{EnergySteps: 1, Relaxation: 1, Init: "gaussian", InitialStd: 1, StepSize: 1}}
	raw, err := os.ReadFile(filepath.Join(dir, "energy_config.json"))
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(raw, &m.Cfg); err != nil {
		return nil, fmt.Errorf("energy_config.json: %w", err)
	}
	if m.Cfg.Init != "gaussian" {
		return nil, fmt.Errorf("energraft: candidate_initialization %q (only gaussian is ported)", m.Cfg.Init)
	}
	if m.Backbone, err = LoadCausalLMFromHF(dir, dev, dtype.F32); err != nil {
		return nil, err
	}
	bc := m.Backbone.Cfg
	if !bc.TieEmbed || bc.SlidingWindow > 0 {
		return nil, fmt.Errorf("energraft: the backbone must have tied embeddings and full attention")
	}
	for i, b := range m.Backbone.Blocks {
		if b.GateProj == nil || b.QNorm == nil || b.KNorm == nil || b.N1 == nil || b.N2 == nil {
			return nil, fmt.Errorf("energraft: layer %d is not a dense Qwen3 layer", i)
		}
	}
	head, _, err := safetensors.LoadFiles([]string{filepath.Join(dir, "energy_head.safetensors")}, device.CPU())
	if err != nil {
		return nil, err
	}
	get := func(n string) (*tensor.Tensor, error) {
		t, ok := head[n]
		if !ok {
			return nil, fmt.Errorf("energy_head.safetensors: missing %s", n)
		}
		if t, err = ops.Cast(t, dtype.F32); err != nil {
			return nil, err
		}
		return t.To(dev)
	}
	lin := func(p string, bias bool) (*nn.Linear, error) {
		w, err := get(p + ".weight")
		if err != nil {
			return nil, err
		}
		l, err := nn.NewLinearPlaceholder(w.Dim(1), w.Dim(0), false, dev)
		if err == nil {
			err = l.SetWeight(w)
		}
		if err != nil {
			return nil, err
		}
		if bias {
			if l.B, err = get(p + ".bias"); err != nil {
				return nil, err
			}
		}
		return l, nil
	}
	if m.Baseline, err = get("baseline"); err != nil {
		return nil, err
	}
	if m.Center, err = get("center"); err != nil {
		return nil, err
	}
	if m.First, err = lin("first", true); err != nil {
		return nil, err
	}
	if m.Last, err = lin("last", false); err != nil {
		return nil, err
	}
	return m, nil
}

// frozenLinear is y = x·Wᵀ (+ b) with W, b constants: backward returns only
// the input gradient g·W (no transposed copy, no weight gradient).
type frozenLinear struct{ l *nn.Linear }

func (frozenLinear) Name() string { return "FrozenLinear" }

func (f frozenLinear) Forward(_ *autograd.Context, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	y, err := f.l.Forward(in[0])
	return []*tensor.Tensor{y}, err
}

func (f frozenLinear) Backward(_ *autograd.Context, g []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if g[0] == nil {
		return []*tensor.Tensor{nil}, nil
	}
	if f.l.W == nil || f.l.W.DType() != g[0].DType() {
		return nil, fmt.Errorf("energraft: frozen linear needs a %v weight, have %v", g[0].DType(), f.l.W.DType())
	}
	gx, err := ops.MatMul(g[0], f.l.W)
	return []*tensor.Tensor{gx}, err
}

// agOps chains autograd calls, keeping the first error.
type agOps struct{ err error }

func (a *agOps) do(f func() (*tensor.Tensor, error)) *tensor.Tensor {
	if a.err != nil {
		return nil
	}
	t, err := f()
	if err != nil {
		a.err = err
	}
	return t
}

func (a *agOps) lin(l *nn.Linear, x *tensor.Tensor) *tensor.Tensor {
	return a.do(func() (*tensor.Tensor, error) {
		y, err := autograd.Apply(frozenLinear{l}, []*tensor.Tensor{x})
		if err != nil {
			return nil, err
		}
		return y[0], nil
	})
}

func (a *agOps) rms(x *tensor.Tensor, n *nn.RMSNorm) *tensor.Tensor {
	return a.do(func() (*tensor.Tensor, error) { return autograd.RMSNorm(x, n.Weight, float32(n.Eps)) })
}

func (a *agOps) add(x, y *tensor.Tensor) *tensor.Tensor {
	return a.do(func() (*tensor.Tensor, error) { return autograd.Add(x, y) })
}

func (a *agOps) mul(x, y *tensor.Tensor) *tensor.Tensor {
	return a.do(func() (*tensor.Tensor, error) { return autograd.Mul(x, y) })
}

func (a *agOps) reshape(x *tensor.Tensor, d ...int) *tensor.Tensor {
	return a.do(func() (*tensor.Tensor, error) { return autograd.Reshape(x, d...) })
}

func (a *agOps) narrow(x *tensor.Tensor, dim, s, n int) *tensor.Tensor {
	return a.do(func() (*tensor.Tensor, error) { return autograd.Narrow(x, dim, s, n) })
}

func (a *agOps) cat(axis int, xs ...*tensor.Tensor) *tensor.Tensor {
	return a.do(func() (*tensor.Tensor, error) { return autograd.Concat(xs, axis) })
}

// energraftContext is the observed prompt: per layer the cached keys /
// values as (KH, T, HD) and the candidate's RoPE at position T.
type energraftContext struct {
	k, kT, v []*tensor.Tensor
	cos, sin *tensor.Tensor // (1, HD)
	scale    *tensor.Tensor // (1,) = 1/√HD
}

func (m *Energraft) context(c *KVCache) (*energraftContext, error) {
	cfg := m.Backbone.Cfg
	KH, HD, T := cfg.NumKVHeads, cfg.HeadDim, c.Len
	ctx := &energraftContext{}
	for l := range m.Backbone.Blocks {
		if c.K[l] == nil || c.K[l].Dim(0) != 1 || c.K[l].Dim(2) != T {
			return nil, fmt.Errorf("energraft: layer %d cache is not a full (1, KH, %d, HD) prefix", l, T)
		}
		k, err := narrowContig(c.K[l], 2, 0, T)
		if err == nil {
			k, err = k.Reshape(KH, T, HD)
		}
		var v, kT *tensor.Tensor
		if err == nil {
			v, err = narrowContig(c.V[l], 2, 0, T)
		}
		if err == nil {
			v, err = v.Reshape(KH, T, HD)
		}
		if err == nil {
			kT, err = k.Transpose(1, 2)
		}
		if err == nil {
			kT, err = kT.Contiguous()
		}
		if err != nil {
			return nil, err
		}
		ctx.k, ctx.kT, ctx.v = append(ctx.k, k), append(ctx.kT, kT), append(ctx.v, v)
	}
	// rotate_half RoPE at the candidate's position T (F32 like transformers).
	half := HD / 2
	cs, sn := make([]float32, HD), make([]float32, HD)
	for i := range half {
		inv := float32(1 / math.Pow(cfg.RopeBase, float64(2*i)/float64(HD)))
		a := float64(float32(T) * inv)
		cs[i], cs[i+half] = float32(math.Cos(a)), float32(math.Cos(a))
		sn[i], sn[i+half] = float32(math.Sin(a)), float32(math.Sin(a))
	}
	var err error
	if ctx.scale, err = tensor.New([]float32{float32(1 / math.Sqrt(float64(HD)))}, []int{1}, m.dev); err != nil {
		return nil, err
	}
	if ctx.cos, err = tensor.New(cs, []int{1, HD}, m.dev); err != nil {
		return nil, err
	}
	ctx.sin, err = tensor.New(sn, []int{1, HD}, m.dev)
	return ctx, err
}

// rope rotates x (H, HD) by the context's position (rotate_half).
func (a *agOps) rope(x *tensor.Tensor, ctx *energraftContext) *tensor.Tensor {
	if a.err != nil {
		return nil
	}
	hd := x.Dim(1)
	x1, x2 := a.narrow(x, 1, 0, hd/2), a.narrow(x, 1, hd/2, hd/2)
	neg := a.do(func() (*tensor.Tensor, error) { return autograd.Neg(x2) })
	rot := a.cat(1, neg, x1)
	return a.add(a.mul(x, ctx.cos), a.mul(rot, ctx.sin))
}

// energy is the head's scalar for the candidate embedding emb (1, D): the
// candidate stream through every layer, the final norm, then the residual
// head.
func (m *Energraft) energy(a *agOps, emb *tensor.Tensor, ctx *energraftContext) *tensor.Tensor {
	cfg := m.Backbone.Cfg
	NH, KH, HD := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	G := NH / KH
	h := emb
	for l, b := range m.Backbone.Blocks {
		n := a.rms(h, b.N1)
		q := a.rope(a.rms(a.reshape(a.lin(b.QProj, n), NH, HD), b.QNorm), ctx)
		k := a.rope(a.rms(a.reshape(a.lin(b.KProj, n), KH, HD), b.KNorm), ctx)
		v := a.reshape(a.lin(b.VProj, n), KH, 1, HD)
		qg := a.reshape(q, KH, G, HD)
		sc := a.do(func() (*tensor.Tensor, error) { return autograd.MatMul(qg, ctx.kT[l]) }) // (KH, G, T)
		qk := a.mul(qg, a.reshape(k, KH, 1, HD))
		self := a.do(func() (*tensor.Tensor, error) { return autograd.AxisSum(qk, []int{2}, true) }) // (KH, G, 1)
		all := a.mul(a.cat(2, sc, self), ctx.scale)
		p := a.do(func() (*tensor.Tensor, error) { return autograd.Softmax(all, 2) })
		T := ctx.v[l].Dim(1)
		pc, ps := a.narrow(p, 2, 0, T), a.narrow(p, 2, T, 1)
		att := a.add(a.do(func() (*tensor.Tensor, error) { return autograd.MatMul(pc, ctx.v[l]) }), a.mul(ps, v))
		h = a.add(h, a.lin(b.OProj, a.reshape(att, 1, NH*HD)))
		n2 := a.rms(h, b.N2)
		gl := a.lin(b.GateProj, n2)
		gate := a.do(func() (*tensor.Tensor, error) { return autograd.SiLU(gl) })
		h = a.add(h, a.lin(b.DownProj, a.mul(gate, a.lin(b.UpProj, n2))))
	}
	f := a.rms(h, m.Backbone.FinalNorm)
	fb := a.mul(f, m.Baseline)
	orig := a.do(func() (*tensor.Tensor, error) { return autograd.Sum(fb) })
	centered := a.do(func() (*tensor.Tensor, error) { return autograd.Sub(f, m.Center) })
	fl := a.lin(m.First, centered)
	hid := a.do(func() (*tensor.Tensor, error) { return autograd.SiLU(fl) })
	ll := a.lin(m.Last, hid)
	res := a.do(func() (*tensor.Tensor, error) { return autograd.Sum(ll) })
	return a.add(orig, res)
}

// Refine runs the energy updates (steps ≤ 0: the release's energy_steps)
// from the initial candidate state (1, V) against the cached prompt c and
// returns the state after each update.
func (m *Energraft) Refine(c *KVCache, state *tensor.Tensor, steps int) ([]*tensor.Tensor, error) {
	if steps <= 0 {
		steps = m.Cfg.EnergySteps
	}
	if steps < 1 {
		return nil, fmt.Errorf("energraft: at least one energy update")
	}
	ctx, err := m.context(c)
	if err != nil {
		return nil, err
	}
	E := m.Backbone.TokenEmb.Weight
	var out []*tensor.Tensor
	for i := range steps {
		p, err := ops.Softmax(state, -1)
		if err != nil {
			return nil, err
		}
		emb, err := ops.MatMul(p, E) // (1, D)
		if err != nil {
			return nil, err
		}
		emb.SetRequiresGrad(true)
		a := &agOps{}
		e := m.energy(a, emb, ctx)
		if a.err != nil {
			return nil, a.err
		}
		if err := autograd.Backward(e, nil); err != nil {
			return nil, err
		}
		g := emb.Grad()
		if g == nil {
			return nil, fmt.Errorf("energraft: no gradient reached the candidate embedding")
		}
		dl, err := m.Backbone.LMHead.Forward(g) // E·∇, (1, V)
		if err == nil && m.Cfg.StepSize != 1 {
			dl, err = ops.Affine(dl, m.Cfg.StepSize, 0)
		}
		var proposal *tensor.Tensor
		if err == nil {
			proposal, err = ops.Sub(p, dl)
		}
		if err != nil {
			return nil, err
		}
		switch beta := m.Cfg.Relaxation; {
		case i == 0 || beta == 1:
			state = proposal
		case beta == 0:
		default: // state + β·(proposal − state)
			d, err := ops.Sub(proposal, state)
			if err == nil {
				d, err = ops.Affine(d, beta, 0)
			}
			if err == nil {
				state, err = ops.Add(state, d)
			}
			if err != nil {
				return nil, err
			}
		}
		out = append(out, state)
	}
	return out, nil
}

// InitialState draws the Gaussian candidate logits (1, V).
func (m *Energraft) InitialState(rng *rand.Rand) (*tensor.Tensor, error) {
	v := make([]float32, m.Backbone.Cfg.VocabSize)
	for i := range v {
		v[i] = float32(rng.NormFloat64() * m.Cfg.InitialStd)
	}
	return tensor.New(v, []int{1, len(v)}, m.dev)
}

// EnergraftOpts controls Generate. Init, when set, supplies each token's
// initial candidate state instead of drawing it from Rng (for parity with
// a reference); Steps 0 uses the release's energy_steps.
type EnergraftOpts struct {
	MaxTokens int
	Steps     int
	Rng       *rand.Rand
	EOS       map[int64]bool
	Init      func(step int) (*tensor.Tensor, error)
	OnToken   func(id int64) bool // return false to stop
}

// Generate greedily decodes after prompt: prefill into a KV cache, then per
// token refine a candidate and take its argmax, feeding it back through the
// backbone.
func (m *Energraft) Generate(prompt []int64, o EnergraftOpts) ([]int64, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("energraft: empty prompt")
	}
	steps := o.Steps
	if steps <= 0 {
		steps = m.Cfg.EnergySteps
	}
	if o.Rng == nil {
		o.Rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}
	c := NewKVCache(len(m.Backbone.Blocks))
	defer c.Reset()
	feed := func(ids []int64) error {
		x, err := tensor.New(ids, []int{1, len(ids)}, m.dev)
		if err != nil {
			return err
		}
		_, err = m.Backbone.ForwardWithCache(x, c)
		return err
	}
	if err := feed(prompt); err != nil {
		return nil, err
	}
	var out []int64
	for t := range o.MaxTokens {
		var init *tensor.Tensor
		var err error
		if o.Init != nil {
			init, err = o.Init(t)
		} else {
			init, err = m.InitialState(o.Rng)
		}
		if err != nil {
			return out, err
		}
		states, err := m.Refine(c, init, steps)
		if err != nil {
			return out, err
		}
		h, err := states[len(states)-1].To(device.CPU())
		if err != nil {
			return out, err
		}
		row := tensor.MustSlice[float32](h)
		best := 0
		for i, v := range row {
			if v > row[best] {
				best = i
			}
		}
		id := int64(best)
		out = append(out, id)
		if o.OnToken != nil && !o.OnToken(id) {
			break
		}
		if o.EOS[id] {
			break
		}
		if err := feed([]int64{id}); err != nil {
			return out, err
		}
	}
	return out, nil
}
