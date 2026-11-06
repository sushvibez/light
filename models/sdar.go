package models

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"

	"light/device"
	"light/tensor"
)

// SDAR (Synergistic Diffusion-AutoRegression, JetLM/SDAR-*) is a Qwen3
// body decoded block by block with masked diffusion: a block of
// `BlockLength` positions starts as mask tokens, attention is bidirectional
// inside the block and causal across blocks, and each denoising pass
// predicts every masked position at once, committing the most confident
// ones. The KV cache only stores a block once it is fully decoded. This
// file is the port of the reference generate.py loop; the mask and the
// "attend without storing" pass live in Llama.ForwardDiffusion.

// SDAROpts configures GenerateDiffusion.
type SDAROpts struct {
	BlockLength    int     // tokens per block (model-specific; SDAR-*-Chat is 4, *-b8 is 8, …)
	DenoisingSteps int     // passes per block; each commits ≈ BlockLength/DenoisingSteps tokens
	Threshold      float32 // > 0: low_confidence_dynamic — commit every masked token above it (at least the static count)
	MaskID         int64   // <|MASK|> (151669 for SDAR)
	EOS            []int64 // stop ids; checked at block granularity
	MaxTokens      int
	Temperature    float32 // 0 = greedy; otherwise sample with TopK / TopP (SamplingSeed)
	TopK           int
	TopP           float32
	Seed           uint64
}

// DefaultSDAROpts mirrors SDAR's evaluation setting (block 4, 4 steps, greedy).
func DefaultSDAROpts() SDAROpts {
	return SDAROpts{BlockLength: 4, DenoisingSteps: 4, MaskID: 151669, EOS: []int64{151645, 151643}, MaxTokens: 256}
}

// ForwardDiffusion runs tokens (B, T) through the model with the
// block-bidirectional mask of `blockLen` absolute positions. With store
// the block's keys/values are appended to the cache (and cache.Len
// advanced); without it the pass attends over the cache but leaves it
// unchanged, which is what every denoising iteration needs.
func (m *Llama) ForwardDiffusion(tokens *tensor.Tensor, cache *KVCache, blockLen int, store bool) (*tensor.Tensor, error) {
	if blockLen <= 0 {
		return nil, fmt.Errorf("ForwardDiffusion: blockLen must be > 0")
	}
	m.diffBlock, m.diffStore = blockLen, store
	defer func() { m.diffBlock, m.diffStore = 0, false }()
	return m.ForwardWithCache(tokens, cache)
}

// GenerateDiffusion decodes a continuation of prompt with SDAR's
// block-diffusion loop and returns the generated ids (without the prompt).
// onBlock, when set, receives each finished block's new tokens.
func (m *Llama) GenerateDiffusion(prompt []int64, opts SDAROpts, onBlock func([]int64)) ([]int64, error) {
	B, steps := opts.BlockLength, opts.DenoisingSteps
	if B <= 0 || steps <= 0 || opts.MaxTokens <= 0 {
		return nil, fmt.Errorf("GenerateDiffusion: BlockLength, DenoisingSteps and MaxTokens must be > 0")
	}
	dev := m.TokenEmb.Weight.Device()
	cache := NewKVCache(m.NumLayers())
	rng := rand.New(rand.NewPCG(opts.Seed, opts.Seed^0x9e3779b97f4a7c15))
	fwd := func(ids []int64, store bool) ([]float32, error) {
		x, err := tensor.New(ids, []int{1, len(ids)}, dev)
		if err != nil {
			return nil, err
		}
		logits, err := m.ForwardDiffusion(x, cache, B, store)
		if err != nil {
			return nil, err
		}
		if !store {
			cpu, err := logits.To(device.CPU())
			if err != nil {
				return nil, err
			}
			return tensor.MustSlice[float32](cpu), nil
		}
		return nil, nil
	}
	// Prefill the complete prompt blocks; a partial last block joins block 0.
	prefill := len(prompt) / B * B
	if prefill > 0 {
		if _, err := fwd(prompt[:prefill], true); err != nil {
			return nil, err
		}
	}
	rem := append([]int64(nil), prompt[prefill:]...)
	var gen []int64
	for len(gen) < opts.MaxTokens {
		cur := make([]int64, B)
		copy(cur, rem)
		for i := len(rem); i < B; i++ {
			cur[i] = opts.MaskID
		}
		nmask := B - len(rem)
		sched := transferSchedule(nmask, steps)
		for step := 0; step <= steps; step++ {
			var masked []int
			for i, t := range cur {
				if t == opts.MaskID {
					masked = append(masked, i)
				}
			}
			if len(masked) == 0 {
				if _, err := fwd(cur, true); err != nil {
					return nil, err
				}
				break
			}
			logits, err := fwd(cur, false)
			if err != nil {
				return nil, err
			}
			V := len(logits) / B
			x0 := make([]int64, B)
			conf := make([]float32, B)
			for _, i := range masked {
				x0[i], conf[i] = sdarPick(logits[i*V:(i+1)*V], opts, rng)
			}
			k := min(sched[min(step, steps-1)], len(masked))
			sort.Slice(masked, func(a, b int) bool { return conf[masked[a]] > conf[masked[b]] })
			commit := masked[:k]
			if opts.Threshold > 0 {
				above := 0
				for _, i := range masked {
					if conf[i] >= opts.Threshold {
						above++
					}
				}
				if above > k {
					commit = masked[:above]
				}
			}
			for _, i := range commit {
				cur[i] = x0[i]
			}
		}
		newTok := cur[len(rem):]
		rem = rem[:0]
		stop := false
		for n, t := range newTok {
			for _, e := range opts.EOS {
				if t == e {
					newTok = newTok[:n]
					stop = true
				}
			}
			if stop {
				break
			}
		}
		gen = append(gen, newTok...)
		if onBlock != nil && len(newTok) > 0 {
			onBlock(newTok)
		}
		if stop {
			break
		}
	}
	if len(gen) > opts.MaxTokens {
		gen = gen[:opts.MaxTokens]
	}
	return gen, nil
}

// transferSchedule splits n masked tokens over `steps` passes:
// n/steps each, the first n%steps passes one more (generate.py's
// get_num_transfer_tokens).
func transferSchedule(n, steps int) []int {
	s := make([]int, steps)
	for i := range s {
		s[i] = n / steps
		if i < n%steps {
			s[i]++
		}
	}
	return s
}

// sdarPick returns the token chosen for one masked position and its
// confidence (the softmax probability of the chosen token). Greedy picks
// the argmax; sampling applies temperature, top-k and top-p and draws.
func sdarPick(row []float32, opts SDAROpts, rng *rand.Rand) (int64, float32) {
	best, mx := 0, float32(math.Inf(-1))
	for i, v := range row {
		if v > mx {
			best, mx = i, v
		}
	}
	if opts.Temperature <= 0 {
		var sum float64
		for _, v := range row {
			sum += math.Exp(float64(v - mx))
		}
		return int64(best), float32(1 / sum)
	}
	// Scaled softmax restricted to top-k / top-p.
	type cand struct {
		id int
		p  float64
	}
	var sum float64
	cands := make([]cand, len(row))
	for i, v := range row {
		p := math.Exp(float64((v - mx) / opts.Temperature))
		cands[i] = cand{i, p}
		sum += p
	}
	for i := range cands {
		cands[i].p /= sum
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].p > cands[b].p })
	if opts.TopK > 0 && opts.TopK < len(cands) {
		cands = cands[:opts.TopK]
	}
	if opts.TopP > 0 && opts.TopP < 1 {
		var acc float64
		for i := range cands {
			acc += cands[i].p
			if acc >= float64(opts.TopP) {
				cands = cands[:i+1]
				break
			}
		}
	}
	var tot float64
	for _, c := range cands {
		tot += c.p
	}
	r := rng.Float64() * tot
	for _, c := range cands {
		r -= c.p
		if r <= 0 {
			return int64(c.id), float32(c.p)
		}
	}
	c := cands[len(cands)-1]
	return int64(c.id), float32(c.p)
}
