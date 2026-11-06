package models

import (
	"math"
	"math/rand/v2"
	"sort"
)

// SampleOpts controls how a token is drawn from a row of logits.
//
// Combine knobs as follows: temperature is applied first (logits ← logits/τ);
// then top-k truncation, then top-p (nucleus). With Temperature == 0 the
// other knobs are ignored and the argmax is returned (greedy decoding).
type SampleOpts struct {
	Temperature float32 // 0 → greedy (argmax). Otherwise τ in the softmax.
	TopK        int     // > 0 → keep only the K highest logits before sampling.
	TopP        float32 // 0 or ≥1 → no cap. Otherwise smallest set whose probability mass ≥ TopP.
	Rng         *rand.Rand
}

// SampleLogits picks one token id from a length-V logits slice using the
// given options. It does not mutate `logits`.
func SampleLogits(logits []float32, opts SampleOpts) int {
	V := len(logits)
	if V == 0 {
		return -1
	}
	if opts.Temperature == 0 {
		return argmax(logits)
	}
	rng := opts.Rng
	if rng == nil {
		rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}

	// Stable softmax with temperature: subtract max for numerical stability.
	maxL := float32(math.Inf(-1))
	for _, x := range logits {
		if x > maxL {
			maxL = x
		}
	}
	invT := 1.0 / opts.Temperature
	probs := make([]float32, V)
	idx := make([]int, V)
	var sum float32
	for i, x := range logits {
		p := float32(math.Exp(float64((x - maxL) * invT)))
		probs[i] = p
		idx[i] = i
		sum += p
	}
	if sum == 0 || math.IsNaN(float64(sum)) {
		return argmax(logits)
	}
	for i := range probs {
		probs[i] /= sum
	}

	// top-k: keep the K largest probs. We sort indices descending by prob.
	if opts.TopK > 0 && opts.TopK < V {
		sort.Slice(idx, func(a, b int) bool { return probs[idx[a]] > probs[idx[b]] })
		idx = idx[:opts.TopK]
	} else {
		sort.Slice(idx, func(a, b int) bool { return probs[idx[a]] > probs[idx[b]] })
	}

	// top-p: keep the smallest prefix whose cumulative prob ≥ TopP.
	if opts.TopP > 0 && opts.TopP < 1 {
		var cum float32
		cut := len(idx)
		for i, id := range idx {
			cum += probs[id]
			if cum >= opts.TopP {
				cut = i + 1
				break
			}
		}
		idx = idx[:cut]
	}

	// Re-normalise the survivor set and sample.
	var renorm float32
	for _, id := range idx {
		renorm += probs[id]
	}
	if renorm == 0 {
		return idx[0]
	}
	r := rng.Float32() * renorm
	var acc float32
	for _, id := range idx {
		acc += probs[id]
		if r <= acc {
			return id
		}
	}
	return idx[len(idx)-1]
}

// argmax returns the index of the largest entry. Ties resolve to the first.
func argmax(xs []float32) int {
	best := 0
	for i := 1; i < len(xs); i++ {
		if xs[i] > xs[best] {
			best = i
		}
	}
	return best
}
