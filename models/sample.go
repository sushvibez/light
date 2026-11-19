package models

import (
	"math"
	"math/rand/v2"
	"slices"
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
//
// Without top-k/top-p it draws by a linear cumulative walk in index order
// (the full softmax distribution, no sort). With truncation it selects the
// survivors in descending-probability order (ties: lower id first) — a
// size-K heap for top-k, a lazily popped max-heap for top-p alone — and
// walks only that prefix, so it never sorts the whole vocabulary.
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
	var sum float32
	for i, x := range logits {
		p := float32(math.Exp(float64((x - maxL) * invT)))
		probs[i] = p
		sum += p
	}
	if sum == 0 || math.IsNaN(float64(sum)) {
		return argmax(logits)
	}

	topK := opts.TopK > 0 && opts.TopK < V
	topP := opts.TopP > 0 && opts.TopP < 1
	if !topK && !topP {
		// Unrestricted: P(i) = probs[i]/sum, drawn in index order. r is in
		// [0, sum), and the strict r < acc never lands on a zero-mass token.
		r := rng.Float32() * sum
		var acc float32
		last := 0
		for i, p := range probs {
			if p == 0 {
				continue
			}
			acc += p
			if r < acc {
				return i
			}
			last = i
		}
		return last // r rounded past the float32 total
	}

	for i := range probs {
		probs[i] /= sum
	}
	// before(a, b): a ranks ahead of b (higher probability, then lower id).
	before := func(a, b int32) bool {
		pa, pb := probs[a], probs[b]
		return pa > pb || (pa == pb && a < b)
	}

	var idx []int32
	if topK {
		idx = selectTopK(probs, opts.TopK, before)
	} else {
		idx = popUntilMass(probs, opts.TopP, before)
	}

	// top-p: keep the smallest prefix whose cumulative prob ≥ TopP.
	if topP {
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
		return int(idx[0])
	}
	r := rng.Float32() * renorm
	var acc float32
	for _, id := range idx {
		acc += probs[id]
		if r <= acc {
			return int(id)
		}
	}
	return int(idx[len(idx)-1])
}

// selectTopK returns the k highest-ranked ids in rank order, using a size-k
// heap whose root is the worst kept id (O(V log k), most ids rejected by one
// comparison).
func selectTopK(probs []float32, k int, before func(a, b int32) bool) []int32 {
	h := make([]int32, k)
	for i := range h {
		h[i] = int32(i)
	}
	worse := func(a, b int32) bool { return before(b, a) } // root = worst kept
	for i := k/2 - 1; i >= 0; i-- {
		siftDown(h, i, worse)
	}
	for i := k; i < len(probs); i++ {
		if before(int32(i), h[0]) {
			h[0] = int32(i)
			siftDown(h, 0, worse)
		}
	}
	slices.SortFunc(h, func(a, b int32) int {
		if before(a, b) {
			return -1
		}
		return 1
	})
	return h
}

// popUntilMass returns ids in rank order until their cumulative probability
// reaches p (the top-p nucleus, plus nothing more), popping a max-heap over
// the whole vocabulary. Past popSortCutoff pops the nucleus is wide, so the
// rest of the heap is sorted in one go instead.
func popUntilMass(probs []float32, p float32, before func(a, b int32) bool) []int32 {
	const popSortCutoff = 2048
	h := make([]int32, len(probs))
	for i := range h {
		h[i] = int32(i)
	}
	for i := len(h)/2 - 1; i >= 0; i-- {
		siftDown(h, i, before)
	}
	out := make([]int32, 0, 64)
	var cum float32
	for len(h) > 0 {
		if len(out) == popSortCutoff {
			slices.SortFunc(h, func(a, b int32) int {
				if before(a, b) {
					return -1
				}
				return 1
			})
			return append(out, h...)
		}
		top := h[0]
		out = append(out, top)
		cum += probs[top]
		if cum >= p {
			break
		}
		n := len(h) - 1
		h[0] = h[n]
		h = h[:n]
		siftDown(h, 0, before)
	}
	return out
}

// siftDown restores the heap property below i; first(a, b) means a belongs
// nearer the root.
func siftDown(h []int32, i int, first func(a, b int32) bool) {
	n := len(h)
	for {
		l := 2*i + 1
		if l >= n {
			return
		}
		c := l
		if r := l + 1; r < n && first(h[r], h[l]) {
			c = r
		}
		if !first(h[c], h[i]) {
			return
		}
		h[i], h[c] = h[c], h[i]
		i = c
	}
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
