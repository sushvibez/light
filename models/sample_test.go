package models

import (
	"math"
	"math/rand/v2"
	"sort"
	"testing"
)

// sampleLogitsRef is the original full-sort SampleLogits, kept as the
// reference the faster implementation is checked (and benchmarked) against.
func sampleLogitsRef(logits []float32, opts SampleOpts) int {
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
	sort.Slice(idx, func(a, b int) bool { return probs[idx[a]] > probs[idx[b]] })
	if opts.TopK > 0 && opts.TopK < V {
		idx = idx[:opts.TopK]
	}
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

// distinctLogits returns V logits with no exact ties (a random permutation
// of a spread of values), so the descending order is unique.
func distinctLogits(V int, scale float32, seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, 1))
	perm := r.Perm(V)
	out := make([]float32, V)
	for i, p := range perm {
		out[i] = scale * (float32(p)/float32(V) - 0.5)
	}
	return out
}

// TestSampleLogitsTruncatedMatchesRef: with top-k and/or top-p the new
// sampler walks the same survivor set in the same (descending) order as the
// full sort, so the same RNG stream yields the same token.
func TestSampleLogitsTruncatedMatchesRef(t *testing.T) {
	for _, V := range []int{1, 2, 7, 100, 5000} {
		logits := distinctLogits(V, 20, uint64(V))
		for _, temp := range []float32{0.3, 1, 2.5} {
			for _, k := range []int{0, 1, 3, 40, 70, 600, V, V + 5} {
				for _, p := range []float32{0, 0.05, 0.5, 0.9, 0.999, 1} {
					if (k <= 0 || k >= V) && (p <= 0 || p >= 1) {
						continue // unrestricted: distribution test below
					}
					opts := SampleOpts{Temperature: temp, TopK: k, TopP: p}
					opts.Rng = rand.New(rand.NewPCG(42, 7))
					ref := make([]int, 50)
					for i := range ref {
						ref[i] = sampleLogitsRef(logits, opts)
					}
					opts.Rng = rand.New(rand.NewPCG(42, 7))
					for i := range ref {
						if got := SampleLogits(logits, opts); got != ref[i] {
							t.Fatalf("V=%d T=%g k=%d p=%g draw %d: got %d, ref %d", V, temp, k, p, i, got, ref[i])
						}
					}
				}
			}
		}
	}
}

// TestSampleLogitsUnrestrictedDistribution: without truncation the sampler
// walks the vocabulary in index order instead of sorted order; the
// distribution is the same softmax.
func TestSampleLogitsUnrestrictedDistribution(t *testing.T) {
	logits := []float32{1, 3, -2, 2, 2, 0.5, -10}
	const T = 1.3
	want := make([]float64, len(logits))
	var z float64
	for i, x := range logits {
		want[i] = math.Exp(float64(x) / T)
		z += want[i]
	}
	const N = 400000
	got := make([]float64, len(logits))
	opts := SampleOpts{Temperature: T, Rng: rand.New(rand.NewPCG(1, 2))}
	for range N {
		got[SampleLogits(logits, opts)]++
	}
	for i := range want {
		w := want[i] / z
		g := got[i] / N
		sd := math.Sqrt(w * (1 - w) / N)
		if math.Abs(g-w) > 5*sd+1e-9 {
			t.Errorf("token %d: freq %.5f, want %.5f (±%.5f)", i, g, w, 5*sd)
		}
	}
}

func TestSampleLogitsEdgeCases(t *testing.T) {
	if SampleLogits(nil, SampleOpts{Temperature: 1}) != -1 {
		t.Fatal("empty logits")
	}
	if got := SampleLogits([]float32{1, 5, 5, 2}, SampleOpts{}); got != 1 {
		t.Fatalf("greedy tie: got %d, want 1", got)
	}
	nan := float32(math.NaN())
	if got := SampleLogits([]float32{0, nan, 3}, SampleOpts{Temperature: 1, Rng: rand.New(rand.NewPCG(1, 1))}); got != argmax([]float32{0, nan, 3}) {
		t.Fatalf("NaN fallback: got %d", got)
	}
	// top-k = 1 is greedy over the probabilities.
	logits := distinctLogits(1000, 30, 3)
	for range 20 {
		if got := SampleLogits(logits, SampleOpts{Temperature: 0.7, TopK: 1}); got != argmax(logits) {
			t.Fatalf("top-k 1: got %d, want %d", got, argmax(logits))
		}
	}
}

func benchSample(b *testing.B, f func([]float32, SampleOpts) int, opts SampleOpts) {
	logits := make([]float32, 152064)
	r := rand.New(rand.NewPCG(9, 9))
	for i := range logits {
		logits[i] = float32(r.NormFloat64() * 3)
	}
	opts.Rng = rand.New(rand.NewPCG(1, 1))
	b.ResetTimer()
	for range b.N {
		f(logits, opts)
	}
}

func BenchmarkSampleLogits(b *testing.B) {
	cases := []struct {
		name string
		opts SampleOpts
	}{
		{"plain", SampleOpts{Temperature: 0.8}},
		{"topk40", SampleOpts{Temperature: 0.8, TopK: 40}},
		{"topp0.9", SampleOpts{Temperature: 0.8, TopP: 0.9}},
		{"topk40_topp0.9", SampleOpts{Temperature: 0.8, TopK: 40, TopP: 0.9}},
		{"topk5000", SampleOpts{Temperature: 0.8, TopK: 5000}},
	}
	for _, c := range cases {
		b.Run("new/"+c.name, func(b *testing.B) { benchSample(b, SampleLogits, c.opts) })
		b.Run("ref/"+c.name, func(b *testing.B) { benchSample(b, sampleLogitsRef, c.opts) })
	}
}
