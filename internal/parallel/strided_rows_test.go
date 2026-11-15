package parallel

import (
	"math/rand/v2"
	"testing"
)

// StridedRuns over coalesced axes visits exactly the (i, ao, bo) triples
// Strided does, for random shapes with broadcast (0) and permuted strides
// and every [lo, hi) split.
func TestStridedRunsMatchesStrided(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for trial := range 300 {
		rank := 1 + rng.IntN(4)
		dims := make([]int, rank)
		n := 1
		for i := range dims {
			dims[i] = 1 + rng.IntN(4)
			n *= dims[i]
		}
		strides := func() []int {
			s := make([]int, rank)
			acc := 1
			for i := rank - 1; i >= 0; i-- {
				switch rng.IntN(3) {
				case 0:
					s[i] = 0
				default:
					s[i] = acc
					acc *= dims[i]
				}
			}
			if rng.IntN(4) == 0 && rank > 1 { // a transposed operand
				i, j := rng.IntN(rank), rng.IntN(rank)
				if dims[i] == dims[j] {
					s[i], s[j] = s[j], s[i]
				}
			}
			return s
		}
		sa, sb := strides(), strides()
		lo := rng.IntN(n + 1)
		hi := lo + rng.IntN(n-lo+1)
		type t3 struct{ i, a, b int }
		var want, got []t3
		Strided(dims, sa, sb, 3, 5, lo, hi, func(i, ao, bo int) { want = append(want, t3{i, ao, bo}) })
		d, a, b := Coalesce(dims, sa, sb)
		StridedRuns(d, a, b, 3, 5, lo, hi, func(i, ao, bo, m int) {
			for k := range m {
				got = append(got, t3{i + k, ao + k*a[len(a)-1], bo + k*b[len(b)-1]})
			}
		})
		if len(got) != len(want) {
			t.Fatalf("trial %d dims %v sa %v sb %v [%d,%d): %d visits, want %d", trial, dims, sa, sb, lo, hi, len(got), len(want))
		}
		for k := range want {
			if got[k] != want[k] {
				t.Fatalf("trial %d dims %v sa %v sb %v [%d,%d): visit %d = %v, want %v", trial, dims, sa, sb, lo, hi, k, got[k], want[k])
			}
		}
	}
}
