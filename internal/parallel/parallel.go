// Package parallel splits a loop over [0, n) across GOMAXPROCS goroutines.
package parallel

import (
	"runtime"
	"sync"
)

// For runs fn on contiguous chunks [lo, hi) covering [0, n). Each chunk has
// at least grain items, so small loops run inline on the caller.
func For(n, grain int, fn func(lo, hi int)) {
	if n <= 0 {
		return
	}
	workers := runtime.GOMAXPROCS(0)
	if grain < 1 {
		grain = 1
	}
	if w := n / grain; w < workers {
		workers = w
	}
	if workers <= 1 {
		fn(0, n)
		return
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func() {
			defer wg.Done()
			fn(lo, hi)
		}()
	}
	wg.Wait()
}

// Grain is the element count below which elementwise loops stay on the
// caller's goroutine.
const Grain = 1 << 15

// Strided visits the row-major elements [lo, hi) of an array of shape dims
// with the matching element offsets into two strided operands (sb may be
// nil). Offsets are advanced incrementally, not recomputed per element.
func Strided(dims, sa, sb []int, baseA, baseB, lo, hi int, visit func(i, ao, bo int)) {
	rank := len(dims)
	idx := make([]int, rank)
	ao, bo := baseA, baseB
	r := lo
	for ax := rank - 1; ax >= 0; ax-- {
		idx[ax] = r % dims[ax]
		r /= dims[ax]
		ao += idx[ax] * sa[ax]
		if sb != nil {
			bo += idx[ax] * sb[ax]
		}
	}
	for i := lo; i < hi; i++ {
		visit(i, ao, bo)
		for ax := rank - 1; ax >= 0; ax-- {
			idx[ax]++
			ao += sa[ax]
			if sb != nil {
				bo += sb[ax]
			}
			if idx[ax] < dims[ax] {
				break
			}
			ao -= dims[ax] * sa[ax]
			if sb != nil {
				bo -= dims[ax] * sb[ax]
			}
			idx[ax] = 0
		}
	}
}
