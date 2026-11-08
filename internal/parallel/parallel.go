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
