package nn

import (
	"fmt"
	"sync"
)

// ExpertCache is ExpertStream's residency policy for MoE layers that run
// their experts one at a time from per-expert weights of any shape V (FP8
// block-scaled Linears, widened FP4, a latent MLP): at most Slots experts
// stay loaded, the rest are read by Load when picked and the least
// recently used is dropped. Prefetch starts background loads (a later
// layer's predicted picks); Get waits for a pending one. A dropped expert
// the caller still holds stays valid until it lets go of it.
type ExpertCache[V any] struct {
	Load  func(e int) (V, error)
	Slots int
	// Loads counts experts read in; Hits Gets served from memory.
	// Prefetched counts background loads Prefetch started; PrefetchUsed
	// those a Get then asked for.
	Loads, Hits, Prefetched, PrefetchUsed int64

	mu      sync.Mutex
	res     map[int]*cachedExpert[V]
	clock   uint64
	pending map[int]*pendingExpert[V]
	fromPF  map[int]bool
}

type cachedExpert[V any] struct {
	v    V
	used uint64
}

type pendingExpert[V any] struct {
	done chan struct{}
	v    V
	err  error
}

// NewExpertCache returns a cache holding up to slots experts.
func NewExpertCache[V any](slots int, load func(e int) (V, error)) (*ExpertCache[V], error) {
	if slots < 1 || load == nil {
		return nil, fmt.Errorf("nn.ExpertCache: need ≥ 1 slot and a loader")
	}
	return &ExpertCache[V]{Load: load, Slots: slots, res: map[int]*cachedExpert[V]{},
		pending: map[int]*pendingExpert[V]{}, fromPF: map[int]bool{}}, nil
}

// start launches expert e's load; it joins the cache when done (c.mu held).
func (c *ExpertCache[V]) start(e int) *pendingExpert[V] {
	p := &pendingExpert[V]{done: make(chan struct{})}
	c.pending[e] = p
	go func() {
		v, err := c.Load(e)
		c.mu.Lock()
		p.v, p.err = v, err
		delete(c.pending, e)
		if err == nil {
			c.Loads++
			c.install(e, v)
		} else {
			delete(c.fromPF, e)
		}
		c.mu.Unlock()
		close(p.done)
	}()
	return p
}

// install adds e, dropping the least recently used expert when full
// (c.mu held).
func (c *ExpertCache[V]) install(e int, v V) {
	for len(c.res) >= c.Slots {
		old, oldUse := -1, uint64(0)
		for k, r := range c.res {
			if old < 0 || r.used < oldUse {
				old, oldUse = k, r.used
			}
		}
		delete(c.res, old)
		delete(c.fromPF, old)
	}
	c.clock++
	c.res[e] = &cachedExpert[V]{v: v, used: c.clock}
}

// Prefetch starts reading the given experts in the background unless they
// are loaded or already on their way; at most Slots are started.
func (c *ExpertCache[V]) Prefetch(ids []int) { c.prefetch(ids, true) }

// prefetch is Prefetch; counted marks the loads as predictions (Each's
// look-ahead is not one).
func (c *ExpertCache[V]) prefetch(ids []int, counted bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	n := 0
	for _, e := range ids {
		if n == c.Slots {
			break
		}
		n++
		if _, ok := c.res[e]; ok {
			continue
		}
		if _, ok := c.pending[e]; ok {
			continue
		}
		c.start(e)
		if counted {
			c.Prefetched++
			c.fromPF[e] = true
		}
	}
}

// Get returns expert e, loading it (or waiting for its prefetch) if needed.
func (c *ExpertCache[V]) Get(e int) (V, error) {
	c.mu.Lock()
	if c.fromPF[e] {
		c.PrefetchUsed++
		delete(c.fromPF, e)
	}
	if r, ok := c.res[e]; ok {
		c.clock++
		r.used = c.clock
		c.Hits++
		c.mu.Unlock()
		return r.v, nil
	}
	p, ok := c.pending[e]
	if !ok {
		p = c.start(e)
	}
	c.mu.Unlock()
	<-p.done
	if p.err != nil {
		var zero V
		return zero, fmt.Errorf("nn.ExpertCache: expert %d: %w", e, p.err)
	}
	return p.v, nil
}

// Each calls f on the experts in order, loading the next Slots-1 in the
// background while f runs on the current one. It stops at f's first error.
func (c *ExpertCache[V]) Each(order []int, f func(e int, v V) error) error {
	for i, e := range order {
		if c.Slots > 1 && i+1 < len(order) {
			c.prefetch(order[i+1:min(len(order), i+c.Slots)], false)
		}
		v, err := c.Get(e)
		if err != nil {
			return err
		}
		if err := f(e, v); err != nil {
			return err
		}
	}
	return nil
}

// Resident lists the experts currently loaded.
func (c *ExpertCache[V]) Resident() []int {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]int, 0, len(c.res))
	for e := range c.res {
		out = append(out, e)
	}
	return out
}
