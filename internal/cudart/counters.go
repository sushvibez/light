package cudart

import "sync/atomic"

// Counters are process-wide tallies of the driver / runtime calls Light makes,
// for finding launch-bound hot paths (one decode step of a small model is a
// few thousand launches). They are always on: one atomic add per call.
type Counters struct {
	Launches   uint64 // cuLaunchKernel
	Gemms      uint64 // cuBLAS / cuBLASLt GEMMs
	Mallocs    uint64 // cudaMalloc (misses of the caching allocator)
	Frees      uint64 // cudaFree
	HtoD       uint64 // host → device copies (sync + async)
	DtoH       uint64 // device → host copies (each one syncs)
	DtoD       uint64 // device → device copies
	Memsets    uint64
	Syncs      uint64 // cudaDeviceSynchronize
	CastCopies uint64 // same-dtype casts served by a DtoD copy
}

var counters Counters

func tally(p *uint64) { atomic.AddUint64(p, 1) }

// ReadCounters returns a snapshot of the tallies.
func ReadCounters() Counters {
	return Counters{
		Launches:   atomic.LoadUint64(&counters.Launches),
		Gemms:      atomic.LoadUint64(&counters.Gemms),
		Mallocs:    atomic.LoadUint64(&counters.Mallocs),
		Frees:      atomic.LoadUint64(&counters.Frees),
		HtoD:       atomic.LoadUint64(&counters.HtoD),
		DtoH:       atomic.LoadUint64(&counters.DtoH),
		DtoD:       atomic.LoadUint64(&counters.DtoD),
		Memsets:    atomic.LoadUint64(&counters.Memsets),
		Syncs:      atomic.LoadUint64(&counters.Syncs),
		CastCopies: atomic.LoadUint64(&counters.CastCopies),
	}
}

// Sub returns c − o field by field.
func (c Counters) Sub(o Counters) Counters {
	return Counters{
		Launches: c.Launches - o.Launches, Gemms: c.Gemms - o.Gemms,
		Mallocs: c.Mallocs - o.Mallocs, Frees: c.Frees - o.Frees,
		HtoD: c.HtoD - o.HtoD, DtoH: c.DtoH - o.DtoH, DtoD: c.DtoD - o.DtoD,
		Memsets: c.Memsets - o.Memsets, Syncs: c.Syncs - o.Syncs, CastCopies: c.CastCopies - o.CastCopies,
	}
}

// TallyCastCopy records a same-dtype cast served by a device copy.
func TallyCastCopy() { tally(&counters.CastCopies) }
