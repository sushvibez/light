package cudart

import (
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
)

// Tracing attributes the tallies: which kernels are launched and who issues
// the copies and memsets. It is off unless StartTrace is active (one atomic
// load per call otherwise) and meant for profiling tests, not production.
var (
	tracing atomic.Bool
	traceMu sync.Mutex
	traceN  map[string]int
)

// StartTrace begins attributing driver calls; StopTrace returns the counts.
func StartTrace() {
	traceMu.Lock()
	traceN = map[string]int{}
	traceMu.Unlock()
	tracing.Store(true)
}

// StopTrace ends tracing and returns the counts by kernel name ("launch
// <kernel>") or by the Light function that issued a copy or memset ("HtoD
// <pkg.Func>").
func StopTrace() map[string]int {
	tracing.Store(false)
	traceMu.Lock()
	defer traceMu.Unlock()
	out := traceN
	traceN = nil
	return out
}

// TraceTop formats the n largest counts of a StopTrace result.
func TraceTop(counts map[string]int, n int) string {
	type kv struct {
		k string
		v int
	}
	var all []kv
	for k, v := range counts {
		all = append(all, kv{k, v})
	}
	sort.Slice(all, func(i, j int) bool { return all[i].v > all[j].v || all[i].v == all[j].v && all[i].k < all[j].k })
	var b strings.Builder
	for i, e := range all {
		if i == n {
			break
		}
		b.WriteString("\n    ")
		b.WriteString(e.k)
		b.WriteString(": ")
		b.WriteString(itoa(e.v))
	}
	return b.String()
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}

func traceEvent(key string) {
	traceMu.Lock()
	traceN[key]++
	traceMu.Unlock()
}

// traceCaller records kind attributed to the nearest caller outside this
// package and the storage / tensor allocation layers.
func traceCaller(kind string) {
	var pcs [24]uintptr
	n := runtime.Callers(3, pcs[:])
	frames := runtime.CallersFrames(pcs[:n])
	var who []string
	for len(who) < 2 {
		f, more := frames.Next()
		fn := f.Function
		if !strings.HasPrefix(fn, "light/internal/cudart.") && !strings.HasPrefix(fn, "light/storage.") &&
			!strings.HasPrefix(fn, "light/tensor.") && !strings.HasPrefix(fn, "light/kernels/") &&
			!strings.HasPrefix(fn, "light/internal/registry.") && !strings.HasPrefix(fn, "light/ops.ensureContig") &&
			!strings.HasPrefix(fn, "light/ops.dispatch") && !strings.HasPrefix(fn, "light/ops.binaryDispatch") &&
			!strings.HasPrefix(fn, "light/autograd.") {
			who = append(who, strings.TrimPrefix(fn, "light/"))
		}
		if !more {
			break
		}
	}
	traceEvent(kind + " " + strings.Join(who, " < "))
}

// TraceCaller records an event of the given kind against its caller (for
// kernel launchers that want attribution beyond the kernel name).
func TraceCaller(kind string) {
	if tracing.Load() {
		traceCaller(kind)
	}
}
