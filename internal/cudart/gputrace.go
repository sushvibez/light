//go:build !nocuda && cgo

package cudart

/*
#include <cuda_runtime.h>
*/
import "C"

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"unsafe"
)

// GPU-time attribution: with StartGPUTrace on, every kernel launch and GEMM
// is bracketed by a pair of events on the current stream. Recorded during
// stream capture the pairs become event nodes of the graph, so after a
// replay StopGPUTrace reads the time each node took inside the graph
// (its kernel plus the gap before it), which no host-side timer sees.
var (
	gpuTracing atomic.Bool
	gpuMu      sync.Mutex
	gpuSpans   []gpuSpan
)

type gpuSpan struct {
	name string
	a, b C.cudaEvent_t
}

// GPUTime is one kernel's share of a traced run.
type GPUTime struct {
	Count  int
	Micros float64
}

// StartGPUTrace begins bracketing launches with events.
func StartGPUTrace() {
	gpuMu.Lock()
	gpuSpans = nil
	gpuMu.Unlock()
	gpuTracing.Store(true)
}

// StopGPUTrace waits for the device, reads every span and returns the
// time per kernel name. Spans recorded under capture reflect the last
// replay of the graph.
func StopGPUTrace() (map[string]GPUTime, error) {
	gpuTracing.Store(false)
	if err := DeviceSynchronize(); err != nil {
		return nil, err
	}
	gpuMu.Lock()
	spans := gpuSpans
	gpuSpans = nil
	gpuMu.Unlock()
	out := map[string]GPUTime{}
	for _, s := range spans {
		var ms C.float
		rc := C.cudaEventElapsedTime(&ms, s.a, s.b)
		C.cudaEventDestroy(s.a)
		C.cudaEventDestroy(s.b)
		if rc != C.cudaSuccess {
			C.cudaGetLastError()
			continue
		}
		t := out[s.name]
		t.Count++
		t.Micros += float64(ms) * 1000
		out[s.name] = t
	}
	return out, nil
}

// gpuSpanBegin records the opening event of a span on the current stream
// (nil when not tracing).
func gpuSpanBegin(name string) *gpuSpan {
	if !gpuTracing.Load() {
		return nil
	}
	s := &gpuSpan{name: name}
	if C.cudaEventCreate(&s.a) != C.cudaSuccess || C.cudaEventCreate(&s.b) != C.cudaSuccess {
		return nil
	}
	C.cudaEventRecord(s.a, C.cudaStream_t(unsafe.Pointer(current.Load())))
	return s
}

func (s *gpuSpan) end() {
	if s == nil {
		return
	}
	C.cudaEventRecord(s.b, C.cudaStream_t(unsafe.Pointer(current.Load())))
	gpuMu.Lock()
	gpuSpans = append(gpuSpans, *s)
	gpuMu.Unlock()
}

// GPUTraceTop formats the n largest entries by total time.
func GPUTraceTop(t map[string]GPUTime, n int) string {
	type kv struct {
		k string
		v GPUTime
	}
	var all []kv
	total := 0.0
	for k, v := range t {
		all = append(all, kv{k, v})
		total += v.Micros
	}
	sort.Slice(all, func(i, j int) bool { return all[i].v.Micros > all[j].v.Micros })
	var sb strings.Builder
	fmt.Fprintf(&sb, "\n        %6s %6s %8s %8s  %s", "pct", "count", "avg µs", "total µs", "kernel")
	for i, e := range all {
		if i == n {
			break
		}
		fmt.Fprintf(&sb, "\n        %5.1f%% %6d %8.1f %8.0f  %s", 100*e.v.Micros/total, e.v.Count, e.v.Micros/float64(e.v.Count), e.v.Micros, e.k)
	}
	fmt.Fprintf(&sb, "\n        total %.0f µs over %d spans", total, len(gpuSpansCount(t)))
	return sb.String()
}

func gpuSpansCount(t map[string]GPUTime) []int {
	n := 0
	for _, v := range t {
		n += v.Count
	}
	return make([]int, n)
}
