package diffusion

import (
	"fmt"
	"os"
	"runtime/metrics"
	"time"

	"light/device"
	"light/internal/cudart"
	"light/storage"
	"light/tensor"
)

// phaseTimer prints how long each pipeline phase took to stderr when
// LIGHT_TIMINGS is set (a profiling aid; off by default). A phase that
// queued device work passes a tensor it produced: reading one element back
// waits for the device, so the time is the device's, not the launch queue's.
type phaseTimer struct {
	on    bool
	t     time.Time
	total time.Time
	host  hostCounts
}

// hostCounts are the host-side events of a phase that can stall the GPU
// queue: driver calls, allocator slow paths and Go collections.
type hostCounts struct {
	cu   cudart.Counters
	gcs  int64
	pool storage.PoolStats
}

func readHostCounts() hostCounts {
	s := []metrics.Sample{{Name: "/gc/cycles/total:gc-cycles"}}
	metrics.Read(s)
	var h hostCounts
	h.cu = cudart.ReadCounters()
	h.gcs = int64(s[0].Value.Uint64())
	h.pool = storage.CUDAPoolStats()
	return h
}

func newPhaseTimer() *phaseTimer {
	now := time.Now()
	pt := &phaseTimer{on: os.Getenv("LIGHT_TIMINGS") != "", t: now, total: now}
	if pt.on {
		pt.host = readHostCounts()
	}
	return pt
}

func (pt *phaseTimer) mark(phase string, sync *tensor.Tensor) {
	if !pt.on {
		return
	}
	if sync != nil {
		waitFor(sync)
	}
	now := time.Now()
	h := readHostCounts()
	d := h.cu.Sub(pt.host.cu)
	p, q := h.pool, pt.host.pool
	fmt.Fprintf(os.Stderr, "timing %-18s %7.2f s  (total %.1f s)  launch %d gemm %d malloc %d free %d memset %d dtoh %d htod %d dtod %d sync %d | gc %d (forced %d, %.0f ms) oom-retry %d evict %d (%.0f ms)\n",
		phase, now.Sub(pt.t).Seconds(), now.Sub(pt.total).Seconds(), d.Launches, d.Gemms, d.Mallocs, d.Frees, d.Memsets,
		d.DtoH, d.HtoD, d.DtoD, d.Syncs, h.gcs-pt.host.gcs, p.ForcedGCs-q.ForcedGCs, (p.GCTime-q.GCTime).Seconds()*1e3,
		p.OOMRetries-q.OOMRetries, p.Evictions-q.Evictions, (p.EvictTime-q.EvictTime).Seconds()*1e3)
	pt.t, pt.host = now, h
}

// waitFor reads one element of t back to the host.
func waitFor(t *tensor.Tensor) {
	if t.Device().Kind() == device.KindCPU || t.NumElements() == 0 {
		return
	}
	if c, err := t.Contiguous(); err == nil {
		if f, err := c.Reshape(c.NumElements()); err == nil {
			if one, err := f.Narrow(0, 0, 1); err == nil {
				_, _ = one.To(device.CPU())
				return
			}
		}
	}
	_, _ = t.To(device.CPU())
}

// gpuProfileStep is the denoising step LIGHT_PROFILE_GPU traces (the
// second: the first warms caches and allocations).
const gpuProfileStep = 1

// stepProfile brackets step i with the CUDA per-kernel GPU-time trace when
// LIGHT_PROFILE_GPU is set, printing the top kernels after it; it returns
// the function to call once the step is done.
func stepProfile(i int, dev device.Device) func() {
	if i != gpuProfileStep || os.Getenv("LIGHT_PROFILE_GPU") == "" || !device.IsCUDA(dev) {
		return func() {}
	}
	cudart.StartGPUTrace()
	return func() {
		t, err := cudart.StopGPUTrace()
		if err != nil {
			fmt.Fprintf(os.Stderr, "gpu trace: %v\n", err)
			return
		}
		fmt.Fprintf(os.Stderr, "gpu trace, step %d:\n%s\n", i, cudart.GPUTraceTop(t, 40))
	}
}
