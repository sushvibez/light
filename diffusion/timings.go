package diffusion

import (
	"fmt"
	"os"
	"time"

	"light/device"
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
}

func newPhaseTimer() *phaseTimer {
	now := time.Now()
	return &phaseTimer{on: os.Getenv("LIGHT_TIMINGS") != "", t: now, total: now}
}

func (pt *phaseTimer) mark(phase string, sync *tensor.Tensor) {
	if !pt.on {
		return
	}
	if sync != nil {
		waitFor(sync)
	}
	now := time.Now()
	fmt.Fprintf(os.Stderr, "timing %-18s %7.2f s  (total %.1f s)\n", phase, now.Sub(pt.t).Seconds(), now.Sub(pt.total).Seconds())
	pt.t = now
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
