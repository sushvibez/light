//go:build !nowebgpu

package main

import "light/internal/wgpudev"

// webgpuMemMB is the memory in Light's WebGPU buffers, in MB.
func webgpuMemMB() float64 {
	if d, err := wgpudev.Get(); err == nil {
		return float64(d.AllocatedBytes()) / (1 << 20)
	}
	return 0
}

// webgpuName is the WebGPU adapter and backend, if one is up.
func webgpuName() string {
	if d, err := wgpudev.Get(); err == nil {
		return d.AdapterName + " (" + d.BackendName + ")"
	}
	return ""
}
