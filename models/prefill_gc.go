package models

import (
	"runtime"

	"light/device"
)

// prefillGC collects garbage every few layers of a long prefill on a GPU.
// Device buffers are released by tensor cleanups, which run only after a
// GC, and the collector paces itself on the small Go heap: within one
// forward over hundreds of tokens the dead activations of earlier layers
// (attention scores alone are heads·T² floats) would otherwise pile up and
// exhaust an integrated GPU's memory.
func prefillGC(dev device.Device, T, layer int) {
	if T >= 128 && layer%4 == 3 && !device.IsCPU(dev) {
		runtime.GC()
	}
}
