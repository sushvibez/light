//go:build !nowebgpu

package webgpu

import (
	"fmt"

	"light/device"
	"light/internal/wgpudev"
	"light/tensor"
)

func init() {
	tensor.RegisterPitchedCopy(device.KindWebGPU, pitchedCopy)
}

// pitchedCopy is tensor.Assign's WebGPU path (a KV-cache append into a
// capacity buffer): one buffer-to-buffer copy per row, all in one command
// buffer. WebGPU copies need 4-byte aligned offsets and sizes.
func pitchedCopy(dst, src *tensor.Tensor, rows, width, pitch int) error {
	d, err := dev()
	if err != nil {
		return err
	}
	es := uint64(dst.DType().Size())
	w := uint64(width) * es
	so, do, p := uint64(src.Offset())*es, uint64(dst.Offset())*es, uint64(pitch)*es
	if w%4 != 0 || so%4 != 0 || do%4 != 0 || p%4 != 0 {
		return fmt.Errorf("webgpu: pitched copy of %v needs 4-byte aligned rows (width %d, pitch %d)", dst.DType(), width, pitch)
	}
	regions := make([]wgpudev.CopyRegion, rows)
	for r := range regions {
		regions[r] = wgpudev.CopyRegion{Src: raw(src), SrcOff: so + uint64(r)*w, Dst: raw(dst), DstOff: do + uint64(r)*p, Size: w}
	}
	return d.CopyMany(regions)
}
