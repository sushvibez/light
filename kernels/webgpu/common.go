//go:build !nowebgpu

package webgpu

import (
	"encoding/binary"
	"fmt"
	"math"

	"light/internal/wgpudev"
	"light/tensor"

	"light/internal/gpu"
)

const wgSize = 256

// dev returns the process-wide WebGPU context.
func dev() (*wgpudev.Device, error) { return wgpudev.Get() }

// raw returns the wgpu buffer backing t's storage.
func raw(t *tensor.Tensor) *gpu.Buffer { return t.Storage().WebGPUBuffer().Buf }

// rd / rw / uni build Binding entries.
func rd(b *gpu.Buffer) wgpudev.Binding { return wgpudev.Binding{Kind: wgpudev.ReadStorage, Buf: b} }
func rw(b *gpu.Buffer) wgpudev.Binding {
	return wgpudev.Binding{Kind: wgpudev.ReadWriteStorage, Buf: b}
}
func uni(b *gpu.Buffer) wgpudev.Binding { return wgpudev.Binding{Kind: wgpudev.Uniform, Buf: b} }

func f32bits(f float32) uint32 { return math.Float32bits(f) }

// contig returns t itself when it is packed with zero offset, otherwise a
// fresh contiguous copy on the same device.
func contig(t *tensor.Tensor) (*tensor.Tensor, error) {
	if t.IsContiguous() && t.Offset() == 0 {
		return t, nil
	}
	return t.Contiguous()
}

// i32Bytes packs ints as little-endian int32 words for a meta buffer.
func i32Bytes(vals ...[]int) []byte {
	n := 0
	for _, v := range vals {
		n += len(v)
	}
	out := make([]byte, 4*n)
	i := 0
	for _, v := range vals {
		for _, x := range v {
			if x > math.MaxInt32 || x < math.MinInt32 {
				panic(fmt.Sprintf("webgpu: meta value %d overflows i32", x))
			}
			binary.LittleEndian.PutUint32(out[4*i:], uint32(int32(x)))
			i++
		}
	}
	return out
}

// run1D dispatches a 1-thread-per-element shader over n elements.
func run1D(d *wgpudev.Device, name string, binds []wgpudev.Binding, n int) error {
	x, y := d.Grid1D(uint64(n), wgSize)
	return d.Run(name, shader(name), binds, x, y, 1)
}

// runRows dispatches a workgroup-per-row shader over rows rows.
func runRows(d *wgpudev.Device, name string, binds []wgpudev.Binding, rows int) error {
	x, y := d.Rows(uint64(rows))
	return d.Run(name, shader(name), binds, x, y, 1)
}
