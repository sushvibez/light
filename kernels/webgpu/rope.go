//go:build !nowebgpu

package webgpu

import (
	"encoding/binary"
	"math"
	"sync"

	"light/device"
	"light/dtype"
	"light/internal/gpu"
	"light/internal/registry"
	"light/internal/wgpudev"
)

func init() {
	registry.Register(registry.Key{Op: "rope", Dev: device.KindWebGPU, DType: dtype.F32}, ropeF32(0))
	registry.Register(registry.Key{Op: "rope_i", Dev: device.KindWebGPU, DType: dtype.F32}, ropeF32(1))
	registry.Register(registry.Key{Op: "rope_scaled", Dev: device.KindWebGPU, DType: dtype.F32}, ropeF32(0))
}

// invFreqCache holds base^(-2i/headDim) per (base, headDim), computed in
// f64 like the CPU kernel and uploaded once.
var invFreqCache = struct {
	sync.Mutex
	m map[[2]float64]*gpu.Buffer
}{m: map[[2]float64]*gpu.Buffer{}}

func invFreqBuffer(d *wgpudev.Device, base float64, headDim int) (*gpu.Buffer, error) {
	key := [2]float64{base, float64(headDim)}
	invFreqCache.Lock()
	defer invFreqCache.Unlock()
	if b, ok := invFreqCache.m[key]; ok {
		return b, nil
	}
	data := make([]byte, 4*(headDim/2))
	for i := range headDim / 2 {
		binary.LittleEndian.PutUint32(data[4*i:], math.Float32bits(float32(math.Pow(base, -float64(2*i)/float64(headDim)))))
	}
	b, err := d.StorageInit(data)
	if err != nil {
		return nil, err
	}
	invFreqCache.m[key] = b
	return b, nil
}

// ropeF32 is the launcher for rope (mode 0, rotate-half), rope_i (mode 1,
// interleaved) and rope_scaled (mode 0 with the caller's inverse
// frequencies as a third input).
func ropeF32(mode uint32) registry.Kernel {
	return func(ctx *registry.Context) error {
		x, positions, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
		d, err := dev()
		if err != nil {
			return err
		}
		x, err = contig(x)
		if err != nil {
			return err
		}
		positions, err = contig(positions)
		if err != nil {
			return err
		}
		hd, seq := x.Dim(-1), x.Dim(-2)
		if x.NumElements() == 0 {
			return nil
		}
		var invf *gpu.Buffer
		if len(ctx.Inputs) > 2 {
			f, err := contig(ctx.Inputs[2])
			if err != nil {
				return err
			}
			invf = raw(f)
		} else if invf, err = invFreqBuffer(d, ctx.Attrs["base"].(float64), hd); err != nil {
			return err
		}
		rows := x.NumElements() / hd
		params, err := d.Uniform(uint32(hd), uint32(seq), uint32(rows), mode)
		if err != nil {
			return err
		}
		defer params.Release()
		return run1D(d, "rope", []wgpudev.Binding{rd(raw(x)), rd(raw(positions)), rd(invf), rw(raw(out)), uni(params)}, rows*hd/2)
	}
}
