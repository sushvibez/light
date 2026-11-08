//go:build !nowebgpu

package webgpu

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/internal/wgpudev"
)

func init() {
	for _, dt := range []dtype.DType{dtype.F32, dtype.I32, dtype.I64, dtype.U32} {
		registry.Register(registry.Key{Op: "cast", Dev: device.KindWebGPU, DType: dt}, castLauncher)
	}
}

// castModes maps (from, to) to the shader mode. WGSL has no 64-bit integer
// type, so the I64 paths are emulated word-wise (see cast.wgsl); F64, F16
// and BF16 have no WebGPU cast.
var castModes = map[[2]dtype.DType]uint32{
	{dtype.F32, dtype.I32}: 0,
	{dtype.I32, dtype.F32}: 1,
	{dtype.F32, dtype.I64}: 2,
	{dtype.I64, dtype.F32}: 3,
	{dtype.I32, dtype.I64}: 4,
	{dtype.I64, dtype.I32}: 5,
	{dtype.F32, dtype.U32}: 6,
	{dtype.U32, dtype.F32}: 7,
}

func castLauncher(ctx *registry.Context) error {
	a, out := ctx.Inputs[0], ctx.Output
	from, to := a.DType(), out.DType()
	d, err := dev()
	if err != nil {
		return err
	}
	n := a.NumElements()
	if n == 0 {
		return nil
	}
	if from == to {
		nb := uint64(from.ByteSize(n))
		return d.Copy(raw(a), uint64(a.Offset()*from.Size()), raw(out), 0, (nb+3)&^3)
	}
	mode, ok := castModes[[2]dtype.DType{from, to}]
	if !ok {
		return fmt.Errorf("webgpu cast: unsupported pair %v → %v", from, to)
	}
	params, err := d.Uniform(mode, uint32(n), 0, 0)
	if err != nil {
		return err
	}
	defer params.Release()
	return run1D(d, "cast", []wgpudev.Binding{rd(raw(a)), rw(raw(out)), uni(params)}, n)
}
