//go:build webgpu

package webgpu

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/internal/wgpudev"
)

func init() {
	// The gather kernel moves 32-bit words, so any 4- or 8-byte dtype works.
	for _, dt := range []dtype.DType{dtype.F32, dtype.I32, dtype.U32, dtype.F64, dtype.I64, dtype.U64} {
		registry.Register(registry.Key{Op: "index_select", Dev: device.KindWebGPU, DType: dt}, indexSelect)
	}
	for _, dt := range []dtype.DType{dtype.F32, dtype.I32, dtype.U32, dtype.F64, dtype.I64, dtype.U64} {
		registry.Register(registry.Key{Op: "concat", Dev: device.KindWebGPU, DType: dt}, concat)
	}
}

func indexSelect(ctx *registry.Context) error {
	src, idx, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	var idxStride uint32
	switch idx.DType() {
	case dtype.I64, dtype.U64:
		idxStride = 2
	case dtype.I32, dtype.U32:
		idxStride = 1
	default:
		return fmt.Errorf("webgpu index_select: indices must be I64 or I32, got %v", idx.DType())
	}
	src, err := contig(src)
	if err != nil {
		return err
	}
	idx, err = contig(idx)
	if err != nil {
		return err
	}
	d, err := dev()
	if err != nil {
		return err
	}
	rowSize := 1
	for _, s := range src.Shape().Dims()[1:] {
		rowSize *= s
	}
	rowWords := rowSize * src.DType().Size() / 4
	numIdx := idx.Dim(0)
	n := numIdx * rowWords
	if n == 0 {
		return nil
	}
	params, err := d.Uniform(uint32(numIdx), uint32(rowWords), idxStride, 0)
	if err != nil {
		return err
	}
	defer params.Release()
	return run1D(d, "indexselect", []wgpudev.Binding{rd(raw(src)), rd(raw(idx)), rw(raw(out)), uni(params)}, n)
}

// concat is a structured device-to-device copy: for each outer index, the
// axis-slab of every input is copied in turn. All copies go in one command
// buffer.
func concat(ctx *registry.Context) error {
	axis := ctx.Attrs["axis"].(int)
	out := ctx.Output
	elemSize := out.DType().Size()
	outDims := out.Shape().Dims()
	outer, inner := 1, 1
	for i, s := range outDims {
		if i < axis {
			outer *= s
		} else if i > axis {
			inner *= s
		}
	}
	innerBytes := uint64(inner * elemSize)
	dstRowStride := uint64(outDims[axis]) * innerBytes
	d, err := dev()
	if err != nil {
		return err
	}
	var regions []wgpudev.CopyRegion
	dstBuf := raw(out)
	for o := range outer {
		dstOff := uint64(o) * dstRowStride
		for _, src := range ctx.Inputs {
			srcRowStride := uint64(src.Dim(axis)) * innerBytes
			if srcRowStride == 0 {
				continue
			}
			srcOff := uint64(o)*srcRowStride + uint64(src.Offset()*elemSize)
			regions = append(regions, wgpudev.CopyRegion{
				Src: raw(src), SrcOff: srcOff, Dst: dstBuf, DstOff: dstOff, Size: srcRowStride,
			})
			dstOff += srcRowStride
		}
	}
	return d.CopyMany(regions)
}
