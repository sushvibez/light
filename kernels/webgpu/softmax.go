//go:build !nowebgpu

package webgpu

import (
	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/internal/wgpudev"
	"light/tensor"
)

func init() {
	registry.Register(registry.Key{Op: "softmax", Dev: device.KindWebGPU, DType: dtype.F32}, softmaxLauncher(0))
	registry.Register(registry.Key{Op: "logsoftmax", Dev: device.KindWebGPU, DType: dtype.F32}, softmaxLauncher(1))
}

// softmaxLauncher runs the last-axis row kernel. For any other axis the
// input is permuted so the axis is trailing, reduced, and the result is
// permuted back — all on device.
func softmaxLauncher(mode uint32) registry.Kernel {
	return func(ctx *registry.Context) error {
		a, out := ctx.Inputs[0], ctx.Output
		rank := a.Rank()
		axis := rank - 1
		if v, ok := ctx.Attrs["axis"].(int); ok {
			axis = v
		}
		if axis < 0 {
			axis += rank
		}
		if a.NumElements() == 0 {
			return nil
		}
		if axis == rank-1 {
			src, err := contig(a)
			if err != nil {
				return err
			}
			return rowSoftmax(src, out, mode)
		}
		// Move `axis` last: perm = [all other axes..., axis].
		perm := make([]int, 0, rank)
		for i := range rank {
			if i != axis {
				perm = append(perm, i)
			}
		}
		perm = append(perm, axis)
		pT, err := a.Permute(perm...)
		if err != nil {
			return err
		}
		pC, err := pT.Contiguous()
		if err != nil {
			return err
		}
		tmp, err := tensor.Zeros(dtype.F32, pC.Shape().Dims(), a.Device())
		if err != nil {
			return err
		}
		if err := rowSoftmax(pC, tmp, mode); err != nil {
			return err
		}
		// Inverse permutation back to the original layout, then copy into out.
		inv := make([]int, rank)
		for i, p := range perm {
			inv[p] = i
		}
		back, err := tmp.Permute(inv...)
		if err != nil {
			return err
		}
		return copyStridedToContig(out, back)
	}
}

func rowSoftmax(src, out *tensor.Tensor, mode uint32) error {
	d, err := dev()
	if err != nil {
		return err
	}
	cols := src.Dim(-1)
	rows := src.NumElements() / max(cols, 1)
	if cols == 0 {
		return nil
	}
	params, err := d.Uniform(uint32(cols), uint32(rows), mode, 0)
	if err != nil {
		return err
	}
	defer params.Release()
	return runRows(d, "softmax", []wgpudev.Binding{rd(raw(src)), rw(raw(out)), uni(params)}, rows)
}
