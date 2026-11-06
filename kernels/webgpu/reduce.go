//go:build webgpu

package webgpu

import (
	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/internal/wgpudev"
)

var reduceOp = map[string]uint32{"sum": 0, "mean": 1, "max": 2, "min": 3, "prod": 4}

func init() {
	for op, code := range reduceOp {
		registry.Register(registry.Key{Op: op, Dev: device.KindWebGPU, DType: dtype.F32}, makeReduce(code))
	}
}

// makeReduce follows the CUDA convention: a trailing-axes row kernel, with
// a launcher-side permute+contiguify when the reduced axes are not already
// the trailing ones. The output allocated by ops.reduceDispatch has the
// non-reduced axes in their original relative order, which is exactly the
// row order produced by permuteForReduce.
func makeReduce(code uint32) registry.Kernel {
	return func(ctx *registry.Context) error {
		a, out := ctx.Inputs[0], ctx.Output
		axes, _ := ctx.Attrs["axes"].([]int)
		rank := a.Rank()
		if !isTrailing(axes, rank) {
			perm, newAxes := permuteForReduce(rank, axes)
			pT, err := a.Permute(perm...)
			if err != nil {
				return err
			}
			a, err = pT.Contiguous()
			if err != nil {
				return err
			}
			axes = newAxes
		} else {
			var err error
			a, err = contig(a)
			if err != nil {
				return err
			}
		}
		dims := a.Shape().Dims()
		cols := 1
		for _, ax := range axes {
			if ax < 0 {
				ax += rank
			}
			cols *= dims[ax]
		}
		total := a.NumElements()
		if total == 0 || cols == 0 {
			return nil
		}
		rows := total / cols
		d, err := dev()
		if err != nil {
			return err
		}
		params, err := d.Uniform(code, uint32(rows), uint32(cols), 0)
		if err != nil {
			return err
		}
		defer params.Release()
		return runRows(d, "rowreduce", []wgpudev.Binding{rd(raw(a)), rw(raw(out)), uni(params)}, rows)
	}
}

// isTrailing reports whether axes is exactly the trailing len(axes) axes.
func isTrailing(axes []int, rank int) bool {
	if len(axes) == 0 {
		return rank == 0
	}
	seen := make(map[int]bool, len(axes))
	for _, ax := range axes {
		if ax < 0 {
			ax += rank
		}
		seen[ax] = true
	}
	k := len(seen)
	for i := rank - k; i < rank; i++ {
		if !seen[i] {
			return false
		}
	}
	return true
}

// permuteForReduce moves the reduction axes to the trailing positions,
// keeping non-reduced axes in their original order.
func permuteForReduce(rank int, axes []int) (perm []int, newAxes []int) {
	axSet := make(map[int]bool, len(axes))
	for _, ax := range axes {
		if ax < 0 {
			ax += rank
		}
		axSet[ax] = true
	}
	for i := range rank {
		if !axSet[i] {
			perm = append(perm, i)
		}
	}
	for _, ax := range axes {
		if ax < 0 {
			ax += rank
		}
		perm = append(perm, ax)
	}
	start := rank - len(axes)
	newAxes = make([]int, len(axes))
	for i := range axes {
		newAxes[i] = start + i
	}
	return perm, newAxes
}
