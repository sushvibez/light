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
	registry.Register(registry.Key{Op: "matmul", Dev: device.KindWebGPU, DType: dtype.F32}, matmulF32)
	registry.Register(registry.Key{Op: "matmul_batched", Dev: device.KindWebGPU, DType: dtype.F32}, matmulBatchedF32)
}

const matmulTile = 16

func matmulF32(ctx *registry.Context) error {
	a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	if a.Rank() != 2 || b.Rank() != 2 {
		return fmt.Errorf("webgpu matmul: only 2-D supported (got ranks %d and %d)", a.Rank(), b.Rank())
	}
	a, err := contig(a)
	if err != nil {
		return err
	}
	m, k := a.Dim(0), a.Dim(1)
	n := b.Dim(1)
	// nn.Linear passes W^T as a transpose view of the row-major (N, K)
	// weight: strides (1, K), offset 0. Read it in place via the transB
	// flag instead of materialising a K×N copy on every forward.
	transB := uint32(0)
	if b.Offset() == 0 && b.Strides().Get(0) == 1 && b.Strides().Get(1) == k && !(k == 1 && n == 1) {
		transB = 1
	} else {
		b, err = contig(b)
		if err != nil {
			return err
		}
	}
	d, err := dev()
	if err != nil {
		return err
	}
	if m == 0 || n == 0 {
		return nil
	}
	params, err := d.Uniform(uint32(m), uint32(n), uint32(k), transB)
	if err != nil {
		return err
	}
	defer params.Release()
	wx := uint32((m + matmulTile - 1) / matmulTile)
	wy := uint32((n + matmulTile - 1) / matmulTile)
	return d.Run("matmul", shader("matmul"), []wgpudev.Binding{rd(raw(a)), rd(raw(b)), rw(raw(out)), uni(params)}, wx, wy, 1)
}

func matmulBatchedF32(ctx *registry.Context) error {
	a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	if a.Rank() != 3 || b.Rank() != 3 {
		return fmt.Errorf("webgpu matmul_batched: need rank-3, got %d and %d", a.Rank(), b.Rank())
	}
	a, err := contig(a)
	if err != nil {
		return err
	}
	b, err = contig(b)
	if err != nil {
		return err
	}
	d, err := dev()
	if err != nil {
		return err
	}
	batch, m, k := a.Dim(0), a.Dim(1), a.Dim(2)
	n := b.Dim(2)
	if batch == 0 || m == 0 || n == 0 {
		return nil
	}
	if uint32(batch) > d.MaxWorkgroupsPerDim {
		return fmt.Errorf("webgpu matmul_batched: batch %d exceeds max workgroups per dimension %d", batch, d.MaxWorkgroupsPerDim)
	}
	params, err := d.Uniform(uint32(m), uint32(n), uint32(k), 0)
	if err != nil {
		return err
	}
	defer params.Release()
	wx := uint32((m + matmulTile - 1) / matmulTile)
	wy := uint32((n + matmulTile - 1) / matmulTile)
	return d.Run("batchedmatmul", shader("batchedmatmul"), []wgpudev.Binding{rd(raw(a)), rd(raw(b)), rw(raw(out)), uni(params)}, wx, wy, uint32(batch))
}
