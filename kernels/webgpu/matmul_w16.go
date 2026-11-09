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
	for _, dt := range []dtype.DType{dtype.BF16, dtype.F16} {
		registry.Register(registry.Key{Op: "matmul_w16", Dev: device.KindWebGPU, DType: dt}, matmulW16)
	}
}

// matmulW16: ops.MatMulW16 on WebGPU — x (M, K) F32 times a 16-bit weight
// (N, K) transposed, F32 out. Decode (M ≤ 8) reads each weight word once
// in gemv16; larger M runs the tiled matmul16.
func matmulW16(ctx *registry.Context) error {
	x, w, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	M, K, N := x.Dim(0), x.Dim(1), w.Dim(0)
	if K%2 != 0 {
		return fmt.Errorf("webgpu matmul_w16: K %d must be even (16-bit weights are read as word pairs)", K)
	}
	mode := uint32(0)
	if w.DType() == dtype.F16 {
		mode = 1
	}
	d, err := dev()
	if err != nil {
		return err
	}
	if x, err = contig(x); err != nil {
		return err
	}
	if w, err = contig(w); err != nil {
		return err
	}
	params, err := d.Uniform(uint32(M), uint32(N), uint32(K), mode)
	if err != nil {
		return err
	}
	defer params.Release()
	binds := []wgpudev.Binding{rd(raw(x)), rd(raw(w)), rw(raw(out)), uni(params)}
	if M <= gemvMaxM {
		gx, gy := d.Rows(uint64(N))
		return d.Run("gemv16", shader("gemv16"), binds, gx, gy, 1)
	}
	return d.Run("matmul16", shader("matmul16"), binds, uint32((M+matmulTile-1)/matmulTile), uint32((N+matmulTile-1)/matmulTile), 1)
}
