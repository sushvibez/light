//go:build !nocuda && cgo

package cuda

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
	"light/storage"
)

func init() {
	for _, dt := range []dtype.DType{dtype.BF16, dtype.F16} {
		registry.Register(registry.Key{Op: "matmul_w16", Dev: device.KindCUDA, DType: dt}, matmulW16CUDA)
	}
}

// matmulW16CUDA is ops.MatMulW16 on CUDA: x (M, K) F32 times a 16-bit weight
// (N, K) in nn.Linear's layout, transposed, into out (M, N) F32 — the
// activations stay F32 (unlike matmul_mixed, which rounds them to BF16).
// The weight is widened into an F32 scratch block taken straight from the
// caching pool and handed back right after the SGEMM is queued, so a model
// stored at 16 bits holds one widened weight at a time instead of a GC's
// worth of F32 copies. Reuse is safe: everything runs in order on one stream.
func matmulW16CUDA(ctx *registry.Context) error {
	x, w, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	if x.Rank() != 2 || w.Rank() != 2 || x.Dim(1) != w.Dim(1) {
		return fmt.Errorf("cuda matmul_w16: want x (M, K) and w (N, K), got %v, %v", x.Shape(), w.Shape())
	}
	if x.DType() != dtype.F32 || out.DType() != dtype.F32 {
		return fmt.Errorf("cuda matmul_w16: want F32 x and out, got %v, %v", x.DType(), out.DType())
	}
	var err error
	if !x.IsContiguous() || x.Offset() != 0 {
		if x, err = x.Contiguous(); err != nil {
			return err
		}
	}
	if !w.IsContiguous() || w.Offset() != 0 {
		if w, err = w.Contiguous(); err != nil {
			return err
		}
	}
	M, K, N := x.Dim(0), x.Dim(1), w.Dim(0)
	ord := out.Device().Ordinal()
	if err := cudart.SetDevice(ord); err != nil {
		return err
	}
	scratch, err := storage.NewCUDA(out.Device(), dtype.F32, N*K)
	if err != nil {
		return err
	}
	// Untracked by the GC: one Acquire / Release pair returns the block to the
	// pool as soon as the GEMM is queued.
	scratch.Acquire()
	defer scratch.Release()
	if err := launchCast(w.DType(), dtype.F32, w.Storage().CUDAPtr(), scratch.CUDAPtr(), uint64(N*K)); err != nil {
		return fmt.Errorf("cuda matmul_w16: widen weight: %w", err)
	}
	h, err := cudart.GetBlasHandle(ord)
	if err != nil {
		return err
	}
	// out = x · Wᵀ with W row-major (N, K): the transposed-B SGEMM nn.Linear's
	// F32 path uses.
	return cudart.SgemmRowMajorTransB(h, M, N, K, x.Storage().CUDAPtr(), scratch.CUDAPtr(), out.Storage().CUDAPtr())
}
