//go:build !nocuda && cgo

package cuda

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
	"light/tensor"
)

func init() {
	registry.Register(registry.Key{Op: "matmul", Dev: device.KindCUDA, DType: dtype.F32}, matmulF32)
	registry.Register(registry.Key{Op: "matmul", Dev: device.KindCUDA, DType: dtype.F64}, matmulF64)
}

func matmulF32(ctx *registry.Context) error {
	a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	if a.Rank() != 2 || b.Rank() != 2 {
		return fmt.Errorf("cuda matmul: only 2-D supported (got ranks %d and %d)", a.Rank(), b.Rank())
	}
	if !a.IsContiguous() || a.Offset() != 0 {
		var err error
		a, err = a.Contiguous()
		if err != nil {
			return err
		}
	}
	ord := out.Device().Ordinal()
	if err := cudart.SetDevice(ord); err != nil {
		return err
	}
	h, err := cudart.GetBlasHandle(ord)
	if err != nil {
		return err
	}
	m, k := a.Dim(0), a.Dim(1)
	n := b.Dim(1)
	// Detect: b is a transpose-view of a row-major contiguous (N, K) tensor.
	// In that case b.shape == (K, N) but b.strides == (1, K) and offset == 0,
	// pointing at the same storage as the original (N, K). Use cuBLAS's
	// transB flag to consume that storage directly — avoids materialising a
	// fresh K×N copy of the weight on every nn.Linear forward, which for a
	// 3B model's LM head was 1.2 GB × ~250 forwards per prefill (OOM at 16 GB).
	if b.Offset() == 0 && b.Strides().Get(0) == 1 && b.Strides().Get(1) == k {
		if err := cudart.SgemmRowMajorTransB(h, m, n, k,
			a.Storage().CUDAPtr(),
			b.Storage().CUDAPtr(),
			out.Storage().CUDAPtr()); err != nil {
			return err
		}
		return nil
	}
	if !b.IsContiguous() || b.Offset() != 0 {
		var err error
		b, err = b.Contiguous()
		if err != nil {
			return err
		}
	}
	if err := cudart.SgemmRowMajor(h, m, n, k,
		a.Storage().CUDAPtr(),
		b.Storage().CUDAPtr(),
		out.Storage().CUDAPtr()); err != nil {
		return err
	}
	return nil
}

func matmulF64(ctx *registry.Context) error {
	a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	a, b, err := matmulPrep(a, b)
	if err != nil {
		return err
	}
	ord := out.Device().Ordinal()
	if err := cudart.SetDevice(ord); err != nil {
		return err
	}
	h, err := cudart.GetBlasHandle(ord)
	if err != nil {
		return err
	}
	m, k := a.Dim(0), a.Dim(1)
	n := b.Dim(1)
	if err := cudart.DgemmRowMajor(h, m, n, k,
		a.Storage().CUDAPtr(),
		b.Storage().CUDAPtr(),
		out.Storage().CUDAPtr()); err != nil {
		return err
	}
	return nil
}

// matmulPrep validates rank and contiguifies non-contiguous operands in place
// (returning fresh Tensors on the same device). Any subsequent cuBLAS call
// can assume rank-2 contig inputs.
func matmulPrep(a, b *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	if a.Rank() != 2 || b.Rank() != 2 {
		return nil, nil, fmt.Errorf("cuda matmul: only 2-D supported (got ranks %d and %d)", a.Rank(), b.Rank())
	}
	var err error
	if !a.IsContiguous() || a.Offset() != 0 {
		a, err = a.Contiguous()
		if err != nil {
			return nil, nil, err
		}
	}
	if !b.IsContiguous() || b.Offset() != 0 {
		b, err = b.Contiguous()
		if err != nil {
			return nil, nil, err
		}
	}
	return a, b, nil
}

// ---- F16 / BF16 GEMMs and the FP8 cuBLASLt path (2026-10-28) ----

func init() {
	registry.Register(registry.Key{Op: "matmul", Dev: device.KindCUDA, DType: dtype.F16}, gemmExLauncher(cudart.DataF16))
	registry.Register(registry.Key{Op: "matmul", Dev: device.KindCUDA, DType: dtype.BF16}, gemmExLauncher(cudart.DataBF16))
	registry.Register(registry.Key{Op: "matmul_batched", Dev: device.KindCUDA, DType: dtype.F16}, batchedGemmExLauncher(cudart.DataF16))
	// F16×F16→F16 under the mixed key so nn.Linear / ops.MatMulMixed treat
	// F16 weights exactly like BF16 ones.
	registry.Register(registry.Key{Op: "matmul_mixed", Dev: device.KindCUDA, DType: dtype.F16}, gemmExLauncher(cudart.DataF16))
	for _, dt := range []dtype.DType{dtype.F32, dtype.BF16, dtype.F16} {
		registry.Register(registry.Key{Op: "matmul_f8", Dev: device.KindCUDA, DType: dt}, matmulF8Launcher)
	}
}

func blasDataType(dt dtype.DType) (cudart.DataType, error) {
	switch dt {
	case dtype.F16:
		return cudart.DataF16, nil
	case dtype.BF16:
		return cudart.DataBF16, nil
	case dtype.F32:
		return cudart.DataF32, nil
	}
	return 0, fmt.Errorf("cuda gemm: dtype %v has no cuBLAS GemmEx type", dt)
}

// gemmEx2D runs C = A @ B for 2-D row-major operands at the given cuBLAS
// types, using the transB flag when b is a transpose-view of a contiguous
// (N, K) tensor (nn.Linear's weight) and contiguifying it otherwise.
func gemmEx2D(a, b, out *tensor.Tensor, aType, bType, cType cudart.DataType) error {
	if a.Rank() != 2 || b.Rank() != 2 {
		return fmt.Errorf("cuda gemm: only 2-D supported (got ranks %d and %d)", a.Rank(), b.Rank())
	}
	var err error
	if !a.IsContiguous() || a.Offset() != 0 {
		if a, err = a.Contiguous(); err != nil {
			return err
		}
	}
	ord := out.Device().Ordinal()
	if err := cudart.SetDevice(ord); err != nil {
		return err
	}
	h, err := cudart.GetBlasHandle(ord)
	if err != nil {
		return err
	}
	m, k := a.Dim(0), a.Dim(1)
	n := b.Dim(1)
	transB := b.Offset() == 0 && b.Strides().Get(0) == 1 && b.Strides().Get(1) == k
	if !transB && (!b.IsContiguous() || b.Offset() != 0) {
		if b, err = b.Contiguous(); err != nil {
			return err
		}
	}
	if err := cudart.GemmExRowMajor(h, m, n, k,
		a.Storage().CUDAPtr(), b.Storage().CUDAPtr(), out.Storage().CUDAPtr(),
		aType, bType, cType, transB); err != nil {
		return err
	}
	return nil
}

func gemmExLauncher(dt cudart.DataType) registry.Kernel {
	return func(ctx *registry.Context) error {
		return gemmEx2D(ctx.Inputs[0], ctx.Inputs[1], ctx.Output, dt, dt, dt)
	}
}

func batchedGemmExLauncher(dt cudart.DataType) registry.Kernel {
	return func(ctx *registry.Context) error {
		a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
		a, b, err := batchedMatmulPrep(a, b)
		if err != nil {
			return err
		}
		ord := out.Device().Ordinal()
		if err := cudart.SetDevice(ord); err != nil {
			return err
		}
		h, err := cudart.GetBlasHandle(ord)
		if err != nil {
			return err
		}
		if err := cudart.GemmStridedBatchedExRowMajor(h, a.Dim(1), b.Dim(2), a.Dim(2),
			a.Storage().CUDAPtr(), b.Storage().CUDAPtr(), out.Storage().CUDAPtr(), dt, a.Dim(0)); err != nil {
			return err
		}
		return nil
	}
}

// matmulF8Launcher: Inputs = [a F8 (M, K), b F8 (K, N) view, scales F32 (2,)
// = {aScale, bScale} on device]; out at the registry dtype. cuBLASLt's FP8
// GEMM wants the weight as a row-major (N, K) block, which is what
// nn.Linear's transpose view already is; any other b layout is transposed
// into a fresh (N, K) copy first.
func matmulF8Launcher(ctx *registry.Context) error {
	a, b, scales, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	cType, err := blasDataType(out.DType())
	if err != nil {
		return err
	}
	if !a.IsContiguous() || a.Offset() != 0 {
		if a, err = a.Contiguous(); err != nil {
			return err
		}
	}
	m, k := a.Dim(0), a.Dim(1)
	n := b.Dim(1)
	if !(b.Offset() == 0 && b.Strides().Get(0) == 1 && b.Strides().Get(1) == k) {
		bt, err := b.Transpose(0, 1) // (N, K) view
		if err != nil {
			return err
		}
		if b, err = bt.Contiguous(); err != nil { // (N, K) row-major copy
			return err
		}
	}
	ord := out.Device().Ordinal()
	if err := cudart.SetDevice(ord); err != nil {
		return err
	}
	lt, err := cudart.GetLtHandle(ord)
	if err != nil {
		return err
	}
	sPtr := scales.Storage().CUDAPtr()
	if err := cudart.MatmulFP8RowMajor(lt, m, n, k,
		a.Storage().CUDAPtr(), b.Storage().CUDAPtr(), out.Storage().CUDAPtr(),
		sPtr, sPtr+4, cType); err != nil {
		return err
	}
	return nil
}
