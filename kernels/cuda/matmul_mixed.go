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
	// Mixed-precision matmul: F32 activation × BF16 weight → F32 output. The
	// op key uses the *output* dtype.
	registry.Register(registry.Key{Op: "matmul_mixed", Dev: device.KindCUDA, DType: dtype.F32}, matmulMixedF32BF16W)
	// Fully BF16 matmul: BF16 activation × BF16 weight → BF16 output, F32
	// accumulate. Used when activations themselves are kept in BF16
	// (ops.MatMulMixed dispatches by output dtype).
	registry.Register(registry.Key{Op: "matmul_mixed", Dev: device.KindCUDA, DType: dtype.BF16}, matmulBF16BF16W)
}

// matmulBF16BF16W is the all-BF16 nn.Linear path: a (BF16) × b (BF16, possibly
// transposed) → out (BF16). Detects b being a transpose-view of a row-major
// contig (N, K) and uses cuBLAS's transB flag, same trick as the F32 path.
func matmulBF16BF16W(ctx *registry.Context) error {
	a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	if a.Rank() != 2 || b.Rank() != 2 {
		return fmt.Errorf("cuda matmul_mixed bf16: only 2-D supported")
	}
	if a.DType() != dtype.BF16 || b.DType() != dtype.BF16 || out.DType() != dtype.BF16 {
		return fmt.Errorf("cuda matmul_mixed bf16: dtype combo unsupported (a=%v b=%v out=%v)", a.DType(), b.DType(), out.DType())
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
	if b.Offset() == 0 && b.Strides().Get(0) == 1 && b.Strides().Get(1) == k {
		if err := cudart.GemmExBF16OutBF16RowMajorTransB(h, m, n, k,
			a.Storage().CUDAPtr(),
			b.Storage().CUDAPtr(),
			out.Storage().CUDAPtr()); err != nil {
			return err
		}
		return nil
	}
	return fmt.Errorf("cuda matmul_mixed bf16: only transB layout is implemented")
}

// matmulMixedF32BF16W expects:
//
//	a (F32, contiguous, rank 2 (M, K))
//	b (BF16, rank 2 — either contiguous (K, N) or transpose-view of (N, K))
//	out (F32, (M, N))
//
// It casts `a` to BF16 in a transient device buffer and dispatches
// cublasGemmEx with BF16×BF16→F32, F32 accumulate. The transposed-weight
// fast path is preserved.
func matmulMixedF32BF16W(ctx *registry.Context) error {
	a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	if a.Rank() != 2 || b.Rank() != 2 {
		return fmt.Errorf("cuda matmul_mixed: only 2-D supported (got %d, %d)", a.Rank(), b.Rank())
	}
	if a.DType() != dtype.F32 || (b.DType() != dtype.BF16 && b.DType() != dtype.F16) || out.DType() != dtype.F32 {
		return fmt.Errorf("cuda matmul_mixed: dtype combo unsupported (a=%v b=%v out=%v)", a.DType(), b.DType(), out.DType())
	}
	if b.DType() == dtype.F16 {
		// F32 × F16 weight → F32: cast the activation to F16, GemmEx with F32 out.
		aF16, err := tensor.Zeros(dtype.F16, a.Shape().Dims(), a.Device())
		if err != nil {
			return err
		}
		if err := registryDispatch("cast", a, aF16); err != nil {
			return fmt.Errorf("cuda matmul_mixed: cast a→F16: %w", err)
		}
		return gemmEx2D(aF16, b, out, cudart.DataF16, cudart.DataF16, cudart.DataF32)
	}
	if !a.IsContiguous() || a.Offset() != 0 {
		var err error
		a, err = a.Contiguous()
		if err != nil {
			return err
		}
	}
	// Cast the activation to BF16 in a fresh device buffer. ~T·H·2 bytes —
	// tiny next to the BF16 weights we're avoiding to materialize.
	aBF16, err := tensor.Empty(dtype.BF16, a.Shape().Dims(), a.Device()) // the cast writes every element
	if err != nil {
		return err
	}
	if err := registryDispatch("cast", a, aBF16); err != nil {
		return fmt.Errorf("cuda matmul_mixed: cast a→BF16: %w", err)
	}
	ord := out.Device().Ordinal()
	if err := cudart.SetDevice(ord); err != nil {
		return err
	}
	h, err := cudart.GetBlasHandle(ord)
	if err != nil {
		return err
	}
	m, k := aBF16.Dim(0), aBF16.Dim(1)
	n := b.Dim(1)
	// Detect transB layout (`nn.Linear` y = x @ W^T where W is (N, K) row
	// contig — Transpose() returns a view with strides (1, K)). Otherwise
	// the matmul is a bare 2-D GEMM and we contiguify b first.
	if b.Offset() == 0 && b.Strides().Get(0) == 1 && b.Strides().Get(1) == k {
		if err := cudart.GemmExBF16RowMajorTransB(h, m, n, k,
			aBF16.Storage().CUDAPtr(),
			b.Storage().CUDAPtr(),
			out.Storage().CUDAPtr()); err != nil {
			return err
		}
		return nil
	}
	// Non-transB BF16×BF16→F32 isn't required by the inference paths we
	// ship today (every mixed-precision call comes from nn.Linear). Fall
	// back with a clear error so callers can add it if needed.
	return fmt.Errorf("cuda matmul_mixed: only transB layout is implemented")
}

// registryDispatch invokes a registered kernel by op name on the given
// input(s) and output. Useful for chaining ops inside a kernel — here, the
// F32→BF16 cast inside matmul_mixed.
func registryDispatch(op string, in *tensor.Tensor, out *tensor.Tensor) error {
	k, err := registry.Lookup(registry.Key{Op: op, Dev: in.Device().Kind(), DType: in.DType()})
	if err != nil {
		return err
	}
	return k(&registry.Context{Inputs: []*tensor.Tensor{in}, Output: out})
}
