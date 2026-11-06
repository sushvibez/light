package ops

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// MatMulMixed performs a 2-D matmul where activation `a` and weight `b` may
// be at different storage dtypes:
//
//   - a=F32, b=BF16 → out=F32  (mixed-precision; a is cast to BF16 inside)
//   - a=BF16, b=BF16 → out=BF16 (fully BF16 activations + weights)
//   - a=F32, b=F16  → out=F32;  a=F16, b=F16 → out=F16 (same, F16 weights)
//
// Only rank-2 supported; both operands must share device. The output dtype
// drives kernel registry dispatch (`matmul_mixed`/F32 or `matmul_mixed`/BF16).
func MatMulMixed(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	if !a.Device().SameAs(b.Device()) {
		return nil, fmt.Errorf("matmul_mixed: device mismatch")
	}
	if a.Rank() != 2 || b.Rank() != 2 {
		return nil, fmt.Errorf("matmul_mixed: only rank-2 supported (got %d, %d)", a.Rank(), b.Rank())
	}
	if a.Dim(1) != b.Dim(0) {
		return nil, fmt.Errorf("matmul_mixed: inner dim mismatch (%d vs %d)", a.Dim(1), b.Dim(0))
	}
	var outDT dtype.DType
	switch {
	case a.DType() == dtype.F32 && b.DType() == dtype.BF16:
		outDT = dtype.F32
	case a.DType() == dtype.BF16 && b.DType() == dtype.BF16:
		outDT = dtype.BF16
	case a.DType() == dtype.F32 && b.DType() == dtype.F16:
		outDT = dtype.F32
	case a.DType() == dtype.F16 && b.DType() == dtype.F16:
		outDT = dtype.F16
	default:
		return nil, fmt.Errorf("matmul_mixed: unsupported dtype combo (a=%v, b=%v)", a.DType(), b.DType())
	}
	out, err := tensor.Zeros(outDT, []int{a.Dim(0), b.Dim(1)}, a.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "matmul_mixed", Dev: a.Device().Kind(), DType: outDT})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{Inputs: []*tensor.Tensor{a, b}, Output: out}); err != nil {
		return nil, err
	}
	return out, nil
}

// MatMul performs matrix multiplication. Handles both 2-D
// `(M,K) × (K,N) → (M,N)` and batched 3-D `(B,M,K) × (B,K,N) → (B,M,N)`
// inputs. Inputs must share dtype and device.
//
// On CUDA, 2-D dispatches to cuBLAS Sgemm / Dgemm, and 3-D dispatches to
// cuBLAS `SgemmStridedBatched` (F32 only today; add F64 when needed). CPU
// paths are naive loops.
func MatMul(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	if !a.Device().SameAs(b.Device()) {
		return nil, fmt.Errorf("matmul: device mismatch")
	}
	if a.DType() != b.DType() {
		return nil, fmt.Errorf("matmul: dtype mismatch")
	}
	if a.Rank() == 2 && b.Rank() == 2 {
		if a.Dim(1) != b.Dim(0) {
			return nil, fmt.Errorf("matmul: inner dim mismatch (%d vs %d)", a.Dim(1), b.Dim(0))
		}
		out, err := tensor.Zeros(a.DType(), []int{a.Dim(0), b.Dim(1)}, a.Device())
		if err != nil {
			return nil, err
		}
		if err := dispatch("matmul", a.Device(), a.DType(), []*tensor.Tensor{a, b}, out); err != nil {
			return nil, err
		}
		return out, nil
	}
	if a.Rank() == 3 && b.Rank() == 3 {
		if a.Dim(0) != b.Dim(0) {
			return nil, fmt.Errorf("matmul: batch dim mismatch (%d vs %d)", a.Dim(0), b.Dim(0))
		}
		if a.Dim(2) != b.Dim(1) {
			return nil, fmt.Errorf("matmul: inner dim mismatch (%d vs %d)", a.Dim(2), b.Dim(1))
		}
		out, err := tensor.Zeros(a.DType(), []int{a.Dim(0), a.Dim(1), b.Dim(2)}, a.Device())
		if err != nil {
			return nil, err
		}
		if err := dispatch("matmul_batched", a.Device(), a.DType(), []*tensor.Tensor{a, b}, out); err != nil {
			return nil, err
		}
		return out, nil
	}
	return nil, fmt.Errorf("matmul: expected rank 2 or 3 on both sides, got %d and %d", a.Rank(), b.Rank())
}

// MatMulF8 is the FP8 (E4M3) matmul: a (M, K) and b (K, N) hold F8E4M3
// codes and the result is aScale · bScale · (a @ b) at outDT (F32, BF16 or
// F16), with F32 accumulation. On CUDA it runs cuBLASLt's FP8 tensor-core
// GEMM (sm_89+; K must be a multiple of 16); the CPU kernel is the exact
// reference. b is normally nn.Linear's transposed weight view (an (N, K)
// row-major block), which is the layout the GPU kernel consumes directly.
func MatMulF8(a, b *tensor.Tensor, aScale, bScale float32, outDT dtype.DType) (*tensor.Tensor, error) {
	if !a.Device().SameAs(b.Device()) {
		return nil, fmt.Errorf("matmul_f8: device mismatch")
	}
	if a.DType() != dtype.F8E4M3 || b.DType() != dtype.F8E4M3 {
		return nil, fmt.Errorf("matmul_f8: operands must be F8E4M3 (got %v, %v)", a.DType(), b.DType())
	}
	if outDT != dtype.F32 && outDT != dtype.BF16 && outDT != dtype.F16 {
		return nil, fmt.Errorf("matmul_f8: output dtype must be F32/BF16/F16, got %v", outDT)
	}
	if a.Rank() != 2 || b.Rank() != 2 {
		return nil, fmt.Errorf("matmul_f8: only rank-2 supported (got %d, %d)", a.Rank(), b.Rank())
	}
	if a.Dim(1) != b.Dim(0) {
		return nil, fmt.Errorf("matmul_f8: inner dim mismatch (%d vs %d)", a.Dim(1), b.Dim(0))
	}
	scales, err := tensor.New([]float32{aScale, bScale}, []int{2}, a.Device())
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(outDT, []int{a.Dim(0), b.Dim(1)}, a.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "matmul_f8", Dev: a.Device().Kind(), DType: outDT})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{a, b, scales},
		Output: out,
		Attrs:  map[string]any{"a_scale": float64(aScale), "b_scale": float64(bScale)},
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// QuantizeF8 converts a float tensor to F8E4M3 with a per-tensor scale:
// q = round(x / scale), scale = max|x| / 448 (1 when x is all zeros), so
// that x ≈ q · scale. Returns the F8 tensor and the scale; feed both to
// MatMulF8. The amax reduction runs on x's device; only the scalar comes
// back to the host.
func QuantizeF8(x *tensor.Tensor) (*tensor.Tensor, float32, error) {
	f32 := x
	var err error
	if x.DType() != dtype.F32 {
		if f32, err = Cast(x, dtype.F32); err != nil {
			return nil, 0, err
		}
	}
	ab, err := Abs(f32)
	if err != nil {
		return nil, 0, err
	}
	mx, err := Max(ab, ReduceOptions{})
	if err != nil {
		return nil, 0, err
	}
	mxCPU, err := mx.To(device.CPU())
	if err != nil {
		return nil, 0, err
	}
	amax := tensor.MustSlice[float32](mxCPU)[0]
	scale := float32(1)
	if amax > 0 {
		scale = amax / 448
	}
	scaled, err := Affine(f32, float64(1/scale), 0)
	if err != nil {
		return nil, 0, err
	}
	q, err := Cast(scaled, dtype.F8E4M3)
	if err != nil {
		return nil, 0, err
	}
	return q, scale, nil
}
