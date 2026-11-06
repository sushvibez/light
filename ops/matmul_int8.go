package ops

import (
	"fmt"

	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// MatMulInt8 multiplies an F32 activation by a weight-only INT8 matrix with
// symmetric per-output-channel dequantisation:
//
//	out[m, n] = scales[n] * Σ_k x[m, k] * float(wq[k, n])
//
//	x:      F32 (M, K)
//	wq:     I8  (K, N)   — from quant.QuantizeInt8PerColumn
//	scales: F32 (N,)
//	returns F32 (M, N)
//
// The integer weights are widened to F32 inside the kernel and accumulated
// in F32, so the result equals MatMul(x, dequant(wq)) up to summation order.
func MatMulInt8(x, wq, scales *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Rank() != 2 || wq.Rank() != 2 || scales.Rank() != 1 {
		return nil, fmt.Errorf("MatMulInt8: want x (M,K), wq (K,N), scales (N,); got %v, %v, %v", x.Shape(), wq.Shape(), scales.Shape())
	}
	if x.DType() != dtype.F32 || wq.DType() != dtype.I8 || scales.DType() != dtype.F32 {
		return nil, fmt.Errorf("MatMulInt8: dtypes must be x=F32, wq=I8, scales=F32 (got %v, %v, %v)", x.DType(), wq.DType(), scales.DType())
	}
	if !x.Device().SameAs(wq.Device()) || !x.Device().SameAs(scales.Device()) {
		return nil, fmt.Errorf("MatMulInt8: device mismatch")
	}
	M, K := x.Dim(0), x.Dim(1)
	N := wq.Dim(1)
	if wq.Dim(0) != K {
		return nil, fmt.Errorf("MatMulInt8: inner dim mismatch (%d vs %d)", K, wq.Dim(0))
	}
	if scales.Dim(0) != N {
		return nil, fmt.Errorf("MatMulInt8: scales must have %d entries, got %d", N, scales.Dim(0))
	}
	xC, err := ensureContig(x)
	if err != nil {
		return nil, err
	}
	wC, err := ensureContig(wq)
	if err != nil {
		return nil, err
	}
	sC, err := ensureContig(scales)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(dtype.F32, []int{M, N}, x.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "matmul_int8", Dev: x.Device().Kind(), DType: dtype.F32})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{Inputs: []*tensor.Tensor{xC, wC, sC}, Output: out}); err != nil {
		return nil, err
	}
	return out, nil
}
