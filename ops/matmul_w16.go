package ops

import (
	"errors"
	"fmt"

	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// MatMulW16 multiplies F32 activations by a 16-bit weight in nn.Linear's
// layout without rounding the activations:
//
//	out[m, n] = Σ_k x[m, k] · f32(w[n, k])
//
//	x: F32 (M, K)
//	w: BF16 or F16 (N, K), row-major (the stored Linear weight)
//	returns F32 (M, N)
//
// Widening a 16-bit value to F32 is exact, so for a checkpoint stored in
// that dtype this equals the F32 model's matmul while keeping the weight in
// half the memory (WebGPU's 16-bit weight storage). Devices without a
// "matmul_w16" kernel widen the weight and run the F32 matmul.
func MatMulW16(x, w *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Rank() != 2 || w.Rank() != 2 || x.Dim(1) != w.Dim(1) {
		return nil, fmt.Errorf("MatMulW16: want x (M, K) and w (N, K), got %v, %v", x.Shape(), w.Shape())
	}
	if x.DType() != dtype.F32 || (w.DType() != dtype.BF16 && w.DType() != dtype.F16) {
		return nil, fmt.Errorf("MatMulW16: want F32 x and BF16 / F16 w, got %v, %v", x.DType(), w.DType())
	}
	if !x.Device().SameAs(w.Device()) {
		return nil, fmt.Errorf("MatMulW16: x on %v, w on %v", x.Device(), w.Device())
	}
	xc, err := ensureContig(x)
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "matmul_w16", Dev: x.Device().Kind(), DType: w.DType()})
	if err != nil {
		if !errors.Is(err, registry.ErrKernelNotFound) {
			return nil, err
		}
		w32, err := Cast(w, dtype.F32)
		if err != nil {
			return nil, err
		}
		wT, err := w32.Transpose(0, 1)
		if err != nil {
			return nil, err
		}
		return MatMul(xc, wT)
	}
	wc, err := ensureContig(w)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(dtype.F32, []int{x.Dim(0), w.Dim(0)}, x.Device())
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{Inputs: []*tensor.Tensor{xc, wc}, Output: out}); err != nil {
		return nil, err
	}
	return out, nil
}
