package ops

import (
	"errors"
	"fmt"
	"light/device"

	"light/dtype"
	"light/internal/registry"
	"light/quant"
	"light/tensor"
)

// MatMulQ multiplies F32 activations by a GGUF block-quantized weight held
// in Linear layout, dequantizing on the fly:
//
//	out[m, n] = Σ_k x[m, k] * dequant(w)[n, k]
//
//	x:   F32 (M, K)
//	w:   quantized (N, K) — Q4_0 / Q5_0 / Q8_0 / Q4_K / Q5_K / Q6_K have fused
//	     CUDA kernels; every other GGUF type (and the CPU device) falls back
//	     to quant.Dequantize + MatMul, which costs a full F32 copy of w.
//	returns F32 (M, N)
//
// K must be a multiple of the format's block size (32 or 256), which GGUF
// guarantees for the weight's inner dimension.
func MatMulQ(x, w *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Rank() != 2 || w.Rank() != 2 {
		return nil, fmt.Errorf("MatMulQ: want x (M,K) and w (N,K); got %v, %v", x.Shape(), w.Shape())
	}
	if x.DType() != dtype.F32 {
		return nil, fmt.Errorf("MatMulQ: x must be F32, got %v", x.DType())
	}
	if !w.DType().IsQuantized() {
		return nil, fmt.Errorf("MatMulQ: w must be a quantized dtype, got %v", w.DType())
	}
	if !x.Device().SameAs(w.Device()) {
		return nil, fmt.Errorf("MatMulQ: x on %v, w on %v", x.Device(), w.Device())
	}
	M, K := x.Dim(0), x.Dim(1)
	N := w.Dim(0)
	if w.Dim(1) != K {
		return nil, fmt.Errorf("MatMulQ: inner dim mismatch (x K=%d, w K=%d)", K, w.Dim(1))
	}
	if K%w.DType().BlockElems() != 0 {
		return nil, fmt.Errorf("MatMulQ: K=%d not a multiple of %v block size %d", K, w.DType(), w.DType().BlockElems())
	}
	xC, err := ensureContig(x)
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "matmul_q", Dev: x.Device().Kind(), DType: w.DType()})
	if err != nil {
		if !errors.Is(err, registry.ErrKernelNotFound) {
			return nil, err
		}
		return matMulQFallback(xC, w)
	}
	out, err := tensor.Zeros(dtype.F32, []int{M, N}, x.Device())
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{Inputs: []*tensor.Tensor{xC, w}, Output: out}); err != nil {
		return nil, err
	}
	return out, nil
}

// matMulQFallback dequantizes w to F32 and runs the ordinary matmul. Exact
// up to summation order; used on CPU and for formats without a fused kernel.
func matMulQFallback(x, w *tensor.Tensor) (*tensor.Tensor, error) {
	wf, err := quant.Dequantize(w)
	if err != nil {
		return nil, err
	}
	wT, err := wf.Transpose(0, 1)
	if err != nil {
		return nil, err
	}
	return MatMul(x, wT)
}

// MatMulQFused reports whether a fused dequant-matmul kernel exists on CUDA
// for the format, i.e. whether a weight in that format can stay quantized
// at no per-call dequantization cost.
func MatMulQFused(dt dtype.DType) bool { return MatMulQFusedOn(device.KindCUDA, dt) }

// MatMulQFusedOn is MatMulQFused for any device kind (CUDA, WebGPU).
func MatMulQFusedOn(kind device.Kind, dt dtype.DType) bool {
	_, err := registry.Lookup(registry.Key{Op: "matmul_q", Dev: kind, DType: dt})
	return err == nil
}
