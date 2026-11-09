package ops

import (
	"errors"
	"fmt"
	"math"

	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// MatMulW8A8 is the int8 × int8 matmul for SmoothQuant-style models:
// out (M, N) F32 = x (M, K) F32 × wqᵀ, with wq (N, K) int8 and per-output-
// channel scales ws (N,) F32 (nn.Linear's (out, in) weight layout, as
// quant.QuantizeInt8PerColumn produces for a transposed weight). The
// activation is quantised symmetrically per row (absmax / 127) on the fly,
// the product accumulates in int32 and the row × column scales are applied
// once at the end. On CUDA that is cublasGemmEx with CUBLAS_COMPUTE_32I
// (tensor-core IMMA); elsewhere a Go reference with the same arithmetic.
func MatMulW8A8(x, wq, ws *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Rank() != 2 || wq.Rank() != 2 || ws.Rank() != 1 {
		return nil, fmt.Errorf("MatMulW8A8: want x (M,K), wq (N,K), ws (N,); got %v, %v, %v", x.Shape(), wq.Shape(), ws.Shape())
	}
	if x.DType() != dtype.F32 || wq.DType() != dtype.I8 || ws.DType() != dtype.F32 {
		return nil, fmt.Errorf("MatMulW8A8: dtypes must be x=F32, wq=I8, ws=F32 (got %v, %v, %v)", x.DType(), wq.DType(), ws.DType())
	}
	if !x.Device().SameAs(wq.Device()) || !x.Device().SameAs(ws.Device()) {
		return nil, errors.New("MatMulW8A8: device mismatch")
	}
	M, K := x.Dim(0), x.Dim(1)
	N := wq.Dim(0)
	if wq.Dim(1) != K || ws.Dim(0) != N {
		return nil, fmt.Errorf("MatMulW8A8: x %v, wq %v, ws %v do not agree", x.Shape(), wq.Shape(), ws.Shape())
	}
	xC, err := ensureContig(x)
	if err != nil {
		return nil, err
	}
	wC, err := ensureContig(wq)
	if err != nil {
		return nil, err
	}
	sC, err := ensureContig(ws)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Empty(dtype.F32, []int{M, N}, x.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "matmul_w8a8", Dev: x.Device().Kind(), DType: dtype.F32})
	if err != nil {
		if !errors.Is(err, registry.ErrKernelNotFound) {
			return nil, err
		}
		return matMulW8A8Reference(xC, wC, sC, out)
	}
	if err := k(&registry.Context{Inputs: []*tensor.Tensor{xC, wC, sC}, Output: out}); err != nil {
		return nil, err
	}
	return out, nil
}

// QuantizeRowsInt8 is the activation quantiser MatMulW8A8 applies: per-row
// symmetric absmax, scale = max|x| / 127, q = clamp(round(x / scale)). CPU
// tensors; exported for tests and for callers that quantise once and reuse.
func QuantizeRowsInt8(x []float32, M, K int) (q []int8, scales []float32) {
	q = make([]int8, M*K)
	scales = make([]float32, M)
	for m := 0; m < M; m++ {
		row := x[m*K : (m+1)*K]
		var amax float32
		for _, v := range row {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax = a
			}
		}
		scale := amax / 127
		var inv float32
		if scale > 0 {
			inv = 1 / scale
		}
		scales[m] = scale
		for k, v := range row {
			r := float32(math.RoundToEven(float64(v * inv)))
			if r > 127 {
				r = 127
			} else if r < -127 {
				r = -127
			}
			q[m*K+k] = int8(r)
		}
	}
	return q, scales
}

func matMulW8A8Reference(x, wq, ws, out *tensor.Tensor) (*tensor.Tensor, error) {
	M, K := x.Dim(0), x.Dim(1)
	N := wq.Dim(0)
	xq, xs := QuantizeRowsInt8(tensor.MustSlice[float32](x), M, K)
	w := tensor.MustSlice[int8](wq)
	s := tensor.MustSlice[float32](ws)
	o := make([]float32, M*N)
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var acc int32
			xr, wr := xq[m*K:(m+1)*K], w[n*K:(n+1)*K]
			for k := range xr {
				acc += int32(xr[k]) * int32(wr[k])
			}
			o[m*N+n] = float32(acc) * xs[m] * s[n]
		}
	}
	res, err := tensor.New(o, []int{M, N}, out.Device())
	if err != nil {
		return nil, err
	}
	return res, nil
}
