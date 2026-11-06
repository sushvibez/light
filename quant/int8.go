package quant

import (
	"fmt"
	"math"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// QuantizeInt8PerColumn quantizes an F32 weight matrix w (K, N) to symmetric
// per-output-channel INT8:
//
//	scales[n] = max_k |w[k, n]| / 127
//	q[k, n]   = round(w[k, n] / scales[n])   ∈ [-127, 127]
//
// A column that is entirely zero gets scale 1 (and all-zero codes) so the
// dequantised product stays finite. The returned q is I8 (K, N) and scales
// is F32 (N,), both on w's device; the quantisation itself runs on the host
// (a CUDA w is copied down and the results copied back).
//
// Pair the result with ops.MatMulInt8(x, q, scales).
func QuantizeInt8PerColumn(w *tensor.Tensor) (q *tensor.Tensor, scales *tensor.Tensor, err error) {
	if w.Rank() != 2 {
		return nil, nil, fmt.Errorf("quant.QuantizeInt8PerColumn: w must be (K, N), got %v", w.Shape())
	}
	if w.DType() != dtype.F32 {
		return nil, nil, fmt.Errorf("quant.QuantizeInt8PerColumn: w must be F32, got %v", w.DType())
	}
	dev := w.Device()
	host := w
	if !device.IsCPU(dev) {
		if host, err = w.To(device.CPU()); err != nil {
			return nil, nil, err
		}
	} else if !host.IsContiguous() || host.Offset() != 0 {
		if host, err = host.Contiguous(); err != nil {
			return nil, nil, err
		}
	}
	K, N := host.Dim(0), host.Dim(1)
	data, err := tensor.ToSlice[float32](host)
	if err != nil {
		return nil, nil, err
	}
	sc := make([]float32, N)
	for n := range N {
		var mx float32
		for k := range K {
			if a := float32(math.Abs(float64(data[k*N+n]))); a > mx {
				mx = a
			}
		}
		if mx == 0 {
			sc[n] = 1
		} else {
			sc[n] = mx / 127
		}
	}
	codes := make([]int8, K*N)
	for k := range K {
		for n := range N {
			r := math.Round(float64(data[k*N+n] / sc[n]))
			if r > 127 {
				r = 127
			} else if r < -127 {
				r = -127
			}
			codes[k*N+n] = int8(r)
		}
	}
	if q, err = tensor.New(codes, []int{K, N}, dev); err != nil {
		return nil, nil, err
	}
	if scales, err = tensor.New(sc, []int{N}, dev); err != nil {
		return nil, nil, err
	}
	return q, scales, nil
}

// DequantizeInt8PerColumn is the inverse of QuantizeInt8PerColumn (host
// only): returns F32 (K, N) with w[k, n] = float(q[k, n]) * scales[n].
// Useful for tests and for building a reference MatMul.
func DequantizeInt8PerColumn(q, scales *tensor.Tensor) (*tensor.Tensor, error) {
	if q.Rank() != 2 || q.DType() != dtype.I8 || scales.Rank() != 1 || scales.DType() != dtype.F32 {
		return nil, fmt.Errorf("quant.DequantizeInt8PerColumn: want q I8 (K,N) and scales F32 (N,)")
	}
	dev := q.Device()
	qh, sh := q, scales
	var err error
	if !device.IsCPU(dev) {
		if qh, err = q.To(device.CPU()); err != nil {
			return nil, err
		}
		if sh, err = scales.To(device.CPU()); err != nil {
			return nil, err
		}
	}
	K, N := qh.Dim(0), qh.Dim(1)
	codes, err := tensor.ToSlice[int8](qh)
	if err != nil {
		return nil, err
	}
	sc, err := tensor.ToSlice[float32](sh)
	if err != nil {
		return nil, err
	}
	out := make([]float32, K*N)
	for k := range K {
		for n := range N {
			out[k*N+n] = float32(codes[k*N+n]) * sc[n]
		}
	}
	return tensor.New(out, []int{K, N}, dev)
}
