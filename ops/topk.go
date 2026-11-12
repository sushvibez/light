package ops

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/tensor"
	"unsafe"
)

// TopK returns the top `k` values and their indices along the last axis of a.
// If `largest` is true, returns the k largest; otherwise the k smallest.
// Output values have the same dtype as a; indices are I64.
//
// Output shapes: values (..., k), indices (..., k) where ... = a.shape[:-1].
func TopK(a *tensor.Tensor, k int, largest bool) (*tensor.Tensor, *tensor.Tensor, error) {
	if a.Rank() < 1 {
		return nil, nil, fmt.Errorf("TopK: rank must be >= 1")
	}
	last := a.Dim(-1)
	if k < 0 || k > last {
		return nil, nil, fmt.Errorf("TopK: k=%d out of range [0, %d]", k, last)
	}
	a, err := ensureContig(a)
	if err != nil {
		return nil, nil, err
	}
	if device.IsCUDA(a.Device()) && a.DType() == dtype.F32 && k <= cudaTopKMaxFused {
		// Single-block kernel: the k best entries per row without sorting
		// the row (a sampler's top-k over a 250k vocabulary).
		outDims := append([]int{}, a.Shape().Dims()...)
		outDims[len(outDims)-1] = k
		vals, err := tensor.Empty(dtype.F32, outDims, a.Device())
		if err != nil {
			return nil, nil, err
		}
		idx, err := tensor.Empty(dtype.I64, outDims, a.Device())
		if err != nil {
			return nil, nil, err
		}
		kernelFn, err := registry.Lookup(registry.Key{Op: "topk", Dev: device.KindCUDA, DType: dtype.F32})
		if err != nil {
			return nil, nil, err
		}
		if err := kernelFn(&registry.Context{Inputs: []*tensor.Tensor{a}, Output: vals,
			Attrs: map[string]any{"k": k, "largest": largest, "idx": idx}}); err != nil {
			return nil, nil, err
		}
		return vals, idx, nil
	}
	if device.IsCUDA(a.Device()) {
		// Device path: full row sort (bitonic kernel) then narrow to k.
		vals, idx, err := Sort(a, largest)
		if err != nil {
			return nil, nil, err
		}
		v, err := vals.Narrow(-1, 0, k)
		if err != nil {
			return nil, nil, err
		}
		i, err := idx.Narrow(-1, 0, k)
		if err != nil {
			return nil, nil, err
		}
		if v, err = v.Contiguous(); err != nil {
			return nil, nil, err
		}
		if i, err = i.Contiguous(); err != nil {
			return nil, nil, err
		}
		return v, i, nil
	}
	outDims := append([]int{}, a.Shape().Dims()...)
	outDims[len(outDims)-1] = k
	vals, err := tensor.Zeros(a.DType(), outDims, a.Device())
	if err != nil {
		return nil, nil, err
	}
	idx, err := tensor.Zeros(dtype.I64, outDims, a.Device())
	if err != nil {
		return nil, nil, err
	}
	// Pass the I64 index slice view to the kernel via Attrs — kernel kinds
	// that take a second output are still rare; we use an ad-hoc pointer.
	idxBuf := idx.Storage().CPUBytes()
	idxVals := unsafe.Slice((*int64)(unsafe.Pointer(&idxBuf[0])), idx.NumElements())
	kernelFn, err := registry.Lookup(registry.Key{Op: "topk", Dev: a.Device().Kind(), DType: a.DType()})
	if err != nil {
		return nil, nil, err
	}
	if err := kernelFn(&registry.Context{
		Inputs: []*tensor.Tensor{a},
		Output: vals,
		Attrs: map[string]any{
			"k":       k,
			"largest": largest,
			"idx_out": &struct {
				Values  []int64
				Columns int
			}{Values: idxVals, Columns: k},
		},
	}); err != nil {
		return nil, nil, err
	}
	return vals, idx, nil
}

// cudaTopKMaxFused mirrors kernels/cuda.TopKMaxFused (ops cannot import the
// kernel package).
const cudaTopKMaxFused = 64
