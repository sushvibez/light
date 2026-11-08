package quant

import (
	"errors"
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// CUDADequantFunc dequantizes a GPU-resident block-quantized tensor into a
// preallocated F32 tensor on the same device. The implementation is
// registered from the CUDA build (kernels/cuda); the quant package holds only
// the hook to keep its build independent of CUDA.
type CUDADequantFunc func(src, dst *tensor.Tensor) error

var cudaDequantHook CUDADequantFunc

// RegisterCUDADequant wires the CUDA-backed dequant dispatcher. Called from
// kernels/cuda init() under the `cuda` build tag.
func RegisterCUDADequant(fn CUDADequantFunc) { cudaDequantHook = fn }

// dispatchCUDA runs the registered hook if the source tensor lives on CUDA.
// Returns (nil, false, nil) when the source is CPU (caller keeps its CPU
// path). Returns (out, true, nil) on success, or (nil, true, err) on failure.
func dispatchCUDA(src *tensor.Tensor) (*tensor.Tensor, bool, error) {
	if device.IsWebGPU(src.Device()) {
		return dispatchRegistry(src)
	}
	if !device.IsCUDA(src.Device()) {
		return nil, false, nil
	}
	if cudaDequantHook == nil {
		return nil, true, errors.New("quant: CUDA dequant hook not registered (build with -tags cuda)")
	}
	out, err := tensor.Zeros(dtype.F32, src.Shape().Dims(), src.Device())
	if err != nil {
		return nil, true, err
	}
	if err := cudaDequantHook(src, out); err != nil {
		return nil, true, err
	}
	return out, true, nil
}

// dispatchRegistry dequantizes on a device whose kernels register a
// "dequant" op per format (WebGPU), reporting handled = true either way so
// the host-byte fallback never runs on device memory.
func dispatchRegistry(src *tensor.Tensor) (*tensor.Tensor, bool, error) {
	k, err := registry.Lookup(registry.Key{Op: "dequant", Dev: src.Device().Kind(), DType: src.DType()})
	if err != nil {
		return nil, true, fmt.Errorf("quant: no %v dequant kernel on %v: %w", src.DType(), src.Device(), err)
	}
	out, err := tensor.Zeros(dtype.F32, src.Shape().Dims(), src.Device())
	if err != nil {
		return nil, true, err
	}
	if err := k(&registry.Context{Inputs: []*tensor.Tensor{src}, Output: out}); err != nil {
		return nil, true, err
	}
	return out, true, nil
}
