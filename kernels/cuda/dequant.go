//go:build !nocuda && cgo

package cuda

import (
	"fmt"
	"unsafe"

	"light/dtype"
	"light/internal/cudart"
	"light/quant"
	"light/tensor"
)

func init() {
	quant.RegisterCUDADequant(launchDequant)
}

// dequantKernelName maps src dtype to the corresponding CUDA kernel name.
// Kept in this file so a new quant format is a one-line addition here plus
// a new kernel in src/dequant.cu.
var dequantKernelName = map[dtype.DType]string{
	dtype.Q4_0: "dequant_q4_0_f32",
	dtype.Q4_1: "dequant_q4_1_f32",
	dtype.Q5_0: "dequant_q5_0_f32",
	dtype.Q5_1: "dequant_q5_1_f32",
	dtype.Q8_0: "dequant_q8_0_f32",
	dtype.Q8_1: "dequant_q8_1_f32",
	dtype.Q2_K: "dequant_q2_k_f32",
	dtype.Q3_K: "dequant_q3_k_f32",
	dtype.Q4_K: "dequant_q4_k_f32",
	dtype.Q5_K: "dequant_q5_k_f32",
	dtype.Q6_K: "dequant_q6_k_f32",
	dtype.Q8_K: "dequant_q8_k_f32",
}

// launchDequant is registered as quant.CUDADequantFunc. It dispatches by
// src dtype to the appropriate dequant kernel.
func launchDequant(src, dst *tensor.Tensor) error {
	if dst.DType() != dtype.F32 {
		return fmt.Errorf("cuda dequant: dst dtype is %v, want F32", dst.DType())
	}
	kname, ok := dequantKernelName[src.DType()]
	if !ok {
		return fmt.Errorf("cuda dequant: unsupported src dtype %v", src.DType())
	}
	if !dst.IsContiguous() || dst.Offset() != 0 {
		return fmt.Errorf("cuda dequant: dst must be contiguous with zero offset")
	}
	// Source tensor uses quantized storage (byte-sized layout). The hook
	// callers guarantee the element count is a multiple of 32.
	src, err := src.Contiguous()
	if err != nil {
		return err
	}
	n := uint64(src.NumElements())
	if err := cudart.SetDevice(dst.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel(kname)
	if err != nil {
		return err
	}
	srcPtr := src.Storage().CUDAPtr()
	dstPtr := dst.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&srcPtr),
		unsafe.Pointer(&dstPtr),
		unsafe.Pointer(&n),
	}
	if err := Launch1D(fn, n, args); err != nil {
		return err
	}
	return nil
}
