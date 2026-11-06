package quant

import (
	"fmt"

	"light/dtype"
	"light/tensor"
)

// Dequantize converts any GGUF block-quantized tensor to F32 on the same
// device, dispatching on its dtype. CUDA sources use the registered CUDA
// dequant hook.
func Dequantize(src *tensor.Tensor) (*tensor.Tensor, error) {
	switch src.DType() {
	case dtype.Q4_0:
		return DequantQ4_0(src)
	case dtype.Q4_1:
		return DequantQ4_1(src)
	case dtype.Q5_0:
		return DequantQ5_0(src)
	case dtype.Q5_1:
		return DequantQ5_1(src)
	case dtype.Q8_0:
		return DequantQ8_0(src)
	case dtype.Q8_1:
		return DequantQ8_1(src)
	case dtype.Q2_K:
		return DequantQ2_K(src)
	case dtype.Q3_K:
		return DequantQ3_K(src)
	case dtype.Q4_K:
		return DequantQ4_K(src)
	case dtype.Q5_K:
		return DequantQ5_K(src)
	case dtype.Q6_K:
		return DequantQ6_K(src)
	case dtype.Q8_K:
		return DequantQ8_K(src)
	default:
		return nil, fmt.Errorf("quant.Dequantize: %v is not a quantized dtype", src.DType())
	}
}
