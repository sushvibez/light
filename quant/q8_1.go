package quant

import (
	"encoding/binary"
	"fmt"

	"light/dtype"
	"light/tensor"
)

// Q8_1 block layout (36 bytes, 32 elements):
//
//	uint16 d      — F16 delta (scale)
//	uint16 s      — F16 sum (d * sum(q[0..31])); used by gemm to avoid a full re-summation
//	int8   qs[32] — int8 quants
//
// Dequant: y[i] = d * q[i]. (The `s` field is metadata for matmul; ignored here.)
const Q8_1BlockBytes = 36

// DequantQ8_1 reads a Q8_1 tensor and returns F32. CPU only.
func DequantQ8_1(src *tensor.Tensor) (*tensor.Tensor, error) {
	if src.DType() != dtype.Q8_1 {
		return nil, fmt.Errorf("DequantQ8_1: input dtype is %v, want Q8_1", src.DType())
	}
	n := src.NumElements()
	if n%QK4_0 != 0 {
		return nil, fmt.Errorf("DequantQ8_1: element count %d not a multiple of 32", n)
	}
	if out, handled, err := dispatchCUDA(src); handled {
		return out, err
	}
	src, err := src.Contiguous()
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(dtype.F32, src.Shape().Dims(), src.Device())
	if err != nil {
		return nil, err
	}
	srcBuf := src.Storage().CPUBytes()
	outBuf := out.Storage().CPUBytes()
	numBlocks := n / QK4_0
	if len(srcBuf) != numBlocks*Q8_1BlockBytes {
		return nil, fmt.Errorf("DequantQ8_1: src has %d bytes, expected %d", len(srcBuf), numBlocks*Q8_1BlockBytes)
	}
	for b := range numBlocks {
		blk := srcBuf[b*Q8_1BlockBytes : (b+1)*Q8_1BlockBytes]
		delta := halfFromUint16(binary.LittleEndian.Uint16(blk[0:2]))
		// blk[2:4] is the sum metadata — ignored for pure dequant.
		qs := blk[4:]
		base := b * QK4_0 * 4
		for i := range QK4_0 {
			q := int8(qs[i])
			putF32(outBuf[base+i*4:], delta*float32(q))
		}
	}
	return out, nil
}
