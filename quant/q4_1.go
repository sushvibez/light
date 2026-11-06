package quant

import (
	"encoding/binary"
	"fmt"

	"light/dtype"
	"light/tensor"
)

// Q4_1 block layout (20 bytes, 32 elements):
//
//	uint16 d          — F16 delta
//	uint16 m          — F16 min
//	uint8  qs[16]     — unsigned 4-bit nibbles (low = element j, high = element j+16)
//
// Dequant: y[i] = d * q[i] + m, where q[i] is 0..15 unsigned.
const Q4_1BlockBytes = 20

// DequantQ4_1 reads a Q4_1 tensor and returns F32. CPU only.
func DequantQ4_1(src *tensor.Tensor) (*tensor.Tensor, error) {
	if src.DType() != dtype.Q4_1 {
		return nil, fmt.Errorf("DequantQ4_1: input dtype is %v, want Q4_1", src.DType())
	}
	n := src.NumElements()
	if n%QK4_0 != 0 {
		return nil, fmt.Errorf("DequantQ4_1: element count %d not a multiple of 32", n)
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
	if len(srcBuf) != numBlocks*Q4_1BlockBytes {
		return nil, fmt.Errorf("DequantQ4_1: src has %d bytes, expected %d", len(srcBuf), numBlocks*Q4_1BlockBytes)
	}
	for b := range numBlocks {
		blk := srcBuf[b*Q4_1BlockBytes : (b+1)*Q4_1BlockBytes]
		delta := halfFromUint16(binary.LittleEndian.Uint16(blk[0:2]))
		minV := halfFromUint16(binary.LittleEndian.Uint16(blk[2:4]))
		qs := blk[4:]
		base := b * QK4_0 * 4
		for j := range 16 {
			lo := int(qs[j] & 0x0F)
			hi := int(qs[j] >> 4)
			putF32(outBuf[base+(j)*4:], delta*float32(lo)+minV)
			putF32(outBuf[base+(j+16)*4:], delta*float32(hi)+minV)
		}
	}
	return out, nil
}
