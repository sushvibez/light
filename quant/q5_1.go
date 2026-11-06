package quant

import (
	"encoding/binary"
	"fmt"

	"light/dtype"
	"light/tensor"
)

// Q5_1 block layout (24 bytes, 32 elements):
//
//	uint16 d             — F16 delta
//	uint16 m             — F16 min
//	uint8  qh[4]         — 32 high bits
//	uint8  qs[16]        — low 4-bit nibbles
//
// Dequant (unsigned 5-bit in [0, 31]):
//
//	y[i] = d * q5[i] + m
//	q5[i] = (qs[…] & 0x0F) | ((qh >> i) & 1) << 4
const Q5_1BlockBytes = 24

// DequantQ5_1 reads a Q5_1 tensor and returns F32. CPU only.
func DequantQ5_1(src *tensor.Tensor) (*tensor.Tensor, error) {
	if src.DType() != dtype.Q5_1 {
		return nil, fmt.Errorf("DequantQ5_1: input dtype is %v, want Q5_1", src.DType())
	}
	n := src.NumElements()
	if n%QK4_0 != 0 {
		return nil, fmt.Errorf("DequantQ5_1: element count %d not a multiple of 32", n)
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
	if len(srcBuf) != numBlocks*Q5_1BlockBytes {
		return nil, fmt.Errorf("DequantQ5_1: src has %d bytes, expected %d", len(srcBuf), numBlocks*Q5_1BlockBytes)
	}
	for b := range numBlocks {
		blk := srcBuf[b*Q5_1BlockBytes : (b+1)*Q5_1BlockBytes]
		delta := halfFromUint16(binary.LittleEndian.Uint16(blk[0:2]))
		minV := halfFromUint16(binary.LittleEndian.Uint16(blk[2:4]))
		qh := binary.LittleEndian.Uint32(blk[4:8])
		qs := blk[8:]
		base := b * QK4_0 * 4
		for j := range 16 {
			xh0 := int((qh >> j) & 1)
			xh1 := int((qh >> (j + 16)) & 1)
			lo := int(qs[j]&0x0F) | (xh0 << 4)
			hi := int(qs[j]>>4) | (xh1 << 4)
			putF32(outBuf[base+(j)*4:], delta*float32(lo)+minV)
			putF32(outBuf[base+(j+16)*4:], delta*float32(hi)+minV)
		}
	}
	return out, nil
}
