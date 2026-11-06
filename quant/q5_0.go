package quant

import (
	"encoding/binary"
	"fmt"

	"light/dtype"
	"light/tensor"
)

// Q5_0 block layout (22 bytes, 32 elements):
//
//	uint16 d             — F16 delta
//	uint8  qh[4]         — 32 high bits (1 per element)
//	uint8  qs[16]        — low 4 bits of each element (low nibble = j, high nibble = j+16)
//
// Dequant (ggml convention): signed 5-bit in [-16, 15]:
//
//	xh = ((qh[i/8] >> (i%8)) & 1) << 4
//	y[i] = d * ((q[i] | xh) - 16)
//
// where q[i] is the unsigned 4-bit low nibble for element i.
const Q5_0BlockBytes = 22

// DequantQ5_0 reads a Q5_0 tensor and returns F32. CPU only.
func DequantQ5_0(src *tensor.Tensor) (*tensor.Tensor, error) {
	if src.DType() != dtype.Q5_0 {
		return nil, fmt.Errorf("DequantQ5_0: input dtype is %v, want Q5_0", src.DType())
	}
	n := src.NumElements()
	if n%QK4_0 != 0 {
		return nil, fmt.Errorf("DequantQ5_0: element count %d not a multiple of 32", n)
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
	if len(srcBuf) != numBlocks*Q5_0BlockBytes {
		return nil, fmt.Errorf("DequantQ5_0: src has %d bytes, expected %d", len(srcBuf), numBlocks*Q5_0BlockBytes)
	}
	for b := range numBlocks {
		blk := srcBuf[b*Q5_0BlockBytes : (b+1)*Q5_0BlockBytes]
		delta := halfFromUint16(binary.LittleEndian.Uint16(blk[0:2]))
		qh := binary.LittleEndian.Uint32(blk[2:6])
		qs := blk[6:]
		base := b * QK4_0 * 4
		for j := range 16 {
			xh0 := int((qh >> j) & 1)
			xh1 := int((qh >> (j + 16)) & 1)
			lo := int(qs[j]&0x0F) | (xh0 << 4)
			hi := int(qs[j]>>4) | (xh1 << 4)
			putF32(outBuf[base+(j)*4:], delta*float32(lo-16))
			putF32(outBuf[base+(j+16)*4:], delta*float32(hi-16))
		}
	}
	return out, nil
}
