package quant

import (
	"encoding/binary"
	"fmt"

	"light/dtype"
	"light/tensor"
)

// Q8_0 block layout (34 bytes, 32 elements):
//
//	uint16 d          — F16 scale
//	int8   qs[32]     — int8 quants in [-128, 127]
//
// Dequant: y[i] = d * qs[i].
const Q8_0BlockBytes = 34

// DequantQ8_0 reads a Q8_0 tensor and returns F32. CPU only.
func DequantQ8_0(src *tensor.Tensor) (*tensor.Tensor, error) {
	if src.DType() != dtype.Q8_0 {
		return nil, fmt.Errorf("DequantQ8_0: input dtype is %v, want Q8_0", src.DType())
	}
	n := src.NumElements()
	if n%QK4_0 != 0 {
		return nil, fmt.Errorf("DequantQ8_0: element count %d not a multiple of 32", n)
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
	if len(srcBuf) != numBlocks*Q8_0BlockBytes {
		return nil, fmt.Errorf("DequantQ8_0: src has %d bytes, expected %d", len(srcBuf), numBlocks*Q8_0BlockBytes)
	}
	for b := range numBlocks {
		blk := srcBuf[b*Q8_0BlockBytes : (b+1)*Q8_0BlockBytes]
		scale := halfFromUint16(binary.LittleEndian.Uint16(blk[0:2]))
		qs := blk[2:]
		base := b * QK4_0 * 4
		for i := range QK4_0 {
			q := int8(qs[i])
			putF32(outBuf[base+i*4:], scale*float32(q))
		}
	}
	return out, nil
}
