package quant

import (
	"encoding/binary"
	"fmt"

	"light/dtype"
	"light/tensor"
)

// Q5_K super-block layout (176 bytes, 256 elements):
//
//	uint16  d          — F16 super-scale for quantized scales
//	uint16  dmin       — F16 super-scale for quantized mins
//	uint8   scales[12] — packed 6-bit sub-block scales (8) + mins (8) — same as Q4_K
//	uint8   qh[32]     — high (5th) bit per element; bit (2p)  is element (p*64+l),
//	                      bit (2p+1) is element (p*64+32+l)
//	uint8   qs[128]    — low-4 quants (same packing as Q4_K)
//
// Each sub-block has a 6-bit scale and 6-bit min unpacked via the shared
// getScaleMinK4. The 5-bit quant q5 ∈ [0, 31]; dequant: y = d*sc*q5 - dmin*m.
const Q5_KBlockBytes = 176

// DequantQ5_K reads a Q5_K tensor and returns F32.
func DequantQ5_K(src *tensor.Tensor) (*tensor.Tensor, error) {
	if src.DType() != dtype.Q5_K {
		return nil, fmt.Errorf("DequantQ5_K: input dtype is %v, want Q5_K", src.DType())
	}
	n := src.NumElements()
	if n%QK_K != 0 {
		return nil, fmt.Errorf("DequantQ5_K: element count %d not a multiple of %d", n, QK_K)
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
	numBlocks := n / QK_K
	if len(srcBuf) != numBlocks*Q5_KBlockBytes {
		return nil, fmt.Errorf("DequantQ5_K: src has %d bytes, expected %d", len(srcBuf), numBlocks*Q5_KBlockBytes)
	}
	for b := range numBlocks {
		blk := srcBuf[b*Q5_KBlockBytes : (b+1)*Q5_KBlockBytes]
		d := halfFromUint16(binary.LittleEndian.Uint16(blk[0:2]))
		dmin := halfFromUint16(binary.LittleEndian.Uint16(blk[2:4]))
		scales := blk[4:16]
		qh := blk[16:48]
		qs := blk[48:176]

		base := b * QK_K * 4
		for p := range 4 {
			sc0, m0 := getScaleMinK4(2*p, scales)
			sc1, m1 := getScaleMinK4(2*p+1, scales)
			d1 := d * float32(sc0)
			mo1 := dmin * float32(m0)
			d2 := d * float32(sc1)
			mo2 := dmin * float32(m1)
			bit1 := uint8(1 << (2 * p))
			bit2 := uint8(1 << (2*p + 1))
			qsOff := p * 32
			for l := range 32 {
				byteVal := qs[qsOff+l]
				qhByte := qh[l]
				q5lo := int(byteVal & 0x0F)
				if qhByte&bit1 != 0 {
					q5lo |= 0x10
				}
				q5hi := int(byteVal >> 4)
				if qhByte&bit2 != 0 {
					q5hi |= 0x10
				}
				putF32(outBuf[base+(p*64+l)*4:], d1*float32(q5lo)-mo1)
				putF32(outBuf[base+(p*64+32+l)*4:], d2*float32(q5hi)-mo2)
			}
		}
	}
	return out, nil
}
