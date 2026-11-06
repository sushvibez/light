package quant

import (
	"encoding/binary"
	"fmt"

	"light/dtype"
	"light/tensor"
)

// Q2_K super-block layout (84 bytes, 256 elements):
//
//	uint8   scales[16]  — per-sub-block 4-bit scale (low nibble) + 4-bit min
//	                      (high nibble). 16 sub-blocks of 16 elements each.
//	uint8   qs[64]      — 2-bit quants. Byte qs[k] holds four elements'
//	                      2-bit values, one per shift ∈ {0, 2, 4, 6}.
//	uint16  d           — F16 super-scale (for sub-block scales)
//	uint16  dmin        — F16 super-scale (for sub-block mins)
//
// Element layout per ggml (QK_K = 256, split into 2 halves of 128):
//
//	half       = idx / 128       (0 or 1)
//	wh         = idx % 128
//	j          = wh / 32         (0..3) → shift = 2 * j
//	sub_index  = (wh % 32) / 16  (0 or 1)
//	l          = (wh % 32) % 16
//	scale_idx  = half * 8 + j * 2 + sub_index
//	qs_byte    = qs[half * 32 + sub_index * 16 + l]
//	q2         = (qs_byte >> shift) & 3
//	y = d * (scale_idx_low4) * q2 - dmin * (scale_idx_high4)
const Q2_KBlockBytes = 84

// DequantQ2_K reads a Q2_K tensor and returns F32.
func DequantQ2_K(src *tensor.Tensor) (*tensor.Tensor, error) {
	if src.DType() != dtype.Q2_K {
		return nil, fmt.Errorf("DequantQ2_K: input dtype is %v, want Q2_K", src.DType())
	}
	n := src.NumElements()
	if n%QK_K != 0 {
		return nil, fmt.Errorf("DequantQ2_K: element count %d not a multiple of %d", n, QK_K)
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
	if len(srcBuf) != numBlocks*Q2_KBlockBytes {
		return nil, fmt.Errorf("DequantQ2_K: src has %d bytes, expected %d", len(srcBuf), numBlocks*Q2_KBlockBytes)
	}
	for b := range numBlocks {
		blk := srcBuf[b*Q2_KBlockBytes : (b+1)*Q2_KBlockBytes]
		scales := blk[0:16]
		qs := blk[16:80]
		d := halfFromUint16(binary.LittleEndian.Uint16(blk[80:82]))
		dmin := halfFromUint16(binary.LittleEndian.Uint16(blk[82:84]))

		outBase := b * QK_K * 4
		for half := range 2 {
			qsBase := half * 32
			scBase := half * 8
			yBase := half * 128
			for j := range 4 {
				shift := 2 * j
				sc0 := scales[scBase+j*2]
				sc1 := scales[scBase+j*2+1]
				dl0 := d * float32(sc0&0x0F)
				ml0 := dmin * float32(sc0>>4)
				dl1 := d * float32(sc1&0x0F)
				ml1 := dmin * float32(sc1>>4)
				for l := range 16 {
					q0 := int(qs[qsBase+l]>>shift) & 3
					q1 := int(qs[qsBase+16+l]>>shift) & 3
					putF32(outBuf[outBase+(yBase+j*32+l)*4:], dl0*float32(q0)-ml0)
					putF32(outBuf[outBase+(yBase+j*32+16+l)*4:], dl1*float32(q1)-ml1)
				}
			}
		}
	}
	return out, nil
}
