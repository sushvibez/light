package quant

import (
	"encoding/binary"
	"fmt"

	"light/dtype"
	"light/tensor"
)

// Q6_K super-block layout (210 bytes, 256 elements):
//
//	uint8   ql[128]   — quants, lower 4 bits
//	uint8   qh[64]    — quants, upper 2 bits
//	int8    scales[16]— signed 8-bit sub-block scales (one per 16 elements)
//	uint16  d         — F16 super-scale
//
// The 256 elements split into 2 halves of 128 each; each half has 8 scales,
// 64 bytes of ql, and 32 bytes of qh. Within a half, the 128 elements form
// 4 groups of 32 indexed by l = 0..31:
//
//	elem[l +  0]: raw = (ql[l]    & 0xF) | ((qh[l] >> 0) & 3) << 4
//	elem[l + 32]: raw = (ql[l+32] & 0xF) | ((qh[l] >> 2) & 3) << 4
//	elem[l + 64]: raw = (ql[l]    >> 4)  | ((qh[l] >> 4) & 3) << 4
//	elem[l + 96]: raw = (ql[l+32] >> 4)  | ((qh[l] >> 6) & 3) << 4
//
// Each raw value is 6 bits in [0, 63]; signed q = raw - 32 in [-32, 31].
// Scale index within the half is base + group_offset + (l/16) where
// group_offset is 0/2/4/6 for the four groups.
// Dequant: y = d * scale * q.
const Q6_KBlockBytes = 210

// DequantQ6_K reads a Q6_K tensor and returns F32.
func DequantQ6_K(src *tensor.Tensor) (*tensor.Tensor, error) {
	if src.DType() != dtype.Q6_K {
		return nil, fmt.Errorf("DequantQ6_K: input dtype is %v, want Q6_K", src.DType())
	}
	n := src.NumElements()
	if n%QK_K != 0 {
		return nil, fmt.Errorf("DequantQ6_K: element count %d not a multiple of %d", n, QK_K)
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
	if len(srcBuf) != numBlocks*Q6_KBlockBytes {
		return nil, fmt.Errorf("DequantQ6_K: src has %d bytes, expected %d", len(srcBuf), numBlocks*Q6_KBlockBytes)
	}
	for b := range numBlocks {
		blk := srcBuf[b*Q6_KBlockBytes : (b+1)*Q6_KBlockBytes]
		ql := blk[0:128]
		qh := blk[128:192]
		sc := blk[192:208] // signed int8 scales
		d := halfFromUint16(binary.LittleEndian.Uint16(blk[208:210]))

		outBase := b * QK_K * 4
		for half := range 2 {
			qlBase := half * 64
			qhBase := half * 32
			scBase := half * 8
			yBase := half * 128
			for l := range 32 {
				is := l >> 4 // 0 or 1 (which scale within this group)
				qhByte := qh[qhBase+l]
				raw1 := int(ql[qlBase+l]&0x0F) | (int((qhByte>>0)&0x03) << 4)
				raw2 := int(ql[qlBase+l+32]&0x0F) | (int((qhByte>>2)&0x03) << 4)
				raw3 := int(ql[qlBase+l]>>4) | (int((qhByte>>4)&0x03) << 4)
				raw4 := int(ql[qlBase+l+32]>>4) | (int((qhByte>>6)&0x03) << 4)
				q1 := raw1 - 32
				q2 := raw2 - 32
				q3 := raw3 - 32
				q4 := raw4 - 32
				s1 := int8(sc[scBase+0+is])
				s2 := int8(sc[scBase+2+is])
				s3 := int8(sc[scBase+4+is])
				s4 := int8(sc[scBase+6+is])

				putF32(outBuf[outBase+(yBase+l)*4:], d*float32(s1)*float32(q1))
				putF32(outBuf[outBase+(yBase+l+32)*4:], d*float32(s2)*float32(q2))
				putF32(outBuf[outBase+(yBase+l+64)*4:], d*float32(s3)*float32(q3))
				putF32(outBuf[outBase+(yBase+l+96)*4:], d*float32(s4)*float32(q4))
			}
		}
	}
	return out, nil
}
