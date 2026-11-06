package quant

import (
	"encoding/binary"
	"fmt"

	"light/dtype"
	"light/tensor"
)

// Q3_K super-block layout (110 bytes, 256 elements):
//
//	uint8   hmask[32]  — high (3rd) bit per element. Bit `half` of hmask[k]
//	                      encodes the 3rd bit for its element.
//	uint8   qs[64]     — low 2 bits per element (4 elements packed per byte
//	                      via shifts 0,2,4,6).
//	uint8   scales[12] — 16 signed 6-bit scales, packed with the ggml Q3_K shuffle.
//	uint16  d          — F16 super-scale.
//
// Quant formula per ggml:
//
//	q = (qs_byte >> shift) & 3               -- low 2 bits
//	if hmask_bit == 0: q -= 4                -- range becomes [-4, -1]
//	y = d * (scale_raw - 32) * q             -- scale is signed 6-bit
//
// Element layout mirrors Q2_K/Q3_K:
//
//	half = idx / 128; wh = idx % 128
//	j = wh / 32  → shift = 2 * j
//	sub_index = (wh % 32) / 16; l = (wh % 32) % 16
//	scale_idx = half * 8 + j * 2 + sub_index
//	qs_byte   = qs[half * 32 + sub_index * 16 + l]
//	hm_byte   = hmask[sub_index * 16 + l]
//	hmask_bit = half     (0 for first 128 elems, 1 for second 128)
const Q3_KBlockBytes = 110

// DequantQ3_K reads a Q3_K tensor and returns F32.
func DequantQ3_K(src *tensor.Tensor) (*tensor.Tensor, error) {
	if src.DType() != dtype.Q3_K {
		return nil, fmt.Errorf("DequantQ3_K: input dtype is %v, want Q3_K", src.DType())
	}
	n := src.NumElements()
	if n%QK_K != 0 {
		return nil, fmt.Errorf("DequantQ3_K: element count %d not a multiple of %d", n, QK_K)
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
	if len(srcBuf) != numBlocks*Q3_KBlockBytes {
		return nil, fmt.Errorf("DequantQ3_K: src has %d bytes, expected %d", len(srcBuf), numBlocks*Q3_KBlockBytes)
	}
	for b := range numBlocks {
		blk := srcBuf[b*Q3_KBlockBytes : (b+1)*Q3_KBlockBytes]
		hmask := blk[0:32]
		qs := blk[32:96]
		scales := blk[96:108]
		d := halfFromUint16(binary.LittleEndian.Uint16(blk[108:110]))

		// Unpack 16 signed 6-bit scales from the 12-byte packed `scales`.
		var unpacked [16]int8
		unpackQ3KScales(scales, &unpacked)

		outBase := b * QK_K * 4
		for half := range 2 {
			qsBase := half * 32
			scBase := half * 8
			yBase := half * 128
			hmBit := uint8(1) << half
			for j := range 4 {
				shift := uint(2 * j)
				sc0 := float32(unpacked[scBase+j*2])
				sc1 := float32(unpacked[scBase+j*2+1])
				for l := range 16 {
					q0raw := int(qs[qsBase+l]>>shift) & 3
					q1raw := int(qs[qsBase+16+l]>>shift) & 3
					q0 := q0raw
					if hmask[l]&hmBit == 0 {
						q0 -= 4
					}
					q1 := q1raw
					if hmask[16+l]&hmBit == 0 {
						q1 -= 4
					}
					putF32(outBuf[outBase+(yBase+j*32+l)*4:], d*sc0*float32(q0))
					putF32(outBuf[outBase+(yBase+j*32+16+l)*4:], d*sc1*float32(q1))
				}
			}
		}
	}
	return out, nil
}

// unpackQ3KScales unpacks 16 signed 6-bit scales from the 12-byte packed
// layout that ggml uses for Q3_K. `out` receives `sc - 32` for each sub-block.
//
// Input (12 bytes):
//
//	bytes 0..3 : low4 = scale[0..3] low4;  top2 = scale[8..11] low2
//	bytes 4..7 : low4 = scale[4..7] low4;  top2 = scale[8..11] mid2 (bits 2..3)
//	bytes 8..11: low4 = scale[0..3] high2; high4 = scale[4..7] high2 + ...
//
// Rather than replicate the bit-shuffle literally, we use the canonical ggml
// extraction formulas.
func unpackQ3KScales(packed []byte, out *[16]int8) {
	var aux [4]uint32
	var a [16]byte
	aux[0] = uint32(packed[0]) | uint32(packed[1])<<8 | uint32(packed[2])<<16 | uint32(packed[3])<<24
	aux[1] = uint32(packed[4]) | uint32(packed[5])<<8 | uint32(packed[6])<<16 | uint32(packed[7])<<24
	aux[2] = uint32(packed[8]) | uint32(packed[9])<<8 | uint32(packed[10])<<16 | uint32(packed[11])<<24

	const kmask1 = uint32(0x03030303)
	const kmask2 = uint32(0x0f0f0f0f)
	tmp := aux[2]
	aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
	aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
	aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
	aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)

	for k := range 4 {
		a[4*k+0] = byte(aux[k])
		a[4*k+1] = byte(aux[k] >> 8)
		a[4*k+2] = byte(aux[k] >> 16)
		a[4*k+3] = byte(aux[k] >> 24)
	}
	for i := range 16 {
		out[i] = int8(a[i]) - 32
	}
}
