package quant

import (
	"encoding/binary"
	"fmt"

	"light/dtype"
	"light/tensor"
)

// QK_K is the super-block size shared by all K-quants (Q2_K..Q8_K).
const QK_K = 256

// Q4_K super-block layout (144 bytes, 256 elements):
//
//	uint16  d          — F16 super-scale for quantized sub-block scales
//	uint16  dmin       — F16 super-scale for quantized sub-block mins
//	uint8   scales[12] — packed 6-bit sub-block scales (8) + 6-bit mins (8)
//	uint8   qs[128]    — packed 4-bit quants (256 × 4b / 8 = 128)
//
// Each super-block is 8 sub-blocks of 32 elements. Sub-block j (0..7) has
// its own scale `sc_j` and min `m_j`, both 6-bit, unpacked from `scales`.
// Dequant: y = d * sc_j * q4 - dmin * m_j, with q4 in [0, 15].
const Q4_KBlockBytes = 144

// DequantQ4_K reads a Q4_K tensor and returns F32. CPU path; CUDA dispatch
// via the shared hook when the source lives on a CUDA device.
func DequantQ4_K(src *tensor.Tensor) (*tensor.Tensor, error) {
	if src.DType() != dtype.Q4_K {
		return nil, fmt.Errorf("DequantQ4_K: input dtype is %v, want Q4_K", src.DType())
	}
	n := src.NumElements()
	if n%QK_K != 0 {
		return nil, fmt.Errorf("DequantQ4_K: element count %d not a multiple of super-block size %d", n, QK_K)
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
	if len(srcBuf) != numBlocks*Q4_KBlockBytes {
		return nil, fmt.Errorf("DequantQ4_K: src has %d bytes, expected %d", len(srcBuf), numBlocks*Q4_KBlockBytes)
	}
	for b := range numBlocks {
		blk := srcBuf[b*Q4_KBlockBytes : (b+1)*Q4_KBlockBytes]
		d := halfFromUint16(binary.LittleEndian.Uint16(blk[0:2]))
		dmin := halfFromUint16(binary.LittleEndian.Uint16(blk[2:4]))
		scales := blk[4:16]
		qs := blk[16:144]

		base := b * QK_K * 4
		for p := range 4 {
			sc0, m0 := getScaleMinK4(2*p, scales)
			sc1, m1 := getScaleMinK4(2*p+1, scales)
			d1 := d * float32(sc0)
			mo1 := dmin * float32(m0)
			d2 := d * float32(sc1)
			mo2 := dmin * float32(m1)
			qsOff := p * 32
			for l := range 32 {
				byteVal := qs[qsOff+l]
				lo := int(byteVal & 0x0F)
				hi := int(byteVal >> 4)
				putF32(outBuf[base+(p*64+l)*4:], d1*float32(lo)-mo1)
				putF32(outBuf[base+(p*64+32+l)*4:], d2*float32(hi)-mo2)
			}
		}
	}
	return out, nil
}

// getScaleMinK4 unpacks the 6-bit scale and 6-bit min for sub-block j (0..7)
// from the 12-byte `scales` array using the ggml Q4_K / Q5_K convention.
func getScaleMinK4(j int, q []byte) (sc, m uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 0x0F) | ((q[j-4] >> 6) << 4),
		(q[j+4] >> 4) | ((q[j] >> 6) << 4)
}
