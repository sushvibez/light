// Package quant implements dequantization of GGUF-style block-quantized
// weights (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K..Q8_K) into float32.
//
// This is the critical bridge for loading llama.cpp / GGUF model files: the
// file format stores quantized weights, but our compute kernels operate on
// float32/float16. Dequant runs once at load time (or lazily per matmul).
//
// Currently only Q4_0 is implemented; others to follow.
package quant

import (
	"fmt"

	"light/dtype"
	"light/tensor"
)

// QK4_0 is the block size (32 elements per block) for Q4_0 / Q4_1 / etc.
const QK4_0 = 32

// Q4_0 block layout in GGUF (18 bytes total):
//
//	uint16 d         — F16 scale
//	uint8[16] qs     — packed 4-bit quants; low nibble is element j, high nibble is element j+16.
//
// Dequant formula:
//
//	value[j]        = d * (int4(qs[j] & 0x0F) - 8)      for j in [0, 16)
//	value[j + 16]   = d * (int4(qs[j] >> 4)  - 8)       for j in [0, 16)
const Q4_0BlockBytes = 18

// DequantQ4_0 reads a Q4_0-dtype tensor (block-encoded) and returns an F32
// tensor with the same logical shape. CPU only.
func DequantQ4_0(src *tensor.Tensor) (*tensor.Tensor, error) {
	if src.DType() != dtype.Q4_0 {
		return nil, fmt.Errorf("DequantQ4_0: input dtype is %v, want Q4_0", src.DType())
	}
	n := src.NumElements()
	if n%QK4_0 != 0 {
		return nil, fmt.Errorf("DequantQ4_0: element count %d not a multiple of block size %d", n, QK4_0)
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
	if err := dequantQ4_0CPU(srcBuf, outBuf, n); err != nil {
		return nil, err
	}
	return out, nil
}

// dequantQ4_0CPU does the actual dequant work on raw bytes.
func dequantQ4_0CPU(src []byte, out []byte, n int) error {
	numBlocks := n / QK4_0
	expectedSrc := numBlocks * Q4_0BlockBytes
	if len(src) != expectedSrc {
		return fmt.Errorf("DequantQ4_0: src has %d bytes, expected %d (%d blocks × %d bytes)",
			len(src), expectedSrc, numBlocks, Q4_0BlockBytes)
	}
	expectedOut := n * 4 // F32
	if len(out) != expectedOut {
		return fmt.Errorf("DequantQ4_0: out has %d bytes, expected %d", len(out), expectedOut)
	}
	for b := range numBlocks {
		blockSrc := src[b*Q4_0BlockBytes : (b+1)*Q4_0BlockBytes]
		scale := halfFromUint16(uint16(blockSrc[0]) | uint16(blockSrc[1])<<8)
		qs := blockSrc[2:]
		outBase := b * QK4_0 * 4
		for j := range QK4_0 / 2 {
			lo := int(qs[j]&0x0F) - 8
			hi := int(qs[j]>>4) - 8
			putF32(out[outBase+(j)*4:], scale*float32(lo))
			putF32(out[outBase+(j+16)*4:], scale*float32(hi))
		}
	}
	return nil
}

// halfFromUint16 converts an IEEE-754 half precision (stored as uint16) to float32.
// Pulled inline here to avoid pulling in dtype.Half's method via conversion.
func halfFromUint16(u uint16) float32 {
	return dtype.Half(u).ToF32()
}

// putF32 writes a float32 as little-endian 4 bytes.
func putF32(dst []byte, v float32) {
	bits := uint32FromFloat(v)
	dst[0] = byte(bits)
	dst[1] = byte(bits >> 8)
	dst[2] = byte(bits >> 16)
	dst[3] = byte(bits >> 24)
}

// uint32FromFloat is math.Float32bits without importing math in the hot path.
func uint32FromFloat(f float32) uint32 {
	// Use math.Float32bits — cheaper to just import. Inline via unsafe here.
	return dtype.F32bits(f)
}
