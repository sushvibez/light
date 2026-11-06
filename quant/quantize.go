package quant

import (
	"encoding/binary"
	"fmt"
	"math"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// Quantize converts an F32 tensor to one of the block formats that have a
// CPU quantizer here (Q4_0, Q5_0, Q8_0), following ggml's reference
// algorithms (quantize_row_*_ref). Blocks run over the flattened elements,
// so the element count must be a multiple of 32; GGUF rows (the last dim)
// are block-aligned by construction. The K-quant formats only have
// dequantizers — their encoders are search-based and not needed to load
// checkpoints.
func Quantize(src *tensor.Tensor, to dtype.DType) (*tensor.Tensor, error) {
	if src.DType() != dtype.F32 {
		return nil, fmt.Errorf("quant.Quantize: source must be F32, got %v", src.DType())
	}
	if src.Device().Kind() != device.KindCPU {
		return nil, fmt.Errorf("quant.Quantize: CPU only")
	}
	n := src.NumElements()
	if n%QK4_0 != 0 {
		return nil, fmt.Errorf("quant.Quantize: %d elements not a multiple of 32", n)
	}
	c, err := src.Contiguous()
	if err != nil {
		return nil, err
	}
	x := tensor.MustSlice[float32](c)
	var enc func(dst []byte, blk []float32)
	switch to {
	case dtype.Q8_0:
		enc = encodeQ8_0
	case dtype.Q5_0:
		enc = encodeQ5_0
	case dtype.Q4_0:
		enc = encodeQ4_0
	default:
		return nil, fmt.Errorf("quant.Quantize: no encoder for %v (Q4_0, Q5_0, Q8_0 supported)", to)
	}
	bb := to.BlockBytes()
	out := make([]byte, (n/QK4_0)*bb)
	for b := range n / QK4_0 {
		enc(out[b*bb:(b+1)*bb], x[b*QK4_0:(b+1)*QK4_0])
	}
	return tensor.FromBytes(out, to, src.Shape().Dims(), device.CPU())
}

// signedMax returns the element with the largest magnitude, sign kept (ggml
// uses it so the value that saturates the grid lands exactly on -8 / -16).
func signedMax(blk []float32) float32 {
	var amax, mx float32
	for _, v := range blk {
		if a := float32(math.Abs(float64(v))); a > amax {
			amax, mx = a, v
		}
	}
	return mx
}

func encodeQ8_0(dst []byte, blk []float32) {
	var amax float32
	for _, v := range blk {
		if a := float32(math.Abs(float64(v))); a > amax {
			amax = a
		}
	}
	d := amax / 127
	id := float32(0)
	if d != 0 {
		id = 1 / d
	}
	binary.LittleEndian.PutUint16(dst[0:2], uint16(dtype.HalfFromF32(d)))
	for i, v := range blk {
		dst[2+i] = byte(int8(math.RoundToEven(float64(v * id))))
	}
}

func encodeQ4_0(dst []byte, blk []float32) {
	d := signedMax(blk) / -8
	id := float32(0)
	if d != 0 {
		id = 1 / d
	}
	binary.LittleEndian.PutUint16(dst[0:2], uint16(dtype.HalfFromF32(d)))
	for j := range 16 {
		lo := clampQ(blk[j]*id+8.5, 15)
		hi := clampQ(blk[j+16]*id+8.5, 15)
		dst[2+j] = byte(lo) | byte(hi)<<4
	}
}

func encodeQ5_0(dst []byte, blk []float32) {
	d := signedMax(blk) / -16
	id := float32(0)
	if d != 0 {
		id = 1 / d
	}
	binary.LittleEndian.PutUint16(dst[0:2], uint16(dtype.HalfFromF32(d)))
	var qh uint32
	for j := range 16 {
		lo := clampQ(blk[j]*id+16.5, 31)
		hi := clampQ(blk[j+16]*id+16.5, 31)
		dst[6+j] = byte(lo&0x0F) | byte(hi&0x0F)<<4
		qh |= uint32((lo&0x10)>>4) << j
		qh |= uint32((hi&0x10)>>4) << (j + 16)
	}
	binary.LittleEndian.PutUint32(dst[2:6], qh)
}

func clampQ(v float32, hi int) int {
	q := int(v) // ggml truncates after the +0.5 offset
	if v < 0 {
		q = 0
	}
	return min(max(q, 0), hi)
}
