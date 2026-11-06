package quant

import (
	"encoding/binary"
	"fmt"
	"math"

	"light/dtype"
	"light/tensor"
)

// Q8_K super-block layout (292 bytes, 256 elements):
//
//	float32 d          — F32 super-block scale (NOT F16 like other K-quants)
//	int8    qs[256]    — signed 8-bit quants
//	int16   bsums[16]  — per-sub-block sums (matmul metadata, ignored here)
//
// Dequant: y[i] = d * qs[i]. Q8_K exists primarily as an intermediate for the
// Q*_K × Q8_K fused matmul path; the bsums field accelerates that path but
// carries no extra information for pure dequant.
const Q8_KBlockBytes = 292

// DequantQ8_K reads a Q8_K tensor and returns F32.
func DequantQ8_K(src *tensor.Tensor) (*tensor.Tensor, error) {
	if src.DType() != dtype.Q8_K {
		return nil, fmt.Errorf("DequantQ8_K: input dtype is %v, want Q8_K", src.DType())
	}
	n := src.NumElements()
	if n%QK_K != 0 {
		return nil, fmt.Errorf("DequantQ8_K: element count %d not a multiple of %d", n, QK_K)
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
	if len(srcBuf) != numBlocks*Q8_KBlockBytes {
		return nil, fmt.Errorf("DequantQ8_K: src has %d bytes, expected %d", len(srcBuf), numBlocks*Q8_KBlockBytes)
	}
	for b := range numBlocks {
		blk := srcBuf[b*Q8_KBlockBytes : (b+1)*Q8_KBlockBytes]
		d := math.Float32frombits(binary.LittleEndian.Uint32(blk[0:4]))
		qs := blk[4:260]
		base := b * QK_K * 4
		for i := range QK_K {
			q := int8(qs[i])
			putF32(outBuf[base+i*4:], d*float32(q))
		}
	}
	return out, nil
}
