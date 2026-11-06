package ops

import (
	"fmt"

	"light/dtype"
	"light/tensor"
)

// MaskedFill returns `base` with `fillValue` substituted wherever mask is
// non-zero. Built on Where for now.
//
//	out = where(mask, fillValue, base)
func MaskedFill(base, mask *tensor.Tensor, fillValue float32) (*tensor.Tensor, error) {
	if mask.DType() != dtype.U8 {
		return nil, fmt.Errorf("masked_fill: mask must be U8 (got %v)", mask.DType())
	}
	if base.DType() != dtype.F32 && base.DType() != dtype.F64 {
		// Other dtypes would need a dtype-parameterized fill value; keep it
		// simple for now.
		return nil, fmt.Errorf("masked_fill: only F32/F64 base supported (got %v)", base.DType())
	}
	// Build fill tensor at base's full shape so broadcast-less Where (the
	// CUDA path) works without any preprocess.
	var fillFinal *tensor.Tensor
	var err error
	switch base.DType() {
	case dtype.F32:
		fillFinal, err = tensor.Full[float32](fillValue, base.Shape().Dims(), base.Device())
	case dtype.F64:
		fillFinal, err = tensor.Full[float64](float64(fillValue), base.Shape().Dims(), base.Device())
	}
	if err != nil {
		return nil, err
	}
	return Where(mask, fillFinal, base)
}

// MaskedSelect returns a 1-D tensor containing the input values where mask
// is non-zero. Output size equals the number of non-zero entries in mask.
// Input and mask must be broadcastable to the same shape.
func MaskedSelect(input, mask *tensor.Tensor) (*tensor.Tensor, error) {
	if mask.DType() != dtype.U8 {
		return nil, fmt.Errorf("masked_select: mask must be U8 (got %v)", mask.DType())
	}
	if !input.Device().SameAs(mask.Device()) {
		return nil, fmt.Errorf("masked_select: device mismatch")
	}
	// We need CPU-side iteration over both tensors; contiguify first.
	input, err := ensureContig(input)
	if err != nil {
		return nil, err
	}
	mask, err = ensureContig(mask)
	if err != nil {
		return nil, err
	}
	if !input.Shape().Equal(mask.Shape()) {
		return nil, fmt.Errorf("masked_select: input/mask shape mismatch (%v vs %v) — broadcasting not supported yet",
			input.Shape(), mask.Shape())
	}
	// Bring to CPU to count and build output (CUDA variant is a dedicated kernel).
	// For now, CPU-only.
	if input.Device().Kind() != 0 { // KindCPU = 0
		return nil, fmt.Errorf("masked_select: CUDA not supported yet (Phase 2 TODO)")
	}
	mBuf := mask.Storage().CPUBytes()
	count := 0
	for _, b := range mBuf {
		if b != 0 {
			count++
		}
	}
	out, err := tensor.Zeros(input.DType(), []int{count}, input.Device())
	if err != nil {
		return nil, err
	}
	elemSize := input.DType().Size()
	if elemSize == 0 {
		return nil, fmt.Errorf("masked_select: unsupported dtype %v", input.DType())
	}
	srcBuf := input.Storage().CPUBytes()
	dstBuf := out.Storage().CPUBytes()
	dstIdx := 0
	for i, b := range mBuf {
		if b != 0 {
			copy(dstBuf[dstIdx*elemSize:(dstIdx+1)*elemSize], srcBuf[i*elemSize:(i+1)*elemSize])
			dstIdx++
		}
	}
	return out, nil
}
