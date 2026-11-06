package ops

import (
	"light/tensor"
)

// Add returns a + b, broadcasting shapes as necessary. Inputs must share a
// device and dtype.
func Add(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	outShape, err := validateBinaryOp("add", a, b)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(a.DType(), outShape.Dims(), a.Device())
	if err != nil {
		return nil, err
	}
	if err := dispatch("add", a.Device(), a.DType(), []*tensor.Tensor{a, b}, out); err != nil {
		return nil, err
	}
	return out, nil
}
