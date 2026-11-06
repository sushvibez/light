package ops

import (
	"fmt"

	"light/dtype"
	"light/internal/registry"
	"light/shape"
	"light/tensor"
)

// LT returns a boolean (U8) tensor with 1 where a < b (elementwise broadcast),
// 0 otherwise.
func LT(a, b *tensor.Tensor) (*tensor.Tensor, error) { return compareDispatch("lt", a, b) }

// LE returns a <= b.
func LE(a, b *tensor.Tensor) (*tensor.Tensor, error) { return compareDispatch("le", a, b) }

// GT returns a > b.
func GT(a, b *tensor.Tensor) (*tensor.Tensor, error) { return compareDispatch("gt", a, b) }

// GE returns a >= b.
func GE(a, b *tensor.Tensor) (*tensor.Tensor, error) { return compareDispatch("ge", a, b) }

// EQ returns a == b.
func EQ(a, b *tensor.Tensor) (*tensor.Tensor, error) { return compareDispatch("eq", a, b) }

// NE returns a != b.
func NE(a, b *tensor.Tensor) (*tensor.Tensor, error) { return compareDispatch("ne", a, b) }

func compareDispatch(op string, a, b *tensor.Tensor) (*tensor.Tensor, error) {
	if !a.Device().SameAs(b.Device()) {
		return nil, fmt.Errorf("%s: device mismatch", op)
	}
	if a.DType() != b.DType() {
		return nil, fmt.Errorf("%s: dtype mismatch (%v vs %v)", op, a.DType(), b.DType())
	}
	outShape, err := shape.Broadcast(a.Shape(), b.Shape())
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	out, err := tensor.Zeros(dtype.U8, outShape.Dims(), a.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: op, Dev: a.Device().Kind(), DType: a.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{Inputs: []*tensor.Tensor{a, b}, Output: out}); err != nil {
		return nil, err
	}
	return out, nil
}

// LogicalAnd returns a * b (elementwise) on U8 tensors, i.e. boolean AND.
func LogicalAnd(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	return logicalBinary("logical_and", a, b)
}

// LogicalOr returns a boolean OR on U8 tensors.
func LogicalOr(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	return logicalBinary("logical_or", a, b)
}

// LogicalNot returns 1 - a on U8 tensors.
func LogicalNot(a *tensor.Tensor) (*tensor.Tensor, error) {
	if a.DType() != dtype.U8 {
		return nil, fmt.Errorf("logical_not: input must be U8 (got %v)", a.DType())
	}
	return unaryDispatch("logical_not", a)
}

func logicalBinary(op string, a, b *tensor.Tensor) (*tensor.Tensor, error) {
	if a.DType() != dtype.U8 || b.DType() != dtype.U8 {
		return nil, fmt.Errorf("%s: inputs must both be U8", op)
	}
	return binaryDispatch(op, a, b)
}
