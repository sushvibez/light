// Package ops holds the public tensor operation surface. Each function
// validates inputs, allocates the output, and dispatches to the correct
// kernel via internal/registry.
package ops

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/shape"
	"light/tensor"

	// Blank-import CPU kernels so their init() calls Register before any op
	// runs. CUDA kernels register via their own blank import when built
	// with -tags cuda (see ops_cuda.go).
	_ "light/kernels/cpu"
)

// validateBinaryOp checks that a and b share a device and dtype and returns
// the broadcast output shape.
func validateBinaryOp(op string, a, b *tensor.Tensor) (shape.Shape, error) {
	if !a.Device().SameAs(b.Device()) {
		return shape.Shape{}, fmt.Errorf("%s: device mismatch (%v vs %v)", op, a.Device(), b.Device())
	}
	if a.DType() != b.DType() {
		return shape.Shape{}, fmt.Errorf("%s: dtype mismatch (%v vs %v)", op, a.DType(), b.DType())
	}
	outShape, err := shape.Broadcast(a.Shape(), b.Shape())
	if err != nil {
		return shape.Shape{}, fmt.Errorf("%s: %w", op, err)
	}
	return outShape, nil
}

// dispatch looks up and invokes a kernel.
func dispatch(op string, dev device.Device, dt dtype.DType, inputs []*tensor.Tensor, out *tensor.Tensor) error {
	k, err := registry.Lookup(registry.Key{Op: op, Dev: dev.Kind(), DType: dt})
	if err != nil {
		return err
	}
	return k(&registry.Context{Inputs: inputs, Output: out})
}
