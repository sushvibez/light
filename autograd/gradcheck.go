package autograd

import (
	"fmt"
	"math"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// ComputeFn is a closure that takes a list of leaf tensors and returns a
// single scalar tensor. gradcheck compares analytical gradients from
// Backward against numerical gradients computed by central finite
// differences along each element of every leaf.
type ComputeFn func(leaves []*tensor.Tensor) (*tensor.Tensor, error)

// GradCheck verifies that analytical gradients match central finite
// differences within the given absolute tolerance. Each leaf tensor must be
// CPU F32; eps is the perturbation used for finite differencing. Returns an
// error describing the first mismatch (or nil on success).
func GradCheck(fn ComputeFn, leaves []*tensor.Tensor, eps, atol float64) error {
	for _, l := range leaves {
		if l.Device().Kind() != device.KindCPU {
			return fmt.Errorf("GradCheck: leaves must be CPU (got %v)", l.Device())
		}
		if l.DType() != dtype.F32 {
			return fmt.Errorf("GradCheck: leaves must be F32 (got %v)", l.DType())
		}
		l.SetRequiresGrad(true)
		l.SetGrad(nil)
	}

	// Analytical: run forward + backward.
	out, err := fn(leaves)
	if err != nil {
		return fmt.Errorf("GradCheck: forward (analytical): %w", err)
	}
	if out.NumElements() != 1 {
		return fmt.Errorf("GradCheck: output must be scalar, got shape %v", out.Shape().Dims())
	}
	if err := Backward(out, nil); err != nil {
		return fmt.Errorf("GradCheck: backward: %w", err)
	}

	analytic := make([][]float32, len(leaves))
	for i, l := range leaves {
		g := l.Grad()
		if g == nil {
			return fmt.Errorf("GradCheck: leaf %d has nil grad after Backward", i)
		}
		cont, err := g.Contiguous()
		if err != nil {
			return err
		}
		analytic[i] = tensor.MustSlice[float32](cont)
	}

	// Numerical: central difference per element, per leaf.
	for li, l := range leaves {
		buf := l.Storage().CPUBytes()
		n := l.NumElements()
		vals := unsafeF32Slice(buf, n)
		for k := range vals {
			orig := vals[k]
			vals[k] = orig + float32(eps)
			// Tree forward only (detach grad tracking by clearing gradFn
			// state via a fresh call; leaves' RequiresGrad stays true but
			// the new outputs get fresh gradFns that we discard).
			yPlusT, err := fn(leaves)
			if err != nil {
				return fmt.Errorf("GradCheck: fd fwd+: %w", err)
			}
			yPlus := tensor.MustSlice[float32](yPlusT)[0]
			vals[k] = orig - float32(eps)
			yMinusT, err := fn(leaves)
			if err != nil {
				return fmt.Errorf("GradCheck: fd fwd-: %w", err)
			}
			yMinus := tensor.MustSlice[float32](yMinusT)[0]
			vals[k] = orig
			numer := float64(yPlus-yMinus) / (2 * eps)
			anal := float64(analytic[li][k])
			if math.Abs(numer-anal) > atol {
				return fmt.Errorf(
					"GradCheck: leaf %d index %d: analytic=%g numeric=%g |Δ|=%g > %g",
					li, k, anal, numer, math.Abs(numer-anal), atol)
			}
		}
	}
	return nil
}

// unsafeF32Slice reinterprets a CPU byte buffer as a []float32 of length n.
// Buf must be F32-typed storage; callers guarantee this.
func unsafeF32Slice(buf []byte, n int) []float32 {
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&buf[0])), n)
}
