package ops

import (
	"fmt"

	"light/internal/registry"
	"light/tensor"
)

// Conv3d runs a 3-D convolution (cross-correlation, PyTorch semantics).
//
//	input:   (N, Cin, D, H, W)
//	weight:  (Cout, Cin, kD, kH, kW)
//	bias:    (Cout,) or nil
//	stride:  [sD, sH, sW]
//	padding: [pD, pH, pW]
//
// Returns (N, Cout, Dout, Hout, Wout) with Dout = (D + 2*pD - kD)/sD + 1 etc.
func Conv3d(input, weight, bias *tensor.Tensor, stride, padding [3]int) (*tensor.Tensor, error) {
	if input.Rank() != 5 {
		return nil, fmt.Errorf("Conv3d: input must be (N, Cin, D, H, W); got rank %d", input.Rank())
	}
	if weight.Rank() != 5 {
		return nil, fmt.Errorf("Conv3d: weight must be (Cout, Cin, kD, kH, kW); got rank %d", weight.Rank())
	}
	if !input.Device().SameAs(weight.Device()) {
		return nil, fmt.Errorf("Conv3d: device mismatch input vs weight")
	}
	if input.DType() != weight.DType() {
		return nil, fmt.Errorf("Conv3d: dtype mismatch input vs weight (%v vs %v)", input.DType(), weight.DType())
	}
	inDims := input.Shape().Dims()
	wDims := weight.Shape().Dims()
	if inDims[1] != wDims[1] {
		return nil, fmt.Errorf("Conv3d: input Cin=%d != weight Cin=%d", inDims[1], wDims[1])
	}
	for i := range 3 {
		if stride[i] <= 0 {
			return nil, fmt.Errorf("Conv3d: stride must be positive")
		}
		if padding[i] < 0 {
			return nil, fmt.Errorf("Conv3d: padding must be >= 0")
		}
	}
	N, Cout := inDims[0], wDims[0]
	outSp := [3]int{}
	for i := range 3 {
		outSp[i] = (inDims[2+i]+2*padding[i]-wDims[2+i])/stride[i] + 1
		if outSp[i] <= 0 {
			return nil, fmt.Errorf("Conv3d: output spatial dim %d non-positive", i)
		}
	}
	src, err := ensureContig(input)
	if err != nil {
		return nil, err
	}
	wC, err := ensureContig(weight)
	if err != nil {
		return nil, err
	}
	inputs := []*tensor.Tensor{src, wC}
	if bias != nil {
		if bias.Rank() != 1 || bias.Dim(0) != Cout {
			return nil, fmt.Errorf("Conv3d: bias must be (Cout,)")
		}
		if !bias.Device().SameAs(input.Device()) || bias.DType() != input.DType() {
			return nil, fmt.Errorf("Conv3d: bias device/dtype mismatch")
		}
		bC, err := ensureContig(bias)
		if err != nil {
			return nil, err
		}
		inputs = append(inputs, bC)
	}
	out, err := tensor.Zeros(input.DType(), []int{N, Cout, outSp[0], outSp[1], outSp[2]}, input.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "conv3d", Dev: input.Device().Kind(), DType: input.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: inputs,
		Output: out,
		Attrs:  map[string]any{"stride": stride, "padding": padding},
	}); err != nil {
		return nil, err
	}
	return out, nil
}
