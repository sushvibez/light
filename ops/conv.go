package ops

import (
	"fmt"

	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// Conv1d runs a 1-D convolution: (N, Cin, L) × (Cout, Cin, kL) → (N, Cout, Lout).
func Conv1d(input, weight, bias *tensor.Tensor, stride, padding int) (*tensor.Tensor, error) {
	if input.Rank() != 3 {
		return nil, fmt.Errorf("Conv1d: input must be (N, Cin, L)")
	}
	if weight.Rank() != 3 {
		return nil, fmt.Errorf("Conv1d: weight must be (Cout, Cin, kL)")
	}
	if !input.Device().SameAs(weight.Device()) || input.DType() != weight.DType() {
		return nil, fmt.Errorf("Conv1d: device/dtype mismatch")
	}
	if stride <= 0 {
		return nil, fmt.Errorf("Conv1d: stride must be positive")
	}
	inDims := input.Shape().Dims()
	wDims := weight.Shape().Dims()
	if inDims[1] != wDims[1] {
		return nil, fmt.Errorf("Conv1d: Cin mismatch")
	}
	N, _, L := inDims[0], inDims[1], inDims[2]
	Cout := wDims[0]
	kL := wDims[2]
	Lout := (L+2*padding-kL)/stride + 1
	if Lout <= 0 {
		return nil, fmt.Errorf("Conv1d: output dim non-positive")
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
			return nil, fmt.Errorf("Conv1d: bias must be (Cout,)")
		}
		bC, err := ensureContig(bias)
		if err != nil {
			return nil, err
		}
		inputs = append(inputs, bC)
	}
	out, err := tensor.Zeros(input.DType(), []int{N, Cout, Lout}, input.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "conv1d", Dev: input.Device().Kind(), DType: input.DType()})
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

// UpsampleNearest doubles/triples/etc. the last two spatial dimensions by
// integer factors using nearest-neighbor replication.
//
//	input: (N, C, H, W)  → output (N, C, H*scale[0], W*scale[1])
func UpsampleNearest(input *tensor.Tensor, scale [2]int) (*tensor.Tensor, error) {
	if input.Rank() != 4 {
		return nil, fmt.Errorf("UpsampleNearest: input must be (N, C, H, W)")
	}
	if scale[0] <= 0 || scale[1] <= 0 {
		return nil, fmt.Errorf("UpsampleNearest: scale must be positive")
	}
	src, err := ensureContig(input)
	if err != nil {
		return nil, err
	}
	dims := input.Shape().Dims()
	out, err := tensor.Zeros(input.DType(),
		[]int{dims[0], dims[1], dims[2] * scale[0], dims[3] * scale[1]},
		input.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "upsample_nearest", Dev: input.Device().Kind(), DType: input.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{src},
		Output: out,
		Attrs:  map[string]any{"scale": scale},
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// Conv2d runs a 2-D convolution.
//
//	input:   (N, Cin, H, W)
//	weight:  (Cout, Cin, kH, kW)
//	bias:    (Cout,) or nil
//	stride:  [sH, sW]
//	padding: [pH, pW]
//
// Returns output of shape (N, Cout, Hout, Wout) where
// `Hout = (H + 2*pH - kH) / sH + 1`.
func Conv2d(input, weight, bias *tensor.Tensor, stride, padding [2]int) (*tensor.Tensor, error) {
	if input.Rank() != 4 {
		return nil, fmt.Errorf("Conv2d: input must be (N, Cin, H, W); got rank %d", input.Rank())
	}
	if weight.Rank() != 4 {
		return nil, fmt.Errorf("Conv2d: weight must be (Cout, Cin, kH, kW)")
	}
	if !input.Device().SameAs(weight.Device()) {
		return nil, fmt.Errorf("Conv2d: device mismatch input vs weight")
	}
	if input.DType() != weight.DType() {
		return nil, fmt.Errorf("Conv2d: dtype mismatch input vs weight (%v vs %v)", input.DType(), weight.DType())
	}
	// Bias is allowed to stay F32 even when input/weight are BF16 — that's
	// what the BF16 launcher expects (per-channel additive constants
	// benefit from full precision and the tensor is tiny). All other
	// combinations must match.
	if bias != nil && bias.DType() != input.DType() && bias.DType() != dtype.F32 {
		return nil, fmt.Errorf("Conv2d: bias dtype %v not allowed for %v input", bias.DType(), input.DType())
	}
	inDims := input.Shape().Dims()
	wDims := weight.Shape().Dims()
	if inDims[1] != wDims[1] {
		return nil, fmt.Errorf("Conv2d: input Cin=%d != weight Cin=%d", inDims[1], wDims[1])
	}
	N, Cin, H, W := inDims[0], inDims[1], inDims[2], inDims[3]
	Cout := wDims[0]
	kH := wDims[2]
	kW := wDims[3]
	sH, sW := stride[0], stride[1]
	pH, pW := padding[0], padding[1]
	if sH <= 0 || sW <= 0 {
		return nil, fmt.Errorf("Conv2d: stride must be positive")
	}
	Hout := (H+2*pH-kH)/sH + 1
	Wout := (W+2*pW-kW)/sW + 1
	if Hout <= 0 || Wout <= 0 {
		return nil, fmt.Errorf("Conv2d: output dims non-positive (H=%d W=%d)", Hout, Wout)
	}
	_ = Cin

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
			return nil, fmt.Errorf("Conv2d: bias must be (Cout,)")
		}
		bC, err := ensureContig(bias)
		if err != nil {
			return nil, err
		}
		inputs = append(inputs, bC)
	}

	out, err := tensor.Zeros(input.DType(), []int{N, Cout, Hout, Wout}, input.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "conv2d", Dev: input.Device().Kind(), DType: input.DType()})
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

// MaxPool2d applies max-pooling over a 2-D window.
//
//	input: (N, C, H, W) → output (N, C, Hout, Wout).
//	Hout = (H + 2*pH - kH) / sH + 1
func MaxPool2d(input *tensor.Tensor, kernel, stride, padding [2]int) (*tensor.Tensor, error) {
	return pool2dDispatch("maxpool2d", input, kernel, stride, padding)
}

// AvgPool2d applies average-pooling over a 2-D window (average is over the
// elements that land inside the unpadded region).
func AvgPool2d(input *tensor.Tensor, kernel, stride, padding [2]int) (*tensor.Tensor, error) {
	return pool2dDispatch("avgpool2d", input, kernel, stride, padding)
}

func pool2dDispatch(op string, input *tensor.Tensor, kernel, stride, padding [2]int) (*tensor.Tensor, error) {
	if input.Rank() != 4 {
		return nil, fmt.Errorf("%s: input must be (N, C, H, W); got rank %d", op, input.Rank())
	}
	inDims := input.Shape().Dims()
	N, C, H, W := inDims[0], inDims[1], inDims[2], inDims[3]
	kH, kW := kernel[0], kernel[1]
	sH, sW := stride[0], stride[1]
	pH, pW := padding[0], padding[1]
	if sH <= 0 || sW <= 0 || kH <= 0 || kW <= 0 {
		return nil, fmt.Errorf("%s: stride/kernel must be positive", op)
	}
	Hout := (H+2*pH-kH)/sH + 1
	Wout := (W+2*pW-kW)/sW + 1
	if Hout <= 0 || Wout <= 0 {
		return nil, fmt.Errorf("%s: output dims non-positive", op)
	}

	src, err := ensureContig(input)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(input.DType(), []int{N, C, Hout, Wout}, input.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: op, Dev: input.Device().Kind(), DType: input.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{src},
		Output: out,
		Attrs:  map[string]any{"kernel": kernel, "stride": stride, "padding": padding},
	}); err != nil {
		return nil, err
	}
	return out, nil
}
