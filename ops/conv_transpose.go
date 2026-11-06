package ops

import (
	"fmt"

	"light/internal/registry"
	"light/tensor"
)

// ConvTranspose2dOpts carries the PyTorch / Candle transposed-convolution
// hyper-parameters. Zero values mean stride 1, no padding, no output
// padding, dilation 1, one group.
type ConvTranspose2dOpts struct {
	Stride        [2]int
	Padding       [2]int
	OutputPadding [2]int
	Dilation      [2]int
	Groups        int
}

func (o ConvTranspose2dOpts) norm() ConvTranspose2dOpts {
	for i := range 2 {
		if o.Stride[i] == 0 {
			o.Stride[i] = 1
		}
		if o.Dilation[i] == 0 {
			o.Dilation[i] = 1
		}
	}
	if o.Groups == 0 {
		o.Groups = 1
	}
	return o
}

// ConvTranspose2d is the gradient-of-conv2d (`nn.ConvTranspose2d`):
//
//	input:  (N, Cin, H, W)
//	weight: (Cin, Cout/groups, kH, kW)   — PyTorch layout, loads unchanged
//	bias:   (Cout,) or nil
//	output: (N, Cout, Hout, Wout), Hout = (H-1)·s - 2p + d·(k-1) + op + 1
//
// Implemented as a gather per output pixel (no atomics), F32/F64 on the
// CPU, F32/BF16 on CUDA.
func ConvTranspose2d(input, weight, bias *tensor.Tensor, opts ConvTranspose2dOpts) (*tensor.Tensor, error) {
	o := opts.norm()
	if input.Rank() != 4 || weight.Rank() != 4 {
		return nil, fmt.Errorf("ConvTranspose2d: input (N, Cin, H, W) and weight (Cin, Cout/g, kH, kW) required, got %v / %v", input.Shape(), weight.Shape())
	}
	if !input.Device().SameAs(weight.Device()) || input.DType() != weight.DType() {
		return nil, fmt.Errorf("ConvTranspose2d: input/weight device or dtype mismatch")
	}
	N, Cin, H, W := input.Dim(0), input.Dim(1), input.Dim(2), input.Dim(3)
	if weight.Dim(0) != Cin {
		return nil, fmt.Errorf("ConvTranspose2d: weight Cin %d != input Cin %d", weight.Dim(0), Cin)
	}
	if Cin%o.Groups != 0 {
		return nil, fmt.Errorf("ConvTranspose2d: Cin %d not divisible by groups %d", Cin, o.Groups)
	}
	Cout := weight.Dim(1) * o.Groups
	kH, kW := weight.Dim(2), weight.Dim(3)
	Hout := (H-1)*o.Stride[0] - 2*o.Padding[0] + o.Dilation[0]*(kH-1) + o.OutputPadding[0] + 1
	Wout := (W-1)*o.Stride[1] - 2*o.Padding[1] + o.Dilation[1]*(kW-1) + o.OutputPadding[1] + 1
	if Hout <= 0 || Wout <= 0 {
		return nil, fmt.Errorf("ConvTranspose2d: output dims non-positive (%d, %d)", Hout, Wout)
	}
	if o.OutputPadding[0] >= o.Stride[0] && o.OutputPadding[0] >= o.Dilation[0] ||
		o.OutputPadding[1] >= o.Stride[1] && o.OutputPadding[1] >= o.Dilation[1] {
		return nil, fmt.Errorf("ConvTranspose2d: output_padding must be smaller than stride or dilation")
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
			return nil, fmt.Errorf("ConvTranspose2d: bias must be (%d,)", Cout)
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
	k, err := registry.Lookup(registry.Key{Op: "conv_transpose2d", Dev: input.Device().Kind(), DType: input.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: inputs,
		Output: out,
		Attrs: map[string]any{
			"stride": o.Stride, "padding": o.Padding, "dilation": o.Dilation, "groups": o.Groups,
		},
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// ConvTranspose1d runs ConvTranspose2d on (N, Cin, L) input and
// (Cin, Cout/g, k) weight by treating the length as width.
func ConvTranspose1d(input, weight, bias *tensor.Tensor, stride, padding, outputPadding, dilation, groups int) (*tensor.Tensor, error) {
	if input.Rank() != 3 || weight.Rank() != 3 {
		return nil, fmt.Errorf("ConvTranspose1d: input (N, Cin, L) and weight (Cin, Cout/g, k) required")
	}
	src, err := ensureContig(input)
	if err != nil {
		return nil, err
	}
	x4, err := src.Reshape(input.Dim(0), input.Dim(1), 1, input.Dim(2))
	if err != nil {
		return nil, err
	}
	wC, err := ensureContig(weight)
	if err != nil {
		return nil, err
	}
	w4, err := wC.Reshape(weight.Dim(0), weight.Dim(1), 1, weight.Dim(2))
	if err != nil {
		return nil, err
	}
	out, err := ConvTranspose2d(x4, w4, bias, ConvTranspose2dOpts{
		Stride: [2]int{1, stride}, Padding: [2]int{0, padding}, OutputPadding: [2]int{0, outputPadding},
		Dilation: [2]int{1, dilation}, Groups: groups,
	})
	if err != nil {
		return nil, err
	}
	return out.Reshape(out.Dim(0), out.Dim(1), out.Dim(3))
}

// UpsampleBilinear2d resizes (N, C, H, W) to (N, C, outH, outW) with
// bilinear interpolation (`F.interpolate(mode="bilinear")`): with
// alignCorners false the source coordinate is (dst+0.5)·in/out - 0.5
// clamped at 0, otherwise dst·(in-1)/(out-1).
func UpsampleBilinear2d(input *tensor.Tensor, outH, outW int, alignCorners bool) (*tensor.Tensor, error) {
	if input.Rank() != 4 {
		return nil, fmt.Errorf("UpsampleBilinear2d: input must be (N, C, H, W)")
	}
	if outH <= 0 || outW <= 0 {
		return nil, fmt.Errorf("UpsampleBilinear2d: output size must be positive")
	}
	src, err := ensureContig(input)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(input.DType(), []int{input.Dim(0), input.Dim(1), outH, outW}, input.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "upsample_bilinear2d", Dev: input.Device().Kind(), DType: input.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{src},
		Output: out,
		Attrs:  map[string]any{"align_corners": alignCorners},
	}); err != nil {
		return nil, err
	}
	return out, nil
}
