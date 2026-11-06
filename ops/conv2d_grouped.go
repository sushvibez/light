package ops

import (
	"fmt"

	"light/tensor"
)

// Conv2dGrouped runs a grouped 2-D convolution. The weight is split into
// `groups` contiguous blocks along both its Cout and Cin/groups axes; the
// input is split along Cin into the same number of groups; each group is
// convolved independently and the results are concatenated along Cout.
//
//	input:   (N, Cin, H, W)           — Cin must be divisible by groups
//	weight:  (Cout, Cin/groups, kH, kW) — Cout must be divisible by groups
//	bias:    (Cout,) or nil
//	groups:  1 (regular), Cin=Cout (depthwise), or any in-between
//
// groups=1 dispatches to the plain Conv2d. Implementation loops over groups
// in Go and composes existing Conv2d kernels, so both CPU and CUDA paths
// work automatically. Slower than a fused grouped kernel, but correct.
func Conv2dGrouped(input, weight, bias *tensor.Tensor, stride, padding [2]int, groups int) (*tensor.Tensor, error) {
	if groups == 1 {
		return Conv2d(input, weight, bias, stride, padding)
	}
	if groups <= 0 {
		return nil, fmt.Errorf("Conv2dGrouped: groups must be positive (got %d)", groups)
	}
	inDims := input.Shape().Dims()
	wDims := weight.Shape().Dims()
	if len(inDims) != 4 || len(wDims) != 4 {
		return nil, fmt.Errorf("Conv2dGrouped: input & weight must be 4-D")
	}
	Cin := inDims[1]
	Cout := wDims[0]
	if Cin%groups != 0 || Cout%groups != 0 {
		return nil, fmt.Errorf("Conv2dGrouped: Cin=%d, Cout=%d must be divisible by groups=%d", Cin, Cout, groups)
	}
	if wDims[1] != Cin/groups {
		return nil, fmt.Errorf("Conv2dGrouped: weight Cin/groups mismatch (got %d, want %d)", wDims[1], Cin/groups)
	}
	cinPerG := Cin / groups
	coutPerG := Cout / groups

	perGroup := make([]*tensor.Tensor, groups)
	for g := range groups {
		xG, err := input.Narrow(1, g*cinPerG, cinPerG)
		if err != nil {
			return nil, err
		}
		xGc, err := xG.Contiguous()
		if err != nil {
			return nil, err
		}
		wG, err := weight.Narrow(0, g*coutPerG, coutPerG)
		if err != nil {
			return nil, err
		}
		wGc, err := wG.Contiguous()
		if err != nil {
			return nil, err
		}
		var bGc *tensor.Tensor
		if bias != nil {
			bG, err := bias.Narrow(0, g*coutPerG, coutPerG)
			if err != nil {
				return nil, err
			}
			bGc, err = bG.Contiguous()
			if err != nil {
				return nil, err
			}
		}
		out, err := Conv2d(xGc, wGc, bGc, stride, padding)
		if err != nil {
			return nil, err
		}
		perGroup[g] = out
	}
	return Concat(perGroup, 1)
}

// Conv2dGroupedBackward computes (grad_input, grad_weight, grad_bias) for
// Conv2dGrouped. Mirrors the forward: split gradOut along channel axis per
// group, call the regular Conv2dBackward per group, concatenate grad_input
// along Cin and grad_weight along Cout, sum grad_bias across groups
// untouched (it's already per-Cout-channel).
func Conv2dGroupedBackward(
	input, weight, gradOut *tensor.Tensor,
	stride, padding [2]int,
	groups int,
	needInput, needWeight, needBias bool,
) (*Conv2dGrads, error) {
	if groups == 1 {
		return Conv2dBackward(input, weight, gradOut, stride, padding, needInput, needWeight, needBias)
	}
	Cin := input.Dim(1)
	Cout := weight.Dim(0)
	cinPerG := Cin / groups
	coutPerG := Cout / groups

	giParts := make([]*tensor.Tensor, groups)
	gwParts := make([]*tensor.Tensor, groups)
	gbParts := make([]*tensor.Tensor, groups)

	for g := range groups {
		xG, err := input.Narrow(1, g*cinPerG, cinPerG)
		if err != nil {
			return nil, err
		}
		xGc, err := xG.Contiguous()
		if err != nil {
			return nil, err
		}
		wG, err := weight.Narrow(0, g*coutPerG, coutPerG)
		if err != nil {
			return nil, err
		}
		wGc, err := wG.Contiguous()
		if err != nil {
			return nil, err
		}
		goG, err := gradOut.Narrow(1, g*coutPerG, coutPerG)
		if err != nil {
			return nil, err
		}
		goGc, err := goG.Contiguous()
		if err != nil {
			return nil, err
		}
		grads, err := Conv2dBackward(xGc, wGc, goGc, stride, padding, needInput, needWeight, needBias)
		if err != nil {
			return nil, err
		}
		if needInput {
			giParts[g] = grads.GradInput
		}
		if needWeight {
			gwParts[g] = grads.GradWeight
		}
		if needBias {
			gbParts[g] = grads.GradBias
		}
	}
	out := &Conv2dGrads{}
	if needInput {
		gi, err := Concat(giParts, 1)
		if err != nil {
			return nil, err
		}
		out.GradInput = gi
	}
	if needWeight {
		gw, err := Concat(gwParts, 0)
		if err != nil {
			return nil, err
		}
		out.GradWeight = gw
	}
	if needBias {
		gb, err := Concat(gbParts, 0)
		if err != nil {
			return nil, err
		}
		out.GradBias = gb
	}
	return out, nil
}
