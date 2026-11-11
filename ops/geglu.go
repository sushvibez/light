package ops

import (
	"fmt"

	"light/internal/registry"
	"light/tensor"
)

// GeGLU is the fused activation used by Stable Diffusion's FeedForward
// (`net.0` in HF Diffusers) and by some LLM MLPs. Given an input tensor
// of shape (..., 2*C), it splits the last axis into a `hidden` half and a
// `gate` half and returns:
//
//	out[..., i] = hidden[..., i] * GELU(gate[..., i])
//
// Output shape: (..., C). diffusers' GEGLU uses the exact (erf) GELU; the
// fused CUDA kernel uses the tanh approximation (within ~1e-3). Devices
// without a fused kernel compose Narrow + GeluErf + Mul.
func GeGLU(x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Rank() < 2 {
		return nil, fmt.Errorf("GeGLU: input rank must be ≥ 2, got %d", x.Rank())
	}
	twoC := x.Dim(-1)
	if twoC%2 != 0 {
		return nil, fmt.Errorf("GeGLU: last dim %d must be even", twoC)
	}
	src, err := ensureContig(x)
	if err != nil {
		return nil, err
	}
	dims := append([]int(nil), x.Shape().Dims()...)
	dims[len(dims)-1] = twoC / 2
	k, err := registry.Lookup(registry.Key{Op: "geglu", Dev: x.Device().Kind(), DType: x.DType()})
	if err != nil {
		h, err := x.Narrow(-1, 0, twoC/2)
		if err != nil {
			return nil, err
		}
		g, err := x.Narrow(-1, twoC/2, twoC/2)
		if err != nil {
			return nil, err
		}
		if g, err = GeluErf(g); err != nil {
			return nil, err
		}
		return Mul(h, g)
	}
	out, err := tensor.Zeros(x.DType(), dims, x.Device())
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{src},
		Output: out,
	}); err != nil {
		return nil, err
	}
	return out, nil
}
