package ops

import (
	"fmt"

	"light/tensor"
)

// InstanceNorm2d normalises every (n, c) plane of an (N, C, H, W) tensor
// over its H*W elements, then applies the per-channel affine (gamma, beta).
// It is exactly GroupNorm with groups = C, and is implemented on top of
// that op so it inherits the existing CPU + CUDA kernels.
//
//	x:     (N, C, H, W)
//	gamma: (C,) same dtype as x
//	beta:  (C,) same dtype as x
func InstanceNorm2d(x, gamma, beta *tensor.Tensor, eps float64) (*tensor.Tensor, error) {
	if x.Rank() != 4 {
		return nil, fmt.Errorf("InstanceNorm2d: x must be (N, C, H, W), got %v", x.Shape())
	}
	return GroupNorm(x, gamma, beta, x.Dim(1), eps)
}
