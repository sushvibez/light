package ops

import (
	"fmt"
	"math"

	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// FlashAttention computes softmax(Q Kᵀ / √D [+ causal mask]) V without
// materialising the (S, S) score matrix: every query row streams over the
// keys with an online softmax (running max / running denominator).
//
//	q, k, v: (B, NH, S, D), same dtype and device
//	causal:  mask out keys j > i
//	returns  (B, NH, S, D)
//
// Numerically it matches nn.SDPA with a causal mask (to ~1e-5 in F32).
// CPU supports F32/F64; CUDA supports F32 with D <= 256 and picks between a
// shared-memory tiled kernel and a simple one-thread-per-query kernel based
// on the device's shared-memory budget.
func FlashAttention(q, k, v *tensor.Tensor, causal bool) (*tensor.Tensor, error) {
	for name, t := range map[string]*tensor.Tensor{"q": q, "k": k, "v": v} {
		if t.Rank() != 4 {
			return nil, fmt.Errorf("FlashAttention: %s must be (B, NH, S, D), got %v", name, t.Shape())
		}
	}
	if !q.Shape().Equal(k.Shape()) || !q.Shape().Equal(v.Shape()) {
		return nil, fmt.Errorf("FlashAttention: q/k/v shapes must match (%v, %v, %v)", q.Shape(), k.Shape(), v.Shape())
	}
	if q.DType() != k.DType() || q.DType() != v.DType() {
		return nil, fmt.Errorf("FlashAttention: dtype mismatch")
	}
	if !q.Device().SameAs(k.Device()) || !q.Device().SameAs(v.Device()) {
		return nil, fmt.Errorf("FlashAttention: device mismatch")
	}
	if q.DType() != dtype.F32 && q.DType() != dtype.F64 {
		return nil, fmt.Errorf("FlashAttention: dtype %v not supported (F32/F64)", q.DType())
	}
	B, NH, S, D := q.Dim(0), q.Dim(1), q.Dim(2), q.Dim(3)
	flat := func(t *tensor.Tensor) (*tensor.Tensor, error) {
		c, err := ensureContig(t)
		if err != nil {
			return nil, err
		}
		return c.Reshape(B*NH, S, D)
	}
	q3, err := flat(q)
	if err != nil {
		return nil, err
	}
	k3, err := flat(k)
	if err != nil {
		return nil, err
	}
	v3, err := flat(v)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(q.DType(), []int{B * NH, S, D}, q.Device())
	if err != nil {
		return nil, err
	}
	kern, err := registry.Lookup(registry.Key{Op: "flash_attention", Dev: q.Device().Kind(), DType: q.DType()})
	if err != nil {
		return nil, err
	}
	if err := kern(&registry.Context{
		Inputs: []*tensor.Tensor{q3, k3, v3},
		Output: out,
		Attrs: map[string]any{
			"scale":  1.0 / math.Sqrt(float64(D)),
			"causal": causal,
		},
	}); err != nil {
		return nil, err
	}
	return out.Reshape(B, NH, S, D)
}
