package ops

import (
	"fmt"
	"math"

	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// FlashAttentionBF16 is FlashAttention on BF16 tensor cores: q, k, v are
// (B, NH, S, D) BF16 (self-attention, same length), the result is F32
// (B, NH, S, D), softmax(q·kᵀ/√D)·v with F32 accumulation. CUDA only, head
// dim 128 (the diffusion transformers'); FlashAttentionBF16OK says when it
// applies, so callers can keep their F32 path otherwise.
func FlashAttentionBF16(q, k, v *tensor.Tensor, causal bool) (*tensor.Tensor, error) {
	if !FlashAttentionBF16OK(q, k, v) {
		return nil, fmt.Errorf("FlashAttentionBF16: want (B, NH, S, 128) BF16 q, k, v on one CUDA device, got %v %v %v %v / %v / %v",
			q.Shape(), q.DType(), q.Device(), k.Shape(), v.Shape(), k.DType())
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
	out, err := tensor.Empty(dtype.F32, []int{B * NH, S, D}, q.Device()) // every element is written
	if err != nil {
		return nil, err
	}
	kern, err := registry.Lookup(registry.Key{Op: "flash_attention_bf16", Dev: q.Device().Kind(), DType: dtype.BF16})
	if err != nil {
		return nil, err
	}
	if err := kern(&registry.Context{
		Inputs: []*tensor.Tensor{q3, k3, v3},
		Output: out,
		Attrs:  map[string]any{"scale": 1.0 / math.Sqrt(float64(D)), "causal": causal},
	}); err != nil {
		return nil, err
	}
	return out.Reshape(B, NH, S, D)
}

// FlashAttentionBF16OK reports whether FlashAttentionBF16 runs on these
// operands: rank-4 BF16 tensors of one shape with head dim 128, on a device
// with the kernel.
func FlashAttentionBF16OK(q, k, v *tensor.Tensor) bool {
	if q.Rank() != 4 || !q.Shape().Equal(k.Shape()) || !q.Shape().Equal(v.Shape()) || q.Dim(3) != 128 {
		return false
	}
	if q.DType() != dtype.BF16 || k.DType() != dtype.BF16 || v.DType() != dtype.BF16 {
		return false
	}
	if !q.Device().SameAs(k.Device()) || !q.Device().SameAs(v.Device()) {
		return false
	}
	_, err := registry.Lookup(registry.Key{Op: "flash_attention_bf16", Dev: q.Device().Kind(), DType: dtype.BF16})
	return err == nil
}
