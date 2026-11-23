package ops

import (
	"fmt"
	"math"

	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// AttentionGEMM computes softmax(Q Kᵀ / √D) V for F32 (B, NH, S, D)
// tensors on CUDA with batched GEMMs and an in-place softmax over query
// chunks (at most 512 MB of scores at a time): exact F32 like
// FlashAttention, but at cuBLAS speed, for the long unmasked sequences of
// diffusion transformers. AttentionGEMMOK reports whether it applies.
func AttentionGEMM(q, k, v *tensor.Tensor) (*tensor.Tensor, error) {
	if !AttentionGEMMOK(q, k, v) {
		return nil, fmt.Errorf("AttentionGEMM: want matching F32 (B, NH, S, D) on CUDA, got %v %v %v", q.Shape(), k.Shape(), v.Shape())
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
	out, err := tensor.Empty(dtype.F32, []int{B * NH, S, D}, q.Device())
	if err != nil {
		return nil, err
	}
	kern, err := registry.Lookup(registry.Key{Op: "attention_gemm", Dev: q.Device().Kind(), DType: dtype.F32})
	if err != nil {
		return nil, err
	}
	if err := kern(&registry.Context{Inputs: []*tensor.Tensor{q3, k3, v3}, Output: out,
		Attrs: map[string]any{"scale": 1.0 / math.Sqrt(float64(D))}}); err != nil {
		return nil, err
	}
	return out.Reshape(B, NH, S, D)
}

// AttentionGEMMOK reports whether AttentionGEMM takes these tensors.
func AttentionGEMMOK(q, k, v *tensor.Tensor) bool {
	if q.Rank() != 4 || !q.Shape().Equal(k.Shape()) || !q.Shape().Equal(v.Shape()) {
		return false
	}
	if q.DType() != dtype.F32 || k.DType() != dtype.F32 || v.DType() != dtype.F32 {
		return false
	}
	if !q.Device().SameAs(k.Device()) || !q.Device().SameAs(v.Device()) {
		return false
	}
	_, err := registry.Lookup(registry.Key{Op: "attention_gemm", Dev: q.Device().Kind(), DType: dtype.F32})
	return err == nil
}
