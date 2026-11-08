package ops

import (
	"fmt"

	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// KVRingWrite writes src (B, H, T, D) into the ring buffer ring (B, H, Cap,
// D) in place: position t goes to slot (slot0 + t) mod Cap. Both F32 on the
// same device; ring contiguous; src may be any view whose last axis is
// contiguous (a slice of a fused qkv projection, say). Decode loops keep a
// fixed-size key / value cache this way instead of re-concatenating it
// every step.
func KVRingWrite(ring, src *tensor.Tensor, slot0 int) error {
	if ring.Rank() != 4 || src.Rank() != 4 || ring.DType() != dtype.F32 || src.DType() != dtype.F32 {
		return fmt.Errorf("KVRingWrite: want F32 rank-4 ring and src, got %v %v / %v %v", ring.Shape(), ring.DType(), src.Shape(), src.DType())
	}
	B, H, T, D := src.Dim(0), src.Dim(1), src.Dim(2), src.Dim(3)
	Cap := ring.Dim(2)
	if ring.Dim(0) != B || ring.Dim(1) != H || ring.Dim(3) != D || T > Cap {
		return fmt.Errorf("KVRingWrite: src %v does not fit ring %v", src.Shape(), ring.Shape())
	}
	if !ring.IsContiguous() || ring.Offset() != 0 {
		return fmt.Errorf("KVRingWrite: ring must be contiguous")
	}
	if !src.Device().SameAs(ring.Device()) {
		return fmt.Errorf("KVRingWrite: ring and src on different devices")
	}
	st := src.Strides().Slice()
	if st[3] != 1 {
		c, err := src.Contiguous()
		if err != nil {
			return err
		}
		src, st = c, c.Strides().Slice()
	}
	k, err := registry.Lookup(registry.Key{Op: "kv_ring_write", Dev: ring.Device().Kind(), DType: dtype.F32})
	if err != nil {
		return err
	}
	return k(&registry.Context{
		Inputs: []*tensor.Tensor{src},
		Output: ring,
		Attrs:  map[string]any{"slot0": (slot0%Cap + Cap) % Cap, "sb": st[0], "sh": st[1], "st": st[2]},
	})
}

// AttnDecode is single-query attention against ring caches: q (B, H, 1, D)
// (any view with a contiguous last axis), k and v (B, H, Cap, D) contiguous
// rings of which slots [0, n) are valid, all F32. It returns
// softmax(scale · q kᵀ) v as (B, H, 1, D). Slot order does not matter
// (positions are already in the keys through RoPE), so a ring that has
// wrapped needs no reordering.
func AttnDecode(q, k, v *tensor.Tensor, n int, scale float32) (*tensor.Tensor, error) {
	if q.Rank() != 4 || k.Rank() != 4 || v.Rank() != 4 || q.Dim(2) != 1 {
		return nil, fmt.Errorf("AttnDecode: want q (B, H, 1, D), k / v (B, H, Cap, D), got %v %v %v", q.Shape(), k.Shape(), v.Shape())
	}
	B, H, D, Cap := q.Dim(0), q.Dim(1), q.Dim(3), k.Dim(2)
	if !k.Shape().Equal(v.Shape()) || k.Dim(0) != B || k.Dim(1) != H || k.Dim(3) != D {
		return nil, fmt.Errorf("AttnDecode: shape mismatch q %v k %v v %v", q.Shape(), k.Shape(), v.Shape())
	}
	if n < 1 || n > Cap {
		return nil, fmt.Errorf("AttnDecode: %d valid slots of %d", n, Cap)
	}
	for _, t := range []*tensor.Tensor{q, k, v} {
		if t.DType() != dtype.F32 {
			return nil, fmt.Errorf("AttnDecode: F32 only, got %v", t.DType())
		}
		if !t.Device().SameAs(q.Device()) {
			return nil, fmt.Errorf("AttnDecode: tensors on different devices")
		}
	}
	if !k.IsContiguous() || k.Offset() != 0 || !v.IsContiguous() || v.Offset() != 0 {
		return nil, fmt.Errorf("AttnDecode: k / v rings must be contiguous")
	}
	qs := q.Strides().Slice()
	if qs[3] != 1 {
		c, err := q.Contiguous()
		if err != nil {
			return nil, err
		}
		q, qs = c, c.Strides().Slice()
	}
	out, err := tensor.Zeros(dtype.F32, []int{B, H, 1, D}, q.Device())
	if err != nil {
		return nil, err
	}
	kern, err := registry.Lookup(registry.Key{Op: "attn_decode", Dev: q.Device().Kind(), DType: dtype.F32})
	if err != nil {
		return nil, err
	}
	if err := kern(&registry.Context{
		Inputs: []*tensor.Tensor{q, k, v},
		Output: out,
		Attrs:  map[string]any{"n": n, "scale": scale, "qb": qs[0], "qh": qs[1]},
	}); err != nil {
		return nil, err
	}
	return out, nil
}
