//go:build !nocuda && cgo

package cuda

import (
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
	"light/tensor"
)

func init() {
	registry.Register(registry.Key{Op: "kv_ring_write", Dev: device.KindCUDA, DType: dtype.F32}, kvRingWriteLauncher)
	registry.Register(registry.Key{Op: "attn_decode", Dev: device.KindCUDA, DType: dtype.F32}, attnDecodeLauncher)
}

// attnDecodeKeys is AD_KEYS in kv_ring.cu: keys per split-K block.
const attnDecodeKeys = 128

// f32Ptr is t's device address advanced by its element offset.
func f32Ptr(t *tensor.Tensor) uintptr {
	return t.Storage().CUDAPtr() + uintptr(t.Offset()*4)
}

// kvRingWriteLauncher: ops.KVRingWrite on CUDA (kv_ring_write_f32).
func kvRingWriteLauncher(ctx *registry.Context) error {
	src, ring := ctx.Inputs[0], ctx.Output
	if err := cudart.SetDevice(ring.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("kv_ring_write_f32")
	if err != nil {
		return err
	}
	sP, dP := f32Ptr(src), f32Ptr(ring)
	B, H, T, D, Cap := int32(src.Dim(0)), int32(src.Dim(1)), int32(src.Dim(2)), int32(src.Dim(3)), int32(ring.Dim(2))
	slot0 := int32(ctx.Attrs["slot0"].(int))
	sb, sh, st := int64(ctx.Attrs["sb"].(int)), int64(ctx.Attrs["sh"].(int)), int64(ctx.Attrs["st"].(int))
	args := []unsafe.Pointer{
		unsafe.Pointer(&sP), unsafe.Pointer(&dP), unsafe.Pointer(&B), unsafe.Pointer(&H), unsafe.Pointer(&T),
		unsafe.Pointer(&D), unsafe.Pointer(&Cap), unsafe.Pointer(&slot0),
		unsafe.Pointer(&sb), unsafe.Pointer(&sh), unsafe.Pointer(&st),
	}
	n := int(B) * int(H) * int(T) * int(D)
	return cudart.LaunchKernel(fn, uint32((n+255)/256), 1, 1, 256, 1, 1, 0, 0, args)
}

// attnDecodeLauncher: ops.AttnDecode on CUDA, split-K over the valid slots
// (attn_decode_partial_f32, one block per head and 128 keys, then
// attn_decode_combine_f32 per head), so a long cache spreads over many
// blocks instead of one per head.
func attnDecodeLauncher(ctx *registry.Context) error {
	q, k, v, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	B, H, D, Cap := q.Dim(0), q.Dim(1), q.Dim(3), k.Dim(2)
	n := ctx.Attrs["n"].(int)
	splits := (n + attnDecodeKeys - 1) / attnDecodeKeys
	part, err := tensor.Zeros(dtype.F32, []int{B * H * splits * (D + 2)}, out.Device())
	if err != nil {
		return err
	}
	f1, err := GetKernel("attn_decode_partial_f32")
	if err != nil {
		return err
	}
	f2, err := GetKernel("attn_decode_combine_f32")
	if err != nil {
		return err
	}
	qP, kP, vP, pP, oP := f32Ptr(q), f32Ptr(k), f32Ptr(v), f32Ptr(part), f32Ptr(out)
	h32, cap32, d32, n32 := int32(H), int32(Cap), int32(D), int32(n)
	scale := ctx.Attrs["scale"].(float32)
	qb, qh := int64(ctx.Attrs["qb"].(int)), int64(ctx.Attrs["qh"].(int))
	args := []unsafe.Pointer{
		unsafe.Pointer(&qP), unsafe.Pointer(&kP), unsafe.Pointer(&vP), unsafe.Pointer(&pP),
		unsafe.Pointer(&h32), unsafe.Pointer(&cap32), unsafe.Pointer(&d32), unsafe.Pointer(&n32), unsafe.Pointer(&scale),
		unsafe.Pointer(&qb), unsafe.Pointer(&qh),
	}
	shared := uint32((D + attnDecodeKeys + 33) * 4)
	if err := cudart.LaunchKernel(f1, uint32(B*H), uint32(splits), 1, 128, 1, 1, shared, 0, args); err != nil {
		return err
	}
	s32 := int32(splits)
	args2 := []unsafe.Pointer{unsafe.Pointer(&pP), unsafe.Pointer(&oP), unsafe.Pointer(&s32), unsafe.Pointer(&d32)}
	block := uint32(min(max(D, 32), 1024))
	return cudart.LaunchKernel(f2, uint32(B*H), 1, 1, block, 1, 1, 0, 0, args2)
}
