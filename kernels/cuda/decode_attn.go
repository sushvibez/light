//go:build !nocuda && cgo

package cuda

import (
	"fmt"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
	"light/tensor"
)

func init() {
	for _, dt := range []dtype.DType{dtype.F32, dtype.BF16} {
		registry.Register(registry.Key{Op: "kv_append", Dev: device.KindCUDA, DType: dt}, kvAppendLauncher)
		registry.Register(registry.Key{Op: "decode_attn", Dev: device.KindCUDA, DType: dt}, decodeAttnLauncher)
	}
	registry.Register(registry.Key{Op: "inc_i64", Dev: device.KindCUDA, DType: dtype.I64}, incI64Launcher)
}

// decodeAttnKeys is DA_KEYS in decode_attn.cu: keys per split-K block.
const (
	decodeAttnKeys   = 32
	decodeAttnMaxG   = 16
	decodeAttnShared = 48 * 1024 // dynamic shared memory without opt-in
)

// DecodeAttnSupported reports whether the CUDA decode-attention kernel
// handles a head dim and GQA group: a multiple of 32, a group of at most
// 16, and the padded query rows plus scores within the default shared
// memory (Gemma 4's 512-wide global heads fit). ops falls back to the
// composed path otherwise.
func DecodeAttnSupported(headDim, group int) bool {
	return headDim%32 == 0 && group >= 1 && group <= decodeAttnMaxG && group*(headDim+1+decodeAttnKeys)*4 <= decodeAttnShared
}

func devPtr(t *tensor.Tensor) uintptr {
	return t.Storage().CUDAPtr() + uintptr(t.Offset()*t.DType().Size())
}

func kernelSuffix(dt dtype.DType) string {
	if dt == dtype.BF16 {
		return "bf16"
	}
	return "f32"
}

// kvAppendLauncher: Inputs = [k, v (B, KH, 1, D) contiguous, pos I64 (1,)],
// Output = kbuf (B, KH, Cap, D); Attrs vbuf.
func kvAppendLauncher(ctx *registry.Context) error {
	k, v, pos, kbuf := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	vbuf := ctx.Attrs["vbuf"].(*tensor.Tensor)
	if err := cudart.SetDevice(kbuf.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("kv_append_" + kernelSuffix(k.DType()))
	if err != nil {
		return err
	}
	kP, vP, kbP, vbP, pP := devPtr(k), devPtr(v), devPtr(kbuf), devPtr(vbuf), devPtr(pos)
	B, KH, D, Cap := int32(k.Dim(0)), int32(k.Dim(1)), int32(k.Dim(3)), int32(kbuf.Dim(2))
	args := []unsafe.Pointer{
		unsafe.Pointer(&kP), unsafe.Pointer(&vP), unsafe.Pointer(&kbP), unsafe.Pointer(&vbP), unsafe.Pointer(&pP),
		unsafe.Pointer(&B), unsafe.Pointer(&KH), unsafe.Pointer(&D), unsafe.Pointer(&Cap),
	}
	return Launch1D(fn, uint64(B)*uint64(KH)*uint64(D), args)
}

// decodeAttnLauncher: Inputs = [q (B, NH, 1, D) contiguous, kbuf, vbuf
// (B, KH, Cap, D), pos I64 (1,)], Output (B, NH, 1, D); Attrs scale. Two
// launches: the split-K partials over ceil(Cap / 128) key blocks per KV
// head, then the per-head combine. The grid depends on the capacity, not
// the position, so the pair captures into a graph.
func decodeAttnLauncher(ctx *registry.Context) error {
	q, kbuf, vbuf, pos, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Inputs[3], ctx.Output
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	B, NH, D := q.Dim(0), q.Dim(1), q.Dim(3)
	KH, Cap := kbuf.Dim(1), kbuf.Dim(2)
	g := NH / KH
	if !DecodeAttnSupported(D, g) {
		return fmt.Errorf("cuda decode_attn: head dim %d / group %d unsupported", D, g)
	}
	splits := (Cap + decodeAttnKeys - 1) / decodeAttnKeys
	part, err := tensor.Empty(dtype.F32, []int{B * NH * splits * (D + 2)}, out.Device())
	if err != nil {
		return err
	}
	sfx := kernelSuffix(q.DType())
	f1, err := GetKernel("decode_attn_partial_" + sfx)
	if err != nil {
		return err
	}
	f2, err := GetKernel("decode_attn_combine_" + sfx)
	if err != nil {
		return err
	}
	qP, kP, vP, pP, posP, oP := devPtr(q), devPtr(kbuf), devPtr(vbuf), devPtr(part), devPtr(pos), devPtr(out)
	nh32, kh32, cap32, d32 := int32(NH), int32(KH), int32(Cap), int32(D)
	scale := float32(ctx.Attrs["scale"].(float64))
	window := int32(ctx.Attrs["window"].(int))
	args := []unsafe.Pointer{
		unsafe.Pointer(&qP), unsafe.Pointer(&kP), unsafe.Pointer(&vP), unsafe.Pointer(&pP), unsafe.Pointer(&posP),
		unsafe.Pointer(&nh32), unsafe.Pointer(&kh32), unsafe.Pointer(&cap32), unsafe.Pointer(&d32), unsafe.Pointer(&scale),
		unsafe.Pointer(&window),
	}
	shared := uint32((g*(D+1) + g*decodeAttnKeys) * 4)
	if err := cudart.LaunchKernel(f1, uint32(B*KH), uint32(splits), 1, 128, 1, 1, shared, 0, args); err != nil {
		return err
	}
	s32 := int32(splits)
	args2 := []unsafe.Pointer{unsafe.Pointer(&pP), unsafe.Pointer(&oP), unsafe.Pointer(&posP), unsafe.Pointer(&s32), unsafe.Pointer(&d32), unsafe.Pointer(&window)}
	block := uint32(min(max(D, 32), 1024))
	return cudart.LaunchKernel(f2, uint32(B*NH), 1, 1, block, 1, 1, 0, 0, args2)
}

// incI64Launcher: Output = the I64 tensor to increment in place.
func incI64Launcher(ctx *registry.Context) error {
	t := ctx.Output
	if err := cudart.SetDevice(t.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("inc_i64")
	if err != nil {
		return err
	}
	p := devPtr(t)
	n := int32(t.NumElements())
	return Launch1D(fn, uint64(n), []unsafe.Pointer{unsafe.Pointer(&p), unsafe.Pointer(&n)})
}
