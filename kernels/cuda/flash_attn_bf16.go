//go:build !nocuda && cgo

package cuda

import (
	"fmt"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
)

func init() {
	registry.Register(registry.Key{Op: "flash_attention_bf16", Dev: device.KindCUDA, DType: dtype.BF16}, flashAttnBF16CUDA)
	registry.Register(registry.Key{Op: "flash_attention_bf16", Dev: device.KindCUDA, DType: dtype.F16}, flashAttnBF16CUDA)
}

const (
	flashBF16TileQ   = 128 // query rows per block: 8 warps x 16 (BF16, head dim 64), 4 x 32 (head dim 128, F16)
	flashBF16Threads = 256
	flashMT2Threads  = 128
)

// flashAttnBF16CUDA launches flash_attn_bf16_d128 (BF16 tensor cores,
// FlashAttention-2 tiling; kernels/cuda/src/flash_attn_bf16.cu).
//
//	Inputs[0..2]: q, k, v contiguous BF16 (or F16), (BH, S, 128), or (B, S, H, 128)
//	              with "seq_major"; k, v may have Sk != S rows (F16)
//	Output:       F32 (F16 for F16 inputs), same layout
//	Attrs:        "scale" float64, "causal" bool, "seq_major" bool, "heads" int
//	              (H, for seq_major)
func flashAttnBF16CUDA(ctx *registry.Context) error {
	q, k, v, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	BH, S, D := q.Dim(0), int32(q.Dim(1)), q.Dim(q.Rank()-1)
	Sk := int32(k.Dim(1)) // key rows: S but for cross attention
	H, seqMajor := int32(1), int32(0)
	if sm, _ := ctx.Attrs["seq_major"].(bool); sm {
		// (B, S, H, D): a block per (query tile, batch-head) as before.
		H, seqMajor = int32(q.Dim(2)), 1
		BH, S = q.Dim(0)*q.Dim(2), int32(q.Dim(1))
	}
	if D != 64 && D != 128 {
		return fmt.Errorf("cuda flash_attention_bf16: head dim %d, want 64 or 128", D)
	}
	if !(q.DType() == dtype.BF16 && out.DType() == dtype.F32) && !(q.DType() == dtype.F16 && out.DType() == dtype.F16) {
		return fmt.Errorf("cuda flash_attention_bf16: want BF16 q, k, v with F32 out or F16 with F16, got %v, %v", q.DType(), out.DType())
	}
	if BH > 65535 {
		return fmt.Errorf("cuda flash_attention_bf16: %d batch-heads exceed the grid's y limit", BH)
	}
	scale := float32(ctx.Attrs["scale"].(float64))
	causal := int32(0)
	if ctx.Attrs["causal"].(bool) {
		causal = 1
	}
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	name := fmt.Sprintf("flash_attn_bf16_d%d", D)
	if q.DType() == dtype.F16 {
		name = fmt.Sprintf("flash_attn_f16_d%d_h", D) // F16 in and out
	}
	threads := uint32(flashBF16Threads)
	if q.DType() == dtype.F16 || D == 128 {
		threads = flashMT2Threads // 4 warps x 32 rows
	}
	fn, err := GetKernel(name)
	if err != nil {
		return err
	}
	qPtr, kPtr, vPtr, oPtr := q.Storage().CUDAPtr(), k.Storage().CUDAPtr(), v.Storage().CUDAPtr(), out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&qPtr), unsafe.Pointer(&kPtr), unsafe.Pointer(&vPtr), unsafe.Pointer(&oPtr),
		unsafe.Pointer(&S), unsafe.Pointer(&Sk), unsafe.Pointer(&H), unsafe.Pointer(&seqMajor), unsafe.Pointer(&scale), unsafe.Pointer(&causal),
	}
	gridX := uint32((int(S) + flashBF16TileQ - 1) / flashBF16TileQ)
	if gridX == 0 {
		return nil
	}
	return cudart.LaunchKernel(fn, gridX, uint32(BH), 1, threads, 1, 1, 0, 0, args)
}
