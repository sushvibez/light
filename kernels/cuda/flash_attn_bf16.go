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
}

const (
	flashBF16D       = 128 // head dim baked into flash_attn_bf16_d128
	flashBF16TileQ   = 128 // query rows per block (8 warps x 16)
	flashBF16Threads = 256
)

// flashAttnBF16CUDA launches flash_attn_bf16_d128 (BF16 tensor cores,
// FlashAttention-2 tiling; kernels/cuda/src/flash_attn_bf16.cu).
//
//	Inputs[0..2]: q, k, v (BH, S, 128) contiguous BF16
//	Output:       (BH, S, 128) F32
//	Attrs:        "scale" float64, "causal" bool
func flashAttnBF16CUDA(ctx *registry.Context) error {
	q, k, v, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	BH, S, D := q.Dim(0), int32(q.Dim(1)), q.Dim(2)
	if D != flashBF16D {
		return fmt.Errorf("cuda flash_attention_bf16: head dim %d, want %d", D, flashBF16D)
	}
	if q.DType() != dtype.BF16 || out.DType() != dtype.F32 {
		return fmt.Errorf("cuda flash_attention_bf16: want BF16 q, k, v and F32 out, got %v, %v", q.DType(), out.DType())
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
	fn, err := GetKernel("flash_attn_bf16_d128")
	if err != nil {
		return err
	}
	qPtr, kPtr, vPtr, oPtr := q.Storage().CUDAPtr(), k.Storage().CUDAPtr(), v.Storage().CUDAPtr(), out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&qPtr), unsafe.Pointer(&kPtr), unsafe.Pointer(&vPtr), unsafe.Pointer(&oPtr),
		unsafe.Pointer(&S), unsafe.Pointer(&scale), unsafe.Pointer(&causal),
	}
	gridX := uint32((int(S) + flashBF16TileQ - 1) / flashBF16TileQ)
	if gridX == 0 {
		return nil
	}
	return cudart.LaunchKernel(fn, gridX, uint32(BH), 1, flashBF16Threads, 1, 1, 0, 0, args)
}
