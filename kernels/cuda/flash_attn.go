//go:build !nocuda

package cuda

import (
	"fmt"
	"sync"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
)

func init() {
	registry.Register(registry.Key{Op: "flash_attention", Dev: device.KindCUDA, DType: dtype.F32}, flashAttnF32CUDA)
}

const (
	flashMaxD      = 256   // accumulator width baked into the kernels
	flashTileQ     = 128   // query rows per block (tiled kernels)
	flashSharedCap = 32768 // bytes of shared memory the tiled kernel aims for
	flashMaxTileK  = 64
)

var (
	sharedMemMu    sync.Mutex
	sharedMemCache = map[int]int{} // ordinal → sharedMemPerBlock bytes
)

// sharedMemPerBlock queries (and caches) the device's per-block shared
// memory budget via cudart.
func sharedMemPerBlock(ordinal int) (int, error) {
	sharedMemMu.Lock()
	defer sharedMemMu.Unlock()
	if v, ok := sharedMemCache[ordinal]; ok {
		return v, nil
	}
	props, err := cudart.GetDeviceProperties(ordinal)
	if err != nil {
		return 0, err
	}
	v := int(props.SharedMemPerBlock)
	sharedMemCache[ordinal] = v
	return v, nil
}

// flashTileK picks how many keys/values a block stages in shared memory:
// 2 * tileK * D floats must fit in min(32 KiB, device limit), capped at 64.
// Returns 0 when not even one key fits (→ simple-kernel fallback).
func flashTileK(D, sharedBytes int) int {
	budget := flashSharedCap
	if sharedBytes < budget {
		budget = sharedBytes
	}
	tileK := budget / (2 * D * 4)
	if tileK > flashMaxTileK {
		tileK = flashMaxTileK
	}
	if tileK < 1 {
		return 0
	}
	return tileK
}

// flashAttnF32CUDA selects between the fixed-D tiled kernels (D ∈ {32, 64,
// 128}), the generic tiled kernel (any D <= 256), and the one-thread-per-row
// simple kernel (when shared memory is too small for a single K/V row).
//
//	Inputs[0..2]: q, k, v (BH, S, D) contiguous F32
//	Output:       (BH, S, D)
//	Attrs:        "scale" float64, "causal" bool
func flashAttnF32CUDA(ctx *registry.Context) error {
	q, k, v, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	BH, S, D := int32(q.Dim(0)), int32(q.Dim(1)), int32(q.Dim(2))
	if D > flashMaxD {
		return fmt.Errorf("cuda flash_attention: head dim %d > %d not supported", D, flashMaxD)
	}
	scale := float32(ctx.Attrs["scale"].(float64))
	causal := int32(0)
	if ctx.Attrs["causal"].(bool) {
		causal = 1
	}
	ord := out.Device().Ordinal()
	if err := cudart.SetDevice(ord); err != nil {
		return err
	}
	shared, err := sharedMemPerBlock(ord)
	if err != nil {
		return err
	}
	qPtr, kPtr, vPtr, oPtr := q.Storage().CUDAPtr(), k.Storage().CUDAPtr(), v.Storage().CUDAPtr(), out.Storage().CUDAPtr()

	tileK := int32(flashTileK(int(D), shared))
	if tileK >= 1 {
		name := "flash_attn_tiled_f32"
		var args []unsafe.Pointer
		switch D {
		case 32, 64, 128:
			name = fmt.Sprintf("flash_attn_tiled_d%d_f32", D)
			args = []unsafe.Pointer{
				unsafe.Pointer(&qPtr), unsafe.Pointer(&kPtr), unsafe.Pointer(&vPtr), unsafe.Pointer(&oPtr),
				unsafe.Pointer(&BH), unsafe.Pointer(&S), unsafe.Pointer(&scale), unsafe.Pointer(&causal),
				unsafe.Pointer(&tileK),
			}
		default:
			args = []unsafe.Pointer{
				unsafe.Pointer(&qPtr), unsafe.Pointer(&kPtr), unsafe.Pointer(&vPtr), unsafe.Pointer(&oPtr),
				unsafe.Pointer(&BH), unsafe.Pointer(&S), unsafe.Pointer(&D), unsafe.Pointer(&scale),
				unsafe.Pointer(&causal), unsafe.Pointer(&tileK),
			}
		}
		fn, err := GetKernel(name)
		if err != nil {
			return err
		}
		gridX := uint32((int(S) + flashTileQ - 1) / flashTileQ)
		if gridX == 0 {
			gridX = 1
		}
		sharedBytes := uint32(2 * int(tileK) * int(D) * 4)
		if err := cudart.LaunchKernel(fn, gridX, uint32(BH), 1, flashTileQ, 1, 1, sharedBytes, 0, args); err != nil {
			return err
		}
		return nil
	}

	// Fallback: no shared-memory tiling.
	fn, err := GetKernel("flash_attn_simple_f32")
	if err != nil {
		return err
	}
	args := []unsafe.Pointer{
		unsafe.Pointer(&qPtr), unsafe.Pointer(&kPtr), unsafe.Pointer(&vPtr), unsafe.Pointer(&oPtr),
		unsafe.Pointer(&BH), unsafe.Pointer(&S), unsafe.Pointer(&D), unsafe.Pointer(&scale), unsafe.Pointer(&causal),
	}
	if err := Launch1D(fn, uint64(BH)*uint64(S), args); err != nil {
		return err
	}
	return nil
}
