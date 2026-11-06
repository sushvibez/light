//go:build !nocuda

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
	registry.Register(registry.Key{Op: "rope_scaled", Dev: device.KindCUDA, DType: dtype.F32}, ropeScaledLauncher("rope_scaled_f32"))
	registry.Register(registry.Key{Op: "rope_scaled", Dev: device.KindCUDA, DType: dtype.BF16}, ropeScaledLauncher("rope_scaled_bf16"))
}

// ropeScaledLauncher runs rope_scaled_{f32,bf16}: Inputs = [x, positions
// (I64), invFreq (F32, headDim/2)], all contiguous and on the same device.
func ropeScaledLauncher(kname string) registry.Kernel {
	return func(ctx *registry.Context) error {
		x, positions, invFreq, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
		if positions.DType() != dtype.I64 {
			return fmt.Errorf("cuda %s: positions must be I64", kname)
		}
		if invFreq.DType() != dtype.F32 {
			return fmt.Errorf("cuda %s: invFreq must be F32", kname)
		}
		headDim := uint64(x.Dim(-1))
		seqLen := uint64(x.Dim(-2))
		rows := uint64(x.NumElements()) / headDim
		if uint64(invFreq.NumElements()) != headDim/2 {
			return fmt.Errorf("cuda %s: invFreq has %d entries, want headDim/2 = %d", kname, invFreq.NumElements(), headDim/2)
		}
		if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
			return err
		}
		fn, err := GetKernel(kname)
		if err != nil {
			return err
		}
		xPtr := x.Storage().CUDAPtr()
		posPtr := positions.Storage().CUDAPtr()
		freqPtr := invFreq.Storage().CUDAPtr()
		outPtr := out.Storage().CUDAPtr()
		args := []unsafe.Pointer{
			unsafe.Pointer(&xPtr),
			unsafe.Pointer(&posPtr),
			unsafe.Pointer(&freqPtr),
			unsafe.Pointer(&outPtr),
			unsafe.Pointer(&rows),
			unsafe.Pointer(&seqLen),
			unsafe.Pointer(&headDim),
		}
		if err := Launch1D(fn, rows*(headDim/2), args); err != nil {
			return err
		}
		return nil
	}
}
