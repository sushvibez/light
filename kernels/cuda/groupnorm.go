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
	registry.Register(registry.Key{Op: "groupnorm", Dev: device.KindCUDA, DType: dtype.F32}, groupnormF32CUDA)
	registry.Register(registry.Key{Op: "groupnorm", Dev: device.KindCUDA, DType: dtype.BF16}, groupnormBF16CUDA)
}

// groupnormF32CUDA expects:
//
//	Inputs[0] (input):  (N, C, H, W) F32
//	Inputs[1] (gamma):  (C,) F32
//	Inputs[2] (beta):   (C,) F32
//	Output:             (N, C, H, W) F32
//	Attrs["groups"]:    int — number of groups G; C must be divisible by G
//	Attrs["eps"]:       float64
func groupnormF32CUDA(ctx *registry.Context) error {
	x, gamma, beta, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	G, _ := ctx.Attrs["groups"].(int)
	eps, _ := ctx.Attrs["eps"].(float64)
	epsF := float32(eps)
	if x.Rank() != 4 {
		return fmt.Errorf("cuda groupnorm: input rank %d != 4 (expect (N,C,H,W))", x.Rank())
	}
	N := uint64(x.Dim(0))
	C := uint64(x.Dim(1))
	HW := uint64(x.Dim(2) * x.Dim(3))
	if G <= 0 || int(C)%G != 0 {
		return fmt.Errorf("cuda groupnorm: C (%d) not divisible by groups (%d)", C, G)
	}
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("groupnorm_f32")
	if err != nil {
		return err
	}
	xPtr := x.Storage().CUDAPtr()
	gPtr := gamma.Storage().CUDAPtr()
	bPtr := beta.Storage().CUDAPtr()
	oPtr := out.Storage().CUDAPtr()
	gG := uint64(G)
	args := []unsafe.Pointer{
		unsafe.Pointer(&xPtr),
		unsafe.Pointer(&gPtr),
		unsafe.Pointer(&bPtr),
		unsafe.Pointer(&oPtr),
		unsafe.Pointer(&N),
		unsafe.Pointer(&C),
		unsafe.Pointer(&gG),
		unsafe.Pointer(&HW),
		unsafe.Pointer(&epsF),
	}
	const block = 256
	const sharedBytes = 2 * block * 4 // 2 * blockDim * sizeof(float)
	grid := uint32(N * gG)
	if grid == 0 {
		grid = 1
	}
	if err := cudart.LaunchKernel(fn, grid, 1, 1, block, 1, 1, sharedBytes, 0, args); err != nil {
		return err
	}
	return nil
}

func groupnormBF16CUDA(ctx *registry.Context) error {
	x, gamma, beta, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	G, _ := ctx.Attrs["groups"].(int)
	eps, _ := ctx.Attrs["eps"].(float64)
	epsF := float32(eps)
	if x.Rank() != 4 {
		return fmt.Errorf("cuda groupnorm_bf16: input rank %d != 4", x.Rank())
	}
	if gamma.DType() != dtype.BF16 || beta.DType() != dtype.BF16 {
		return fmt.Errorf("cuda groupnorm_bf16: gamma/beta must be BF16 (got %v / %v)", gamma.DType(), beta.DType())
	}
	N := uint64(x.Dim(0))
	C := uint64(x.Dim(1))
	HW := uint64(x.Dim(2) * x.Dim(3))
	if G <= 0 || int(C)%G != 0 {
		return fmt.Errorf("cuda groupnorm_bf16: C (%d) not divisible by groups (%d)", C, G)
	}
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("groupnorm_bf16")
	if err != nil {
		return err
	}
	xPtr := x.Storage().CUDAPtr()
	gPtr := gamma.Storage().CUDAPtr()
	bPtr := beta.Storage().CUDAPtr()
	oPtr := out.Storage().CUDAPtr()
	gG := uint64(G)
	args := []unsafe.Pointer{
		unsafe.Pointer(&xPtr),
		unsafe.Pointer(&gPtr),
		unsafe.Pointer(&bPtr),
		unsafe.Pointer(&oPtr),
		unsafe.Pointer(&N),
		unsafe.Pointer(&C),
		unsafe.Pointer(&gG),
		unsafe.Pointer(&HW),
		unsafe.Pointer(&epsF),
	}
	const block = 256
	const sharedBytes = 2 * block * 4
	grid := uint32(N * gG)
	if grid == 0 {
		grid = 1
	}
	if err := cudart.LaunchKernel(fn, grid, 1, 1, block, 1, 1, sharedBytes, 0, args); err != nil {
		return err
	}
	return nil
}
