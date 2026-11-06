//go:build !nocuda && cgo

package cuda

import (
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
)

func init() {
	registry.Register(registry.Key{Op: "conv3d", Dev: device.KindCUDA, DType: dtype.F32}, conv3dF32CUDA)
}

// conv3dF32CUDA launches the one-thread-per-output naive kernel.
//
//	Inputs[0]: (N, Cin, D, H, W) F32   Inputs[1]: (Cout, Cin, kD, kH, kW)
//	Inputs[2]: (Cout,) bias, optional  Output:    (N, Cout, Dout, Hout, Wout)
//	Attrs: stride [3]int, padding [3]int
func conv3dF32CUDA(ctx *registry.Context) error {
	in, w, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	stride := ctx.Attrs["stride"].([3]int)
	pad := ctx.Attrs["padding"].([3]int)
	inDims := in.Shape().Dims()
	wDims := w.Shape().Dims()
	outDims := out.Shape().Dims()

	i32 := func(v int) int32 { return int32(v) }
	N, Cin, D, H, W := i32(inDims[0]), i32(inDims[1]), i32(inDims[2]), i32(inDims[3]), i32(inDims[4])
	Cout, kD, kH, kW := i32(wDims[0]), i32(wDims[2]), i32(wDims[3]), i32(wDims[4])
	Dout, Hout, Wout := i32(outDims[2]), i32(outDims[3]), i32(outDims[4])
	sD, sH, sW := i32(stride[0]), i32(stride[1]), i32(stride[2])
	pD, pH, pW := i32(pad[0]), i32(pad[1]), i32(pad[2])

	var biasPtr uintptr
	hasBias := int32(0)
	if len(ctx.Inputs) > 2 && ctx.Inputs[2] != nil {
		biasPtr = ctx.Inputs[2].Storage().CUDAPtr()
		hasBias = 1
	}
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("conv3d_f32")
	if err != nil {
		return err
	}
	inPtr := in.Storage().CUDAPtr()
	wPtr := w.Storage().CUDAPtr()
	outPtr := out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&inPtr), unsafe.Pointer(&wPtr), unsafe.Pointer(&biasPtr), unsafe.Pointer(&outPtr),
		unsafe.Pointer(&N), unsafe.Pointer(&Cin), unsafe.Pointer(&D), unsafe.Pointer(&H), unsafe.Pointer(&W),
		unsafe.Pointer(&Cout), unsafe.Pointer(&kD), unsafe.Pointer(&kH), unsafe.Pointer(&kW),
		unsafe.Pointer(&Dout), unsafe.Pointer(&Hout), unsafe.Pointer(&Wout),
		unsafe.Pointer(&sD), unsafe.Pointer(&sH), unsafe.Pointer(&sW),
		unsafe.Pointer(&pD), unsafe.Pointer(&pH), unsafe.Pointer(&pW),
		unsafe.Pointer(&hasBias),
	}
	total := uint64(out.NumElements())
	if err := Launch1D(fn, total, args); err != nil {
		return err
	}
	return nil
}
