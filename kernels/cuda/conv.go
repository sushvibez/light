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
	registry.Register(registry.Key{Op: "conv2d", Dev: device.KindCUDA, DType: dtype.F32}, conv2dF32Launcher)
	registry.Register(registry.Key{Op: "conv2d", Dev: device.KindCUDA, DType: dtype.BF16}, conv2dBF16Launcher)
	registry.Register(registry.Key{Op: "conv1d", Dev: device.KindCUDA, DType: dtype.F32}, conv1dF32Launcher)
	registry.Register(registry.Key{Op: "maxpool2d", Dev: device.KindCUDA, DType: dtype.F32}, pool2dLauncher("maxpool2d_f32"))
	registry.Register(registry.Key{Op: "avgpool2d", Dev: device.KindCUDA, DType: dtype.F32}, pool2dLauncher("avgpool2d_f32"))
	registry.Register(registry.Key{Op: "upsample_nearest", Dev: device.KindCUDA, DType: dtype.F32}, upsampleNearestLauncher)
	registry.Register(registry.Key{Op: "upsample_nearest", Dev: device.KindCUDA, DType: dtype.BF16}, upsampleNearestBF16Launcher)
	for _, dt := range []dtype.DType{dtype.F32, dtype.BF16} {
		sfx := map[dtype.DType]string{dtype.F32: "f32", dtype.BF16: "bf16"}[dt]
		registry.Register(registry.Key{Op: "conv_transpose2d", Dev: device.KindCUDA, DType: dt}, convTranspose2dLauncher("conv_transpose2d_"+sfx))
		registry.Register(registry.Key{Op: "upsample_bilinear2d", Dev: device.KindCUDA, DType: dt}, upsampleBilinear2dLauncher("upsample_bilinear2d_"+sfx))
	}
}

// conv2dBF16Launcher dispatches BF16 conv2d through cuDNN. cuDNN picks an
// implicit-GEMM or Winograd algorithm via heuristic and runs BF16 in / F32
// accumulate / BF16 out on tensor cores. Bias is added by cuDNN's
// `cudnnAddTensor` after conv.
func conv2dBF16Launcher(ctx *registry.Context) error {
	return conv2dCudnnLauncher(ctx, cudart.DnnBF16, dtype.BF16)
}

func upsampleNearestBF16Launcher(ctx *registry.Context) error {
	in := ctx.Inputs[0]
	out := ctx.Output
	scale := ctx.Attrs["scale"].([2]int)
	sH, sW := int32(scale[0]), int32(scale[1])
	inDims := in.Shape().Dims()
	outDims := out.Shape().Dims()
	N := uint64(inDims[0])
	C := uint64(inDims[1])
	H := uint64(inDims[2])
	W := uint64(inDims[3])
	Hout := uint64(outDims[2])
	Wout := uint64(outDims[3])
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("upsample_nearest_bf16")
	if err != nil {
		return err
	}
	inPtr := in.Storage().CUDAPtr()
	outPtr := out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&inPtr), unsafe.Pointer(&outPtr),
		unsafe.Pointer(&N), unsafe.Pointer(&C), unsafe.Pointer(&H), unsafe.Pointer(&W),
		unsafe.Pointer(&Hout), unsafe.Pointer(&Wout),
		unsafe.Pointer(&sH), unsafe.Pointer(&sW),
	}
	total := N * C * Hout * Wout
	if err := Launch1D(fn, total, args); err != nil {
		return err
	}
	return nil
}

func conv1dF32Launcher(ctx *registry.Context) error {
	in := ctx.Inputs[0]
	w := ctx.Inputs[1]
	out := ctx.Output
	stride := int32(ctx.Attrs["stride"].(int))
	pad := int32(ctx.Attrs["padding"].(int))
	inDims := in.Shape().Dims()
	wDims := w.Shape().Dims()
	outDims := out.Shape().Dims()
	N := uint64(inDims[0])
	Cin := uint64(inDims[1])
	L := uint64(inDims[2])
	Cout := uint64(wDims[0])
	kL := uint64(wDims[2])
	Lout := uint64(outDims[2])
	var biasPtr uintptr
	hasBias := int32(0)
	if len(ctx.Inputs) > 2 && ctx.Inputs[2] != nil {
		biasPtr = ctx.Inputs[2].Storage().CUDAPtr()
		hasBias = 1
	}
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("conv1d_f32")
	if err != nil {
		return err
	}
	inPtr := in.Storage().CUDAPtr()
	wPtr := w.Storage().CUDAPtr()
	outPtr := out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&inPtr),
		unsafe.Pointer(&wPtr),
		unsafe.Pointer(&biasPtr),
		unsafe.Pointer(&outPtr),
		unsafe.Pointer(&N),
		unsafe.Pointer(&Cin),
		unsafe.Pointer(&L),
		unsafe.Pointer(&Cout),
		unsafe.Pointer(&kL),
		unsafe.Pointer(&Lout),
		unsafe.Pointer(&stride),
		unsafe.Pointer(&pad),
		unsafe.Pointer(&hasBias),
	}
	total := N * Cout * Lout
	if err := Launch1D(fn, total, args); err != nil {
		return err
	}
	return nil
}

func upsampleNearestLauncher(ctx *registry.Context) error {
	in := ctx.Inputs[0]
	out := ctx.Output
	scale := ctx.Attrs["scale"].([2]int)
	sH, sW := int32(scale[0]), int32(scale[1])
	inDims := in.Shape().Dims()
	outDims := out.Shape().Dims()
	N := uint64(inDims[0])
	C := uint64(inDims[1])
	H := uint64(inDims[2])
	W := uint64(inDims[3])
	Hout := uint64(outDims[2])
	Wout := uint64(outDims[3])
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("upsample_nearest_f32")
	if err != nil {
		return err
	}
	inPtr := in.Storage().CUDAPtr()
	outPtr := out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&inPtr),
		unsafe.Pointer(&outPtr),
		unsafe.Pointer(&N),
		unsafe.Pointer(&C),
		unsafe.Pointer(&H),
		unsafe.Pointer(&W),
		unsafe.Pointer(&Hout),
		unsafe.Pointer(&Wout),
		unsafe.Pointer(&sH),
		unsafe.Pointer(&sW),
	}
	total := N * C * Hout * Wout
	if err := Launch1D(fn, total, args); err != nil {
		return err
	}
	return nil
}

// conv2dF32Launcher dispatches F32 conv2d through cuDNN. Same heuristic
// algorithm pick as the BF16 path; tensor cores aren't enabled on F32 since
// CUDNN_TENSOR_OP_MATH would silently downcast filter to BF16/TF32.
func conv2dF32Launcher(ctx *registry.Context) error {
	return conv2dCudnnLauncher(ctx, cudart.DnnF32, dtype.F32)
}

// conv2dCudnnLauncher is the shared cuDNN dispatch used by both F32 and
// BF16 conv2d launchers. It validates dtypes, fetches the per-device cuDNN
// handle, and forwards to cudart.Conv2DForward.
func conv2dCudnnLauncher(ctx *registry.Context, dt cudart.DnnDType, expect dtype.DType) error {
	in := ctx.Inputs[0]
	w := ctx.Inputs[1]
	out := ctx.Output
	stride := ctx.Attrs["stride"].([2]int)
	pad := ctx.Attrs["padding"].([2]int)
	if in.DType() != expect || w.DType() != expect || out.DType() != expect {
		return fmt.Errorf("cuda conv2d %v: dtype mismatch (in=%v w=%v out=%v)", expect, in.DType(), w.DType(), out.DType())
	}
	inDims := in.Shape().Dims()
	wDims := w.Shape().Dims()
	outDims := out.Shape().Dims()
	N := inDims[0]
	Cin := inDims[1]
	H := inDims[2]
	W := inDims[3]
	Cout := wDims[0]
	kH := wDims[2]
	kW := wDims[3]
	Hout := outDims[2]
	Wout := outDims[3]
	var biasPtr uintptr
	if len(ctx.Inputs) > 2 && ctx.Inputs[2] != nil {
		if ctx.Inputs[2].DType() != expect {
			return fmt.Errorf("cuda conv2d %v: bias dtype %v != %v", expect, ctx.Inputs[2].DType(), expect)
		}
		biasPtr = ctx.Inputs[2].Storage().CUDAPtr()
	}
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	h, err := cudart.GetDnnHandle(out.Device().Ordinal())
	if err != nil {
		return err
	}
	return cudart.Conv2DForward(
		h, dt,
		in.Storage().CUDAPtr(), w.Storage().CUDAPtr(), biasPtr, out.Storage().CUDAPtr(),
		N, Cin, H, W,
		Cout, kH, kW,
		Hout, Wout,
		stride[0], stride[1], pad[0], pad[1],
	)
}

func pool2dLauncher(kname string) registry.Kernel {
	return func(ctx *registry.Context) error {
		in := ctx.Inputs[0]
		out := ctx.Output
		kSize := ctx.Attrs["kernel"].([2]int)
		stride := ctx.Attrs["stride"].([2]int)
		pad := ctx.Attrs["padding"].([2]int)
		inDims := in.Shape().Dims()
		outDims := out.Shape().Dims()
		N := uint64(inDims[0])
		C := uint64(inDims[1])
		H := uint64(inDims[2])
		W := uint64(inDims[3])
		Hout := uint64(outDims[2])
		Wout := uint64(outDims[3])
		kH, kW := int32(kSize[0]), int32(kSize[1])
		sH, sW := int32(stride[0]), int32(stride[1])
		pH, pW := int32(pad[0]), int32(pad[1])

		if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
			return err
		}
		fn, err := GetKernel(kname)
		if err != nil {
			return err
		}
		inPtr := in.Storage().CUDAPtr()
		outPtr := out.Storage().CUDAPtr()
		args := []unsafe.Pointer{
			unsafe.Pointer(&inPtr),
			unsafe.Pointer(&outPtr),
			unsafe.Pointer(&N),
			unsafe.Pointer(&C),
			unsafe.Pointer(&H),
			unsafe.Pointer(&W),
			unsafe.Pointer(&Hout),
			unsafe.Pointer(&Wout),
			unsafe.Pointer(&kH),
			unsafe.Pointer(&kW),
			unsafe.Pointer(&sH),
			unsafe.Pointer(&sW),
			unsafe.Pointer(&pH),
			unsafe.Pointer(&pW),
		}
		total := N * C * Hout * Wout
		if err := Launch1D(fn, total, args); err != nil {
			return err
		}
		return nil
	}
}

// ---- conv_transpose2d + upsample_bilinear2d (src/conv_transpose.cu) ----

func convTranspose2dLauncher(name string) registry.Kernel {
	return func(ctx *registry.Context) error {
		in, w, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
		stride := ctx.Attrs["stride"].([2]int)
		pad := ctx.Attrs["padding"].([2]int)
		dil := ctx.Attrs["dilation"].([2]int)
		groups := int32(ctx.Attrs["groups"].(int))
		inD, wD, outD := in.Shape().Dims(), w.Shape().Dims(), out.Shape().Dims()
		N, Cin, H, W := uint64(inD[0]), uint64(inD[1]), uint64(inD[2]), uint64(inD[3])
		kH, kW := uint64(wD[2]), uint64(wD[3])
		Cout, Hout, Wout := uint64(outD[1]), uint64(outD[2]), uint64(outD[3])
		sH, sW := int32(stride[0]), int32(stride[1])
		pH, pW := int32(pad[0]), int32(pad[1])
		dH, dW := int32(dil[0]), int32(dil[1])
		hasBias := int32(0)
		biasPtr := out.Storage().CUDAPtr() // placeholder when no bias
		if len(ctx.Inputs) > 2 && ctx.Inputs[2] != nil {
			hasBias = 1
			biasPtr = ctx.Inputs[2].Storage().CUDAPtr()
		}
		if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
			return err
		}
		fn, err := GetKernel(name)
		if err != nil {
			return err
		}
		inPtr, wPtr, outPtr := in.Storage().CUDAPtr(), w.Storage().CUDAPtr(), out.Storage().CUDAPtr()
		args := []unsafe.Pointer{
			unsafe.Pointer(&inPtr), unsafe.Pointer(&wPtr), unsafe.Pointer(&biasPtr), unsafe.Pointer(&outPtr),
			unsafe.Pointer(&N), unsafe.Pointer(&Cin), unsafe.Pointer(&H), unsafe.Pointer(&W),
			unsafe.Pointer(&Cout), unsafe.Pointer(&kH), unsafe.Pointer(&kW),
			unsafe.Pointer(&Hout), unsafe.Pointer(&Wout),
			unsafe.Pointer(&sH), unsafe.Pointer(&sW), unsafe.Pointer(&pH), unsafe.Pointer(&pW),
			unsafe.Pointer(&dH), unsafe.Pointer(&dW), unsafe.Pointer(&groups), unsafe.Pointer(&hasBias),
		}
		if err := Launch1D(fn, N*Cout*Hout*Wout, args); err != nil {
			return err
		}
		return nil
	}
}

func upsampleBilinear2dLauncher(name string) registry.Kernel {
	return func(ctx *registry.Context) error {
		in, out := ctx.Inputs[0], ctx.Output
		align := int32(0)
		if ctx.Attrs["align_corners"].(bool) {
			align = 1
		}
		inD, outD := in.Shape().Dims(), out.Shape().Dims()
		N, C, H, W := uint64(inD[0]), uint64(inD[1]), uint64(inD[2]), uint64(inD[3])
		Hout, Wout := uint64(outD[2]), uint64(outD[3])
		if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
			return err
		}
		fn, err := GetKernel(name)
		if err != nil {
			return err
		}
		inPtr, outPtr := in.Storage().CUDAPtr(), out.Storage().CUDAPtr()
		args := []unsafe.Pointer{
			unsafe.Pointer(&inPtr), unsafe.Pointer(&outPtr),
			unsafe.Pointer(&N), unsafe.Pointer(&C), unsafe.Pointer(&H), unsafe.Pointer(&W),
			unsafe.Pointer(&Hout), unsafe.Pointer(&Wout), unsafe.Pointer(&align),
		}
		if err := Launch1D(fn, N*C*Hout*Wout, args); err != nil {
			return err
		}
		return nil
	}
}
