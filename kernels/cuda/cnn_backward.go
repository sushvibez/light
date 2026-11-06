//go:build !nocuda

package cuda

import (
	"fmt"
	"unsafe"

	"light/dtype"
	"light/internal/cnnhooks"
	"light/internal/cudart"
	"light/tensor"
)

func init() {
	cnnhooks.RegisterConv2dBackward(conv2dBackwardCUDA)
	cnnhooks.RegisterMaxPool2dBackward(maxPool2dBackwardCUDA)
	cnnhooks.RegisterAvgPool2dBackward(avgPool2dBackwardCUDA)
}

// launchPool2dBackward launches a CUDA pool-backward kernel sized to the
// input tensor (one thread per input cell). Used by both max and avg.
func launchPool2dBackward(
	kname string,
	input, gradOut *tensor.Tensor,
	kernel, stride, padding [2]int,
) (*tensor.Tensor, error) {
	if input.DType() != dtype.F32 || gradOut.DType() != dtype.F32 {
		return nil, fmt.Errorf("cuda pool backward: F32 only")
	}
	inDims := input.Shape().Dims()
	goDims := gradOut.Shape().Dims()
	N, C, H, W := uint64(inDims[0]), uint64(inDims[1]), uint64(inDims[2]), uint64(inDims[3])
	Hout, Wout := uint64(goDims[2]), uint64(goDims[3])
	kH, kW := int32(kernel[0]), int32(kernel[1])
	sH, sW := int32(stride[0]), int32(stride[1])
	pH, pW := int32(padding[0]), int32(padding[1])

	gradIn, err := tensor.Zeros(dtype.F32, []int{int(N), int(C), int(H), int(W)}, input.Device())
	if err != nil {
		return nil, err
	}
	if err := cudart.SetDevice(input.Device().Ordinal()); err != nil {
		return nil, err
	}
	fn, err := GetKernel(kname)
	if err != nil {
		return nil, err
	}
	inPtr := input.Storage().CUDAPtr()
	goPtr := gradOut.Storage().CUDAPtr()
	giPtr := gradIn.Storage().CUDAPtr()
	// Max uses input; Avg doesn't. Both kernels accept the same arg order
	// modulo a leading pointer that the avg kernel ignores; we use separate
	// arg arrays.
	args := []unsafe.Pointer{}
	if kname == "maxpool2d_backward_f32" {
		args = []unsafe.Pointer{
			unsafe.Pointer(&inPtr), unsafe.Pointer(&goPtr), unsafe.Pointer(&giPtr),
			unsafe.Pointer(&N), unsafe.Pointer(&C), unsafe.Pointer(&H), unsafe.Pointer(&W),
			unsafe.Pointer(&Hout), unsafe.Pointer(&Wout),
			unsafe.Pointer(&kH), unsafe.Pointer(&kW),
			unsafe.Pointer(&sH), unsafe.Pointer(&sW),
			unsafe.Pointer(&pH), unsafe.Pointer(&pW),
		}
	} else {
		args = []unsafe.Pointer{
			unsafe.Pointer(&goPtr), unsafe.Pointer(&giPtr),
			unsafe.Pointer(&N), unsafe.Pointer(&C), unsafe.Pointer(&H), unsafe.Pointer(&W),
			unsafe.Pointer(&Hout), unsafe.Pointer(&Wout),
			unsafe.Pointer(&kH), unsafe.Pointer(&kW),
			unsafe.Pointer(&sH), unsafe.Pointer(&sW),
			unsafe.Pointer(&pH), unsafe.Pointer(&pW),
		}
	}
	n := N * C * H * W
	if err := Launch1D(fn, n, args); err != nil {
		return nil, err
	}
	return gradIn, nil
}

func maxPool2dBackwardCUDA(input, gradOut *tensor.Tensor, kernel, stride, padding [2]int) (*tensor.Tensor, error) {
	return launchPool2dBackward("maxpool2d_backward_f32", input, gradOut, kernel, stride, padding)
}

func avgPool2dBackwardCUDA(input, gradOut *tensor.Tensor, kernel, stride, padding [2]int) (*tensor.Tensor, error) {
	return launchPool2dBackward("avgpool2d_backward_f32", input, gradOut, kernel, stride, padding)
}

// conv2dBackwardCUDA runs three kernels for grad_input / grad_weight / grad_bias.
func conv2dBackwardCUDA(
	input, weight, gradOut *tensor.Tensor,
	stride, padding [2]int,
	needInput, needWeight, needBias bool,
) (*cnnhooks.Conv2dGrads, error) {
	if input.DType() != dtype.F32 || weight.DType() != dtype.F32 || gradOut.DType() != dtype.F32 {
		return nil, fmt.Errorf("cuda conv2d backward: F32 only")
	}
	inDims := input.Shape().Dims()
	wDims := weight.Shape().Dims()
	goDims := gradOut.Shape().Dims()
	N := uint64(inDims[0])
	Cin := uint64(inDims[1])
	H := uint64(inDims[2])
	W := uint64(inDims[3])
	Cout := uint64(wDims[0])
	kH := int32(wDims[2])
	kW := int32(wDims[3])
	Hout := uint64(goDims[2])
	Wout := uint64(goDims[3])
	sH := int32(stride[0])
	sW := int32(stride[1])
	pH := int32(padding[0])
	pW := int32(padding[1])

	if err := cudart.SetDevice(input.Device().Ordinal()); err != nil {
		return nil, err
	}

	goPtr := gradOut.Storage().CUDAPtr()
	wPtr := weight.Storage().CUDAPtr()
	inPtr := input.Storage().CUDAPtr()

	var grads cnnhooks.Conv2dGrads

	if needInput {
		gIn, err := tensor.Zeros(dtype.F32, []int{int(N), int(Cin), int(H), int(W)}, input.Device())
		if err != nil {
			return nil, err
		}
		giPtr := gIn.Storage().CUDAPtr()
		fn, err := GetKernel("conv2d_grad_input_f32")
		if err != nil {
			return nil, err
		}
		args := []unsafe.Pointer{
			unsafe.Pointer(&wPtr), unsafe.Pointer(&goPtr), unsafe.Pointer(&giPtr),
			unsafe.Pointer(&N), unsafe.Pointer(&Cin), unsafe.Pointer(&H), unsafe.Pointer(&W),
			unsafe.Pointer(&Cout), unsafe.Pointer(&Hout), unsafe.Pointer(&Wout),
			unsafe.Pointer(&kH), unsafe.Pointer(&kW),
			unsafe.Pointer(&sH), unsafe.Pointer(&sW),
			unsafe.Pointer(&pH), unsafe.Pointer(&pW),
		}
		if err := Launch1D(fn, N*Cin*H*W, args); err != nil {
			return nil, err
		}
		grads.GradInput = gIn
	}

	if needWeight {
		gW, err := tensor.Zeros(dtype.F32, []int{int(Cout), int(Cin), int(kH), int(kW)}, weight.Device())
		if err != nil {
			return nil, err
		}
		gwPtr := gW.Storage().CUDAPtr()
		fn, err := GetKernel("conv2d_grad_weight_f32")
		if err != nil {
			return nil, err
		}
		args := []unsafe.Pointer{
			unsafe.Pointer(&inPtr), unsafe.Pointer(&goPtr), unsafe.Pointer(&gwPtr),
			unsafe.Pointer(&N), unsafe.Pointer(&Cin), unsafe.Pointer(&H), unsafe.Pointer(&W),
			unsafe.Pointer(&Cout), unsafe.Pointer(&Hout), unsafe.Pointer(&Wout),
			unsafe.Pointer(&kH), unsafe.Pointer(&kW),
			unsafe.Pointer(&sH), unsafe.Pointer(&sW),
			unsafe.Pointer(&pH), unsafe.Pointer(&pW),
		}
		if err := Launch1D(fn, Cout*Cin*uint64(kH)*uint64(kW), args); err != nil {
			return nil, err
		}
		grads.GradWeight = gW
	}

	if needBias {
		gB, err := tensor.Zeros(dtype.F32, []int{int(Cout)}, weight.Device())
		if err != nil {
			return nil, err
		}
		gbPtr := gB.Storage().CUDAPtr()
		fn, err := GetKernel("conv2d_grad_bias_f32")
		if err != nil {
			return nil, err
		}
		args := []unsafe.Pointer{
			unsafe.Pointer(&goPtr), unsafe.Pointer(&gbPtr),
			unsafe.Pointer(&N), unsafe.Pointer(&Cout), unsafe.Pointer(&Hout), unsafe.Pointer(&Wout),
		}
		if err := Launch1D(fn, Cout, args); err != nil {
			return nil, err
		}
		grads.GradBias = gB
	}

	return &grads, nil
}
