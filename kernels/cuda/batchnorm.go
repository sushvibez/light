//go:build !nocuda && cgo

package cuda

import (
	"unsafe"

	"light/dtype"
	"light/internal/cnnhooks"
	"light/internal/cudart"
	"light/tensor"
)

func init() {
	cnnhooks.RegisterBatchNorm2dForward(batchNorm2dForwardCUDA)
	cnnhooks.RegisterBatchNorm2dBackward(batchNorm2dBackwardCUDA)
}

const bnBlock = 256
const bnSharedFloats = bnBlock * 2 // two arrays in shared memory (sum, sumSq)

// batchNorm2dForwardCUDA runs the train-mode forward as a 2-kernel pipeline:
// per-channel stats reduction + per-element apply.
func batchNorm2dForwardCUDA(
	input, gamma, beta *tensor.Tensor, eps float32,
) (*cnnhooks.BatchNorm2dForwardResult, error) {
	dims := input.Shape().Dims()
	N, C, H, W := uint64(dims[0]), uint64(dims[1]), uint64(dims[2]), uint64(dims[3])
	HW := H * W
	dev := input.Device()
	if err := cudart.SetDevice(dev.Ordinal()); err != nil {
		return nil, err
	}
	meanT, err := tensor.Zeros(dtype.F32, []int{int(C)}, dev)
	if err != nil {
		return nil, err
	}
	varT, err := tensor.Zeros(dtype.F32, []int{int(C)}, dev)
	if err != nil {
		return nil, err
	}
	sigmaT, err := tensor.Zeros(dtype.F32, []int{int(C)}, dev)
	if err != nil {
		return nil, err
	}
	xHat, err := tensor.Zeros(dtype.F32, dims, dev)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(dtype.F32, dims, dev)
	if err != nil {
		return nil, err
	}

	// Kernel 1: per-channel mean, variance, sigma.
	fnStats, err := GetKernel("batchnorm2d_stats_f32")
	if err != nil {
		return nil, err
	}
	inPtr := input.Storage().CUDAPtr()
	meanPtr := meanT.Storage().CUDAPtr()
	varPtr := varT.Storage().CUDAPtr()
	sigmaPtr := sigmaT.Storage().CUDAPtr()
	statsArgs := []unsafe.Pointer{
		unsafe.Pointer(&inPtr), unsafe.Pointer(&meanPtr), unsafe.Pointer(&varPtr),
		unsafe.Pointer(&sigmaPtr), unsafe.Pointer(&eps),
		unsafe.Pointer(&N), unsafe.Pointer(&C), unsafe.Pointer(&HW),
	}
	sharedBytes := uint32(bnSharedFloats * 4)
	if err := cudart.LaunchKernel(fnStats, uint32(C), 1, 1, bnBlock, 1, 1, sharedBytes, 0, statsArgs); err != nil {
		return nil, err
	}

	// Kernel 2: apply per-element (reads sigma from pass 1).
	fnApply, err := GetKernel("batchnorm2d_apply_f32")
	if err != nil {
		return nil, err
	}
	gammaPtr := gamma.Storage().CUDAPtr()
	betaPtr := beta.Storage().CUDAPtr()
	outPtr := out.Storage().CUDAPtr()
	xHatPtr := xHat.Storage().CUDAPtr()
	applyArgs := []unsafe.Pointer{
		unsafe.Pointer(&inPtr), unsafe.Pointer(&meanPtr), unsafe.Pointer(&sigmaPtr),
		unsafe.Pointer(&gammaPtr), unsafe.Pointer(&betaPtr),
		unsafe.Pointer(&outPtr), unsafe.Pointer(&xHatPtr),
		unsafe.Pointer(&N), unsafe.Pointer(&C), unsafe.Pointer(&HW),
	}
	if err := Launch1D(fnApply, N*C*HW, applyArgs); err != nil {
		return nil, err
	}
	return &cnnhooks.BatchNorm2dForwardResult{
		Out: out, XHat: xHat, Sigma: sigmaT, Mean: meanT, Var: varT,
	}, nil
}

// batchNorm2dBackwardCUDA runs the train-mode backward as a 2-kernel
// pipeline: per-channel stats reduction (sumG, sumG·xHat) + per-element
// apply. grad_gamma and grad_beta are just the per-channel sums, so we
// alias the stats buffers for them.
func batchNorm2dBackwardCUDA(
	gradOut, xHat, sigma, gamma *tensor.Tensor,
) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, error) {
	dims := gradOut.Shape().Dims()
	N, C, H, W := uint64(dims[0]), uint64(dims[1]), uint64(dims[2]), uint64(dims[3])
	HW := H * W
	M := float32(N * HW)
	dev := gradOut.Device()
	if err := cudart.SetDevice(dev.Ordinal()); err != nil {
		return nil, nil, nil, err
	}

	sumG, err := tensor.Zeros(dtype.F32, []int{int(C)}, dev)
	if err != nil {
		return nil, nil, nil, err
	}
	sumGxHat, err := tensor.Zeros(dtype.F32, []int{int(C)}, dev)
	if err != nil {
		return nil, nil, nil, err
	}
	gradIn, err := tensor.Zeros(dtype.F32, dims, dev)
	if err != nil {
		return nil, nil, nil, err
	}

	fnStats, err := GetKernel("batchnorm2d_backward_stats_f32")
	if err != nil {
		return nil, nil, nil, err
	}
	gPtr := gradOut.Storage().CUDAPtr()
	xhPtr := xHat.Storage().CUDAPtr()
	sgPtr := sumG.Storage().CUDAPtr()
	sgxhPtr := sumGxHat.Storage().CUDAPtr()
	statsArgs := []unsafe.Pointer{
		unsafe.Pointer(&gPtr), unsafe.Pointer(&xhPtr),
		unsafe.Pointer(&sgPtr), unsafe.Pointer(&sgxhPtr),
		unsafe.Pointer(&N), unsafe.Pointer(&C), unsafe.Pointer(&HW),
	}
	sharedBytes := uint32(bnSharedFloats * 4)
	if err := cudart.LaunchKernel(fnStats, uint32(C), 1, 1, bnBlock, 1, 1, sharedBytes, 0, statsArgs); err != nil {
		return nil, nil, nil, err
	}

	fnApply, err := GetKernel("batchnorm2d_backward_apply_f32")
	if err != nil {
		return nil, nil, nil, err
	}
	sigmaPtr := sigma.Storage().CUDAPtr()
	gammaPtr := gamma.Storage().CUDAPtr()
	gInPtr := gradIn.Storage().CUDAPtr()
	applyArgs := []unsafe.Pointer{
		unsafe.Pointer(&gPtr), unsafe.Pointer(&xhPtr),
		unsafe.Pointer(&sigmaPtr), unsafe.Pointer(&gammaPtr),
		unsafe.Pointer(&sgPtr), unsafe.Pointer(&sgxhPtr),
		unsafe.Pointer(&gInPtr),
		unsafe.Pointer(&M),
		unsafe.Pointer(&N), unsafe.Pointer(&C), unsafe.Pointer(&HW),
	}
	if err := Launch1D(fnApply, N*C*HW, applyArgs); err != nil {
		return nil, nil, nil, err
	}
	// grad_gamma = sum(g * xHat); grad_beta = sum(g). These are exactly the
	// stats buffers — no further work needed.
	return gradIn, sumGxHat, sumG, nil
}
