// Package cnnhooks holds the hook types and registration functions that let
// the CUDA build (kernels/cuda) provide GPU implementations of CNN backward
// ops without creating an import cycle with the `light/ops` package.
//
// Dependency arrows:
//
//	light/ops       → light/internal/cnnhooks
//	light/kernels/cuda → light/internal/cnnhooks
//
// cnnhooks itself only depends on light/tensor, which has no reverse-edge
// into ops or kernels/cuda.
package cnnhooks

import "light/tensor"

// Conv2dGrads bundles the three gradients produced by a Conv2d backward pass.
type Conv2dGrads struct {
	GradInput  *tensor.Tensor
	GradWeight *tensor.Tensor
	GradBias   *tensor.Tensor
}

// CUDAConv2dBackwardFunc is the signature kernels/cuda registers to run
// Conv2dBackward on GPU.
type CUDAConv2dBackwardFunc func(
	input, weight, gradOut *tensor.Tensor,
	stride, padding [2]int,
	needInput, needWeight, needBias bool,
) (*Conv2dGrads, error)

// CUDAPool2dBackwardFunc is the signature for pool (max/avg) backward.
type CUDAPool2dBackwardFunc func(
	input, gradOut *tensor.Tensor,
	kernel, stride, padding [2]int,
) (*tensor.Tensor, error)

var (
	Conv2dBackwardCUDA    CUDAConv2dBackwardFunc
	MaxPool2dBackwardCUDA CUDAPool2dBackwardFunc
	AvgPool2dBackwardCUDA CUDAPool2dBackwardFunc
)

// RegisterConv2dBackward wires a GPU implementation of Conv2d backward.
func RegisterConv2dBackward(fn CUDAConv2dBackwardFunc) { Conv2dBackwardCUDA = fn }

// RegisterMaxPool2dBackward wires a GPU implementation of MaxPool2d backward.
func RegisterMaxPool2dBackward(fn CUDAPool2dBackwardFunc) { MaxPool2dBackwardCUDA = fn }

// RegisterAvgPool2dBackward wires a GPU implementation of AvgPool2d backward.
func RegisterAvgPool2dBackward(fn CUDAPool2dBackwardFunc) { AvgPool2dBackwardCUDA = fn }

// BatchNorm2dForwardResult bundles outputs of the training-mode forward.
type BatchNorm2dForwardResult struct {
	Out   *tensor.Tensor
	XHat  *tensor.Tensor
	Sigma *tensor.Tensor // sqrt(var + eps) per channel
	Mean  *tensor.Tensor // batch mean per channel
	Var   *tensor.Tensor // biased batch variance per channel (for running-stats EMA)
}

// CUDABatchNorm2dForwardFunc is the train-mode native CUDA forward.
type CUDABatchNorm2dForwardFunc func(
	input, gamma, beta *tensor.Tensor, eps float32,
) (*BatchNorm2dForwardResult, error)

// CUDABatchNorm2dBackwardFunc is the train-mode native CUDA backward.
type CUDABatchNorm2dBackwardFunc func(
	gradOut, xHat, sigma, gamma *tensor.Tensor,
) (gradIn, gradGamma, gradBeta *tensor.Tensor, err error)

var (
	BatchNorm2dForwardCUDA  CUDABatchNorm2dForwardFunc
	BatchNorm2dBackwardCUDA CUDABatchNorm2dBackwardFunc
)

// RegisterBatchNorm2dForward wires a GPU implementation of the forward.
func RegisterBatchNorm2dForward(fn CUDABatchNorm2dForwardFunc) { BatchNorm2dForwardCUDA = fn }

// RegisterBatchNorm2dBackward wires a GPU implementation of the backward.
func RegisterBatchNorm2dBackward(fn CUDABatchNorm2dBackwardFunc) { BatchNorm2dBackwardCUDA = fn }
