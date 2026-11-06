package ops

import (
	"fmt"
	"math"

	"light/device"
	"light/dtype"
	"light/internal/cnnhooks"
	"light/tensor"
)

// RegisterCUDAMaxPool2dBackward wires a GPU implementation.
func RegisterCUDAMaxPool2dBackward(fn cnnhooks.CUDAPool2dBackwardFunc) {
	cnnhooks.RegisterMaxPool2dBackward(fn)
}

// RegisterCUDAAvgPool2dBackward wires a GPU implementation.
func RegisterCUDAAvgPool2dBackward(fn cnnhooks.CUDAPool2dBackwardFunc) {
	cnnhooks.RegisterAvgPool2dBackward(fn)
}

// MaxPool2dBackward computes grad_input for MaxPool2d.
//
// The backward pass re-scans each window to find the arg-max (avoiding the
// need to save indices from forward). Grad from each output cell is added to
// exactly the input position that was the max. Ties go to the first position
// in row-major order (matching the forward kernel).
//
// CPU F32 only.
func MaxPool2dBackward(
	input, gradOut *tensor.Tensor,
	kernel, stride, padding [2]int,
) (*tensor.Tensor, error) {
	if input.Device().Kind() == device.KindCUDA {
		if cnnhooks.MaxPool2dBackwardCUDA == nil {
			return nil, fmt.Errorf("MaxPool2dBackward: CUDA hook not registered")
		}
		return cnnhooks.MaxPool2dBackwardCUDA(input, gradOut, kernel, stride, padding)
	}
	if input.Device().Kind() != device.KindCPU {
		return nil, fmt.Errorf("MaxPool2dBackward: unsupported device")
	}
	if input.DType() != dtype.F32 || gradOut.DType() != dtype.F32 {
		return nil, fmt.Errorf("MaxPool2dBackward: F32 only")
	}
	inC, err := ensureContig(input)
	if err != nil {
		return nil, err
	}
	gC, err := ensureContig(gradOut)
	if err != nil {
		return nil, err
	}
	inDims := inC.Shape().Dims()
	gDims := gC.Shape().Dims()
	N, C, H, W := inDims[0], inDims[1], inDims[2], inDims[3]
	_, _, Hout, Wout := gDims[0], gDims[1], gDims[2], gDims[3]
	kH, kW := kernel[0], kernel[1]
	sH, sW := stride[0], stride[1]
	pH, pW := padding[0], padding[1]

	x := cpuF32Bytes(inC)
	g := cpuF32Bytes(gC)
	gradIn, err := tensor.Zeros(dtype.F32, []int{N, C, H, W}, inC.Device())
	if err != nil {
		return nil, err
	}
	gi := cpuF32Bytes(gradIn)

	for n := range N {
		for c := range C {
			for oh := range Hout {
				for ow := range Wout {
					// Find argmax within the window (same convention as forward).
					bestVal := float32(math.Inf(-1))
					bestH, bestW := -1, -1
					for i := range kH {
						ih := oh*sH + i - pH
						if ih < 0 || ih >= H {
							continue
						}
						for j := range kW {
							iw := ow*sW + j - pW
							if iw < 0 || iw >= W {
								continue
							}
							v := x[((n*C+c)*H+ih)*W+iw]
							if bestH < 0 || v > bestVal {
								bestVal = v
								bestH = ih
								bestW = iw
							}
						}
					}
					if bestH < 0 {
						continue
					}
					gVal := g[((n*C+c)*Hout+oh)*Wout+ow]
					gi[((n*C+c)*H+bestH)*W+bestW] += gVal
				}
			}
		}
	}
	return gradIn, nil
}

// AvgPool2dBackward computes grad_input for AvgPool2d. Each output cell's
// gradient is distributed uniformly across the unpadded cells that
// contributed to its average. CPU F32 only.
func AvgPool2dBackward(
	input, gradOut *tensor.Tensor,
	kernel, stride, padding [2]int,
) (*tensor.Tensor, error) {
	if input.Device().Kind() == device.KindCUDA {
		if cnnhooks.AvgPool2dBackwardCUDA == nil {
			return nil, fmt.Errorf("AvgPool2dBackward: CUDA hook not registered")
		}
		return cnnhooks.AvgPool2dBackwardCUDA(input, gradOut, kernel, stride, padding)
	}
	if input.Device().Kind() != device.KindCPU {
		return nil, fmt.Errorf("AvgPool2dBackward: unsupported device")
	}
	if input.DType() != dtype.F32 || gradOut.DType() != dtype.F32 {
		return nil, fmt.Errorf("AvgPool2dBackward: F32 only")
	}
	inC, err := ensureContig(input)
	if err != nil {
		return nil, err
	}
	gC, err := ensureContig(gradOut)
	if err != nil {
		return nil, err
	}
	inDims := inC.Shape().Dims()
	gDims := gC.Shape().Dims()
	N, C, H, W := inDims[0], inDims[1], inDims[2], inDims[3]
	_, _, Hout, Wout := gDims[0], gDims[1], gDims[2], gDims[3]
	kH, kW := kernel[0], kernel[1]
	sH, sW := stride[0], stride[1]
	pH, pW := padding[0], padding[1]

	g := cpuF32Bytes(gC)
	gradIn, err := tensor.Zeros(dtype.F32, []int{N, C, H, W}, inC.Device())
	if err != nil {
		return nil, err
	}
	gi := cpuF32Bytes(gradIn)

	for n := range N {
		for c := range C {
			for oh := range Hout {
				for ow := range Wout {
					// Match forward: count unpadded cells in this window.
					count := 0
					for i := range kH {
						ih := oh*sH + i - pH
						if ih < 0 || ih >= H {
							continue
						}
						for j := range kW {
							iw := ow*sW + j - pW
							if iw < 0 || iw >= W {
								continue
							}
							count++
						}
					}
					if count == 0 {
						continue
					}
					gVal := g[((n*C+c)*Hout+oh)*Wout+ow] / float32(count)
					for i := range kH {
						ih := oh*sH + i - pH
						if ih < 0 || ih >= H {
							continue
						}
						for j := range kW {
							iw := ow*sW + j - pW
							if iw < 0 || iw >= W {
								continue
							}
							gi[((n*C+c)*H+ih)*W+iw] += gVal
						}
					}
				}
			}
		}
	}
	return gradIn, nil
}
