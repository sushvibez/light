package ops

import (
	"fmt"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/cnnhooks"
	"light/tensor"
)

// Conv2dGrads is re-exported from internal/cnnhooks so callers keep using
// ops.Conv2dGrads unchanged.
type Conv2dGrads = cnnhooks.Conv2dGrads

// RegisterCUDAConv2dBackward is the stable op-level name; delegates to the
// internal hooks package which is what kernels/cuda imports directly.
func RegisterCUDAConv2dBackward(fn cnnhooks.CUDAConv2dBackwardFunc) {
	cnnhooks.RegisterConv2dBackward(fn)
}

// Conv2dBackward computes gradients for Conv2d.
//
//	input:    (N, Cin, H, W)        — forward input
//	weight:   (Cout, Cin, kH, kW)   — forward weight
//	gradOut:  (N, Cout, Hout, Wout) — grad w.r.t. forward output
//	stride:   [sH, sW]
//	padding:  [pH, pW]
//	needInput / needWeight / needBias: turn off any grad that the caller
//	does not need to cut compute.
//
// CPU F32 and CUDA F32 supported.
func Conv2dBackward(
	input, weight, gradOut *tensor.Tensor,
	stride, padding [2]int,
	needInput, needWeight, needBias bool,
) (*Conv2dGrads, error) {
	if input.Device().Kind() == device.KindCUDA {
		if cnnhooks.Conv2dBackwardCUDA == nil {
			return nil, fmt.Errorf("Conv2dBackward: CUDA hook not registered (build with -tags cuda)")
		}
		return cnnhooks.Conv2dBackwardCUDA(input, weight, gradOut, stride, padding, needInput, needWeight, needBias)
	}
	if input.Device().Kind() != device.KindCPU {
		return nil, fmt.Errorf("Conv2dBackward: unsupported device %v", input.Device())
	}
	if input.DType() != dtype.F32 || weight.DType() != dtype.F32 || gradOut.DType() != dtype.F32 {
		return nil, fmt.Errorf("Conv2dBackward: F32 only")
	}
	inC, err := ensureContig(input)
	if err != nil {
		return nil, err
	}
	wC, err := ensureContig(weight)
	if err != nil {
		return nil, err
	}
	gC, err := ensureContig(gradOut)
	if err != nil {
		return nil, err
	}

	inDims := inC.Shape().Dims()
	wDims := wC.Shape().Dims()
	gDims := gC.Shape().Dims()
	N, Cin, H, W := inDims[0], inDims[1], inDims[2], inDims[3]
	Cout, _, kH, kW := wDims[0], wDims[1], wDims[2], wDims[3]
	_, _, Hout, Wout := gDims[0], gDims[1], gDims[2], gDims[3]
	sH, sW := stride[0], stride[1]
	pH, pW := padding[0], padding[1]

	x := cpuF32Bytes(inC)
	w := cpuF32Bytes(wC)
	g := cpuF32Bytes(gC)

	var grads Conv2dGrads

	if needInput {
		gradIn, err := tensor.Zeros(dtype.F32, []int{N, Cin, H, W}, inC.Device())
		if err != nil {
			return nil, err
		}
		gi := cpuF32Bytes(gradIn)
		// For each grad_out element, scatter weighted contributions into grad_in.
		for n := range N {
			for co := range Cout {
				for oh := range Hout {
					for ow := range Wout {
						gVal := g[((n*Cout+co)*Hout+oh)*Wout+ow]
						if gVal == 0 {
							continue
						}
						for ci := range Cin {
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
									wv := w[((co*Cin+ci)*kH+i)*kW+j]
									gi[((n*Cin+ci)*H+ih)*W+iw] += wv * gVal
								}
							}
						}
					}
				}
			}
		}
		grads.GradInput = gradIn
	}

	if needWeight {
		gradW, err := tensor.Zeros(dtype.F32, []int{Cout, Cin, kH, kW}, wC.Device())
		if err != nil {
			return nil, err
		}
		gw := cpuF32Bytes(gradW)
		for n := range N {
			for co := range Cout {
				for oh := range Hout {
					for ow := range Wout {
						gVal := g[((n*Cout+co)*Hout+oh)*Wout+ow]
						if gVal == 0 {
							continue
						}
						for ci := range Cin {
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
									xv := x[((n*Cin+ci)*H+ih)*W+iw]
									gw[((co*Cin+ci)*kH+i)*kW+j] += xv * gVal
								}
							}
						}
					}
				}
			}
		}
		grads.GradWeight = gradW
	}

	if needBias {
		gradB, err := tensor.Zeros(dtype.F32, []int{Cout}, wC.Device())
		if err != nil {
			return nil, err
		}
		gb := cpuF32Bytes(gradB)
		for n := range N {
			for co := range Cout {
				var acc float32
				for oh := range Hout {
					for ow := range Wout {
						acc += g[((n*Cout+co)*Hout+oh)*Wout+ow]
					}
				}
				gb[co] += acc
			}
		}
		grads.GradBias = gradB
	}
	return &grads, nil
}

// cpuF32Bytes reinterprets a CPU F32 tensor as a []float32. The tensor must
// be contiguous.
func cpuF32Bytes(t *tensor.Tensor) []float32 {
	buf := t.Storage().CPUBytes()
	if len(buf) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&buf[0])), len(buf)/4)
}
