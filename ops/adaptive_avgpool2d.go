package ops

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// adaptiveWindow returns the (start, end) input range for output position `oi`
// when mapping an input of length `L` to an output of length `OL`, matching
// PyTorch's AdaptiveAvgPool convention:
//
//	start = floor(oi     * L / OL)
//	end   = ceil ((oi+1) * L / OL)
func adaptiveWindow(oi, L, OL int) (int, int) {
	start := oi * L / OL
	end := ((oi+1)*L + OL - 1) / OL
	if end > L {
		end = L
	}
	return start, end
}

// AdaptiveAvgPool2d pools a 4-D (N, C, H, W) tensor to (N, C, OH, OW) by
// averaging each adaptive window. CPU F32 native; CUDA uses a transparent
// CPU fallback (stage inputs on CPU, compute, transfer back). Output shape
// is fixed by the caller; the kernel size and stride are computed per
// output cell.
func AdaptiveAvgPool2d(input *tensor.Tensor, outH, outW int) (*tensor.Tensor, error) {
	if input.Rank() != 4 {
		return nil, fmt.Errorf("AdaptiveAvgPool2d: input must be (N, C, H, W); got rank %d", input.Rank())
	}
	if outH <= 0 || outW <= 0 {
		return nil, fmt.Errorf("AdaptiveAvgPool2d: output dims must be positive")
	}
	if input.DType() != dtype.F32 {
		return nil, fmt.Errorf("AdaptiveAvgPool2d: F32 only (got %v)", input.DType())
	}
	if input.Device().Kind() != device.KindCPU {
		// Native kernel where registered (WebGPU), else CPU round trip.
		if k, err := registry.Lookup(registry.Key{Op: "adaptive_avgpool2d", Dev: input.Device().Kind(), DType: input.DType()}); err == nil {
			inC, err := ensureContig(input)
			if err != nil {
				return nil, err
			}
			dims := inC.Shape().Dims()
			out, err := tensor.Zeros(dtype.F32, []int{dims[0], dims[1], outH, outW}, inC.Device())
			if err != nil {
				return nil, err
			}
			if err := k(&registry.Context{Inputs: []*tensor.Tensor{inC}, Output: out}); err != nil {
				return nil, err
			}
			return out, nil
		}
		return adaptiveAvgPool2dCUDAFallback(input, outH, outW)
	}
	inC, err := ensureContig(input)
	if err != nil {
		return nil, err
	}
	dims := inC.Shape().Dims()
	N, C, H, W := dims[0], dims[1], dims[2], dims[3]
	out, err := tensor.Zeros(dtype.F32, []int{N, C, outH, outW}, inC.Device())
	if err != nil {
		return nil, err
	}
	x := cpuF32Bytes(inC)
	y := cpuF32Bytes(out)
	for n := range N {
		for c := range C {
			base := (n*C + c) * H * W
			oBase := (n*C + c) * outH * outW
			for oh := range outH {
				hStart, hEnd := adaptiveWindow(oh, H, outH)
				for ow := range outW {
					wStart, wEnd := adaptiveWindow(ow, W, outW)
					var sum float32
					count := (hEnd - hStart) * (wEnd - wStart)
					for h := hStart; h < hEnd; h++ {
						for w := wStart; w < wEnd; w++ {
							sum += x[base+h*W+w]
						}
					}
					if count > 0 {
						y[oBase+oh*outW+ow] = sum / float32(count)
					}
				}
			}
		}
	}
	return out, nil
}

// AdaptiveAvgPool2dBackward computes grad_input given grad_out with the
// inverse of the adaptive-window averaging: each input cell that contributed
// to an output cell receives grad_out / count.
func AdaptiveAvgPool2dBackward(input, gradOut *tensor.Tensor, outH, outW int) (*tensor.Tensor, error) {
	if input.Device().Kind() != device.KindCPU {
		return adaptiveAvgPool2dBackwardCUDAFallback(input, gradOut, outH, outW)
	}
	inC, err := ensureContig(input)
	if err != nil {
		return nil, err
	}
	gC, err := ensureContig(gradOut)
	if err != nil {
		return nil, err
	}
	dims := inC.Shape().Dims()
	N, C, H, W := dims[0], dims[1], dims[2], dims[3]
	g := cpuF32Bytes(gC)
	gi, err := tensor.Zeros(dtype.F32, []int{N, C, H, W}, inC.Device())
	if err != nil {
		return nil, err
	}
	giB := cpuF32Bytes(gi)
	for n := range N {
		for c := range C {
			base := (n*C + c) * H * W
			oBase := (n*C + c) * outH * outW
			for oh := range outH {
				hStart, hEnd := adaptiveWindow(oh, H, outH)
				for ow := range outW {
					wStart, wEnd := adaptiveWindow(ow, W, outW)
					count := (hEnd - hStart) * (wEnd - wStart)
					if count == 0 {
						continue
					}
					gv := g[oBase+oh*outW+ow] / float32(count)
					for h := hStart; h < hEnd; h++ {
						for w := wStart; w < wEnd; w++ {
							giB[base+h*W+w] += gv
						}
					}
				}
			}
		}
	}
	return gi, nil
}

// adaptiveAvgPool2dCUDAFallback stages the CUDA input on CPU, runs the CPU
// forward, and transfers the result back.
func adaptiveAvgPool2dCUDAFallback(input *tensor.Tensor, outH, outW int) (*tensor.Tensor, error) {
	cpu := device.CPU()
	gpu := input.Device()
	inCPU, err := input.To(cpu)
	if err != nil {
		return nil, err
	}
	resCPU, err := AdaptiveAvgPool2d(inCPU, outH, outW)
	if err != nil {
		return nil, err
	}
	return resCPU.To(gpu)
}

func adaptiveAvgPool2dBackwardCUDAFallback(input, gradOut *tensor.Tensor, outH, outW int) (*tensor.Tensor, error) {
	cpu := device.CPU()
	gpu := input.Device()
	inCPU, err := input.To(cpu)
	if err != nil {
		return nil, err
	}
	gCPU, err := gradOut.To(cpu)
	if err != nil {
		return nil, err
	}
	res, err := AdaptiveAvgPool2dBackward(inCPU, gCPU, outH, outW)
	if err != nil {
		return nil, err
	}
	return res.To(gpu)
}
