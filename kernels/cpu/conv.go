package cpu

import (
	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func init() {
	reg := func(op string, dt dtype.DType, k registry.Kernel) {
		registry.Register(registry.Key{Op: op, Dev: device.KindCPU, DType: dt}, k)
	}
	reg("conv2d", dtype.F32, conv2dKernel[float32])
	reg("conv2d", dtype.F64, conv2dKernel[float64])
	reg("maxpool2d", dtype.F32, maxPool2dKernel[float32])
	reg("maxpool2d", dtype.F64, maxPool2dKernel[float64])
	reg("avgpool2d", dtype.F32, avgPool2dKernel[float32])
	reg("avgpool2d", dtype.F64, avgPool2dKernel[float64])
}

// conv2dKernel implements naive 2-D convolution:
//
//	input:  (N, Cin, H, W)
//	weight: (Cout, Cin, kH, kW)
//	bias:   (Cout,) or nil (Inputs[2] absent)
//	output: (N, Cout, Hout, Wout)
//
// Attrs: stride = [sH, sW], padding = [pH, pW].
func conv2dKernel[T dtype.Numeric](ctx *registry.Context) error {
	in := ctx.Inputs[0]
	w := ctx.Inputs[1]
	var bias *float64 // optional
	if v, ok := ctx.Attrs["bias"]; ok && v != nil {
		_ = v
	}
	hasBias := len(ctx.Inputs) > 2 && ctx.Inputs[2] != nil
	var biasS []T
	if hasBias {
		biasS = asSlice[T](ctx.Inputs[2])
	}
	out := ctx.Output
	stride := ctx.Attrs["stride"].([2]int)
	pad := ctx.Attrs["padding"].([2]int)
	sH, sW := stride[0], stride[1]
	pH, pW := pad[0], pad[1]

	inDims := in.Shape().Dims()
	N := inDims[0]
	Cin := inDims[1]
	H := inDims[2]
	W := inDims[3]
	wDims := w.Shape().Dims()
	Cout := wDims[0]
	kH := wDims[2]
	kW := wDims[3]
	outDims := out.Shape().Dims()
	Hout := outDims[2]
	Wout := outDims[3]

	inS := asSlice[T](in)
	wS := asSlice[T](w)
	outS := asSlice[T](out)

	inStrideN := Cin * H * W
	inStrideC := H * W
	wStrideCO := Cin * kH * kW
	wStrideCI := kH * kW
	outStrideN := Cout * Hout * Wout
	outStrideC := Hout * Wout

	for n := range N {
		for co := range Cout {
			var bv T
			if hasBias {
				bv = biasS[co]
			}
			for h := range Hout {
				for x := range Wout {
					acc := bv
					for ci := range Cin {
						for ky := range kH {
							for kx := range kW {
								ih := h*sH - pH + ky
								iw := x*sW - pW + kx
								if ih < 0 || ih >= H || iw < 0 || iw >= W {
									continue
								}
								inOff := n*inStrideN + ci*inStrideC + ih*W + iw
								wOff := co*wStrideCO + ci*wStrideCI + ky*kW + kx
								acc += inS[inOff] * wS[wOff]
							}
						}
					}
					outS[n*outStrideN+co*outStrideC+h*Wout+x] = acc
				}
			}
		}
	}
	_ = bias
	return nil
}

func maxPool2dKernel[T dtype.Numeric](ctx *registry.Context) error {
	return pool2dImpl[T](ctx, false)
}

func avgPool2dKernel[T dtype.Numeric](ctx *registry.Context) error {
	return pool2dImpl[T](ctx, true)
}

func pool2dImpl[T dtype.Numeric](ctx *registry.Context, avg bool) error {
	in := ctx.Inputs[0]
	out := ctx.Output
	kSize := ctx.Attrs["kernel"].([2]int)
	stride := ctx.Attrs["stride"].([2]int)
	pad := ctx.Attrs["padding"].([2]int)
	kH, kW := kSize[0], kSize[1]
	sH, sW := stride[0], stride[1]
	pH, pW := pad[0], pad[1]

	inDims := in.Shape().Dims()
	N := inDims[0]
	C := inDims[1]
	H := inDims[2]
	W := inDims[3]
	outDims := out.Shape().Dims()
	Hout := outDims[2]
	Wout := outDims[3]

	inS := asSlice[T](in)
	outS := asSlice[T](out)
	inStrideN := C * H * W
	inStrideC := H * W
	outStrideN := C * Hout * Wout
	outStrideC := Hout * Wout

	for n := range N {
		for c := range C {
			for h := range Hout {
				for x := range Wout {
					var acc T
					var count int
					first := true
					for ky := range kH {
						for kx := range kW {
							ih := h*sH - pH + ky
							iw := x*sW - pW + kx
							if ih < 0 || ih >= H || iw < 0 || iw >= W {
								continue
							}
							v := inS[n*inStrideN+c*inStrideC+ih*W+iw]
							if avg {
								acc += v
								count++
							} else {
								if first || v > acc {
									acc = v
									first = false
								}
							}
						}
					}
					if avg && count > 0 {
						acc = acc / T(count)
					}
					outS[n*outStrideN+c*outStrideC+h*Wout+x] = acc
				}
			}
		}
	}
	return nil
}
