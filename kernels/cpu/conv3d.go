package cpu

import (
	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func init() {
	registry.Register(registry.Key{Op: "conv3d", Dev: device.KindCPU, DType: dtype.F32}, conv3dKernel[float32])
	registry.Register(registry.Key{Op: "conv3d", Dev: device.KindCPU, DType: dtype.F64}, conv3dKernel[float64])
}

// conv3dKernel implements naive 3-D convolution (cross-correlation):
//
//	input:  (N, Cin, D, H, W)
//	weight: (Cout, Cin, kD, kH, kW)
//	bias:   (Cout,) or absent (Inputs[2] missing)
//	output: (N, Cout, Dout, Hout, Wout)
//
// Attrs: stride = [sD, sH, sW], padding = [pD, pH, pW].
func conv3dKernel[T dtype.Numeric](ctx *registry.Context) error {
	in := ctx.Inputs[0]
	w := ctx.Inputs[1]
	out := ctx.Output
	var biasS []T
	if len(ctx.Inputs) > 2 && ctx.Inputs[2] != nil {
		biasS = asSlice[T](ctx.Inputs[2])
	}
	stride := ctx.Attrs["stride"].([3]int)
	pad := ctx.Attrs["padding"].([3]int)

	inDims := in.Shape().Dims()
	N, Cin, D, H, W := inDims[0], inDims[1], inDims[2], inDims[3], inDims[4]
	wDims := w.Shape().Dims()
	Cout, kD, kH, kW := wDims[0], wDims[2], wDims[3], wDims[4]
	outDims := out.Shape().Dims()
	Dout, Hout, Wout := outDims[2], outDims[3], outDims[4]

	inS := asSlice[T](in)
	wS := asSlice[T](w)
	outS := asSlice[T](out)

	for n := range N {
		for co := range Cout {
			var bv T
			if biasS != nil {
				bv = biasS[co]
			}
			for od := range Dout {
				for oh := range Hout {
					for ow := range Wout {
						acc := bv
						for ci := range Cin {
							inBase := (n*Cin + ci) * D * H * W
							wBase := (co*Cin + ci) * kD * kH * kW
							for kd := range kD {
								id := od*stride[0] - pad[0] + kd
								if id < 0 || id >= D {
									continue
								}
								for kh := range kH {
									ih := oh*stride[1] - pad[1] + kh
									if ih < 0 || ih >= H {
										continue
									}
									for kw := range kW {
										iw := ow*stride[2] - pad[2] + kw
										if iw < 0 || iw >= W {
											continue
										}
										acc += inS[inBase+(id*H+ih)*W+iw] * wS[wBase+(kd*kH+kh)*kW+kw]
									}
								}
							}
						}
						outS[(((n*Cout+co)*Dout+od)*Hout+oh)*Wout+ow] = acc
					}
				}
			}
		}
	}
	return nil
}
