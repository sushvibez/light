package cpu

import (
	"math"

	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func init() {
	reg := func(op string, dt dtype.DType, k registry.Kernel) {
		registry.Register(registry.Key{Op: op, Dev: device.KindCPU, DType: dt}, k)
	}
	reg("conv_transpose2d", dtype.F32, convTranspose2dKernel[float32])
	reg("conv_transpose2d", dtype.F64, convTranspose2dKernel[float64])
	reg("upsample_bilinear2d", dtype.F32, upsampleBilinear2dKernel[float32])
	reg("upsample_bilinear2d", dtype.F64, upsampleBilinear2dKernel[float64])
}

// convTranspose2dKernel gathers, per output pixel, every (input pixel,
// kernel tap) pair that maps onto it:
//
//	oh = ih·sH - pH + ky·dH  ⇔  ih = (oh + pH - ky·dH) / sH when divisible
//
// weight is (Cin, Cout/g, kH, kW); output channel co belongs to group
// g = co / (Cout/g) and reads input channels [g·Cin/g, (g+1)·Cin/g).
func convTranspose2dKernel[T dtype.Numeric](ctx *registry.Context) error {
	in, w, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	var biasS []T
	if len(ctx.Inputs) > 2 && ctx.Inputs[2] != nil {
		biasS = asSlice[T](ctx.Inputs[2])
	}
	stride := ctx.Attrs["stride"].([2]int)
	pad := ctx.Attrs["padding"].([2]int)
	dil := ctx.Attrs["dilation"].([2]int)
	groups := ctx.Attrs["groups"].(int)
	inD, wD, outD := in.Shape().Dims(), w.Shape().Dims(), out.Shape().Dims()
	N, Cin, H, W := inD[0], inD[1], inD[2], inD[3]
	CoutG, kH, kW := wD[1], wD[2], wD[3]
	Cout, Hout, Wout := outD[1], outD[2], outD[3]
	CinG := Cin / groups
	inS, wS, outS := asSlice[T](in), asSlice[T](w), asSlice[T](out)
	for n := range N {
		for co := range Cout {
			g := co / CoutG
			cog := co % CoutG
			var bv T
			if biasS != nil {
				bv = biasS[co]
			}
			for oh := range Hout {
				for ow := range Wout {
					acc := bv
					for ky := range kH {
						num := oh + pad[0] - ky*dil[0]
						if num < 0 || num%stride[0] != 0 {
							continue
						}
						ih := num / stride[0]
						if ih >= H {
							continue
						}
						for kx := range kW {
							numw := ow + pad[1] - kx*dil[1]
							if numw < 0 || numw%stride[1] != 0 {
								continue
							}
							iw := numw / stride[1]
							if iw >= W {
								continue
							}
							for cig := range CinG {
								ci := g*CinG + cig
								acc += inS[((n*Cin+ci)*H+ih)*W+iw] * wS[((ci*CoutG+cog)*kH+ky)*kW+kx]
							}
						}
					}
					outS[((n*Cout+co)*Hout+oh)*Wout+ow] = acc
				}
			}
		}
	}
	return nil
}

// bilinearSource returns the two source indices and the interpolation
// weight for destination index d (PyTorch semantics).
func bilinearSource(d, in, out int, alignCorners bool) (i0, i1 int, lambda float64) {
	var src float64
	if alignCorners {
		if out > 1 {
			src = float64(d) * float64(in-1) / float64(out-1)
		}
	} else {
		src = (float64(d)+0.5)*float64(in)/float64(out) - 0.5
		if src < 0 {
			src = 0
		}
	}
	i0 = int(math.Floor(src))
	if i0 > in-1 {
		i0 = in - 1
	}
	i1 = min(i0+1, in-1)
	return i0, i1, src - float64(i0)
}

func upsampleBilinear2dKernel[T dtype.Numeric](ctx *registry.Context) error {
	in, out := ctx.Inputs[0], ctx.Output
	align := ctx.Attrs["align_corners"].(bool)
	inD, outD := in.Shape().Dims(), out.Shape().Dims()
	N, C, H, W := inD[0], inD[1], inD[2], inD[3]
	Hout, Wout := outD[2], outD[3]
	inS, outS := asSlice[T](in), asSlice[T](out)
	for oh := range Hout {
		h0, h1, lh := bilinearSource(oh, H, Hout, align)
		for ow := range Wout {
			w0, w1, lw := bilinearSource(ow, W, Wout, align)
			for nc := range N * C {
				base := nc * H * W
				v := (1-lh)*((1-lw)*float64(inS[base+h0*W+w0])+lw*float64(inS[base+h0*W+w1])) +
					lh*((1-lw)*float64(inS[base+h1*W+w0])+lw*float64(inS[base+h1*W+w1]))
				outS[nc*Hout*Wout+oh*Wout+ow] = T(v)
			}
		}
	}
	return nil
}
