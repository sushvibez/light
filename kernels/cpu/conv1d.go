package cpu

import (
	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func init() {
	registry.Register(registry.Key{Op: "conv1d", Dev: device.KindCPU, DType: dtype.F32}, conv1dKernel[float32])
	registry.Register(registry.Key{Op: "conv1d", Dev: device.KindCPU, DType: dtype.F64}, conv1dKernel[float64])
	registry.Register(registry.Key{Op: "upsample_nearest", Dev: device.KindCPU, DType: dtype.F32}, upsampleNearestKernel[float32])
	registry.Register(registry.Key{Op: "upsample_nearest", Dev: device.KindCPU, DType: dtype.F64}, upsampleNearestKernel[float64])
}

// conv1dKernel implements naive 1-D convolution:
//
//	input:  (N, Cin, L)
//	weight: (Cout, Cin, kL)
//	bias:   (Cout,) or nil
//	output: (N, Cout, Lout), Lout = (L + 2p - k)/s + 1
func conv1dKernel[T dtype.Numeric](ctx *registry.Context) error {
	in := ctx.Inputs[0]
	w := ctx.Inputs[1]
	hasBias := len(ctx.Inputs) > 2 && ctx.Inputs[2] != nil
	var biasS []T
	if hasBias {
		biasS = asSlice[T](ctx.Inputs[2])
	}
	out := ctx.Output
	stride := ctx.Attrs["stride"].(int)
	pad := ctx.Attrs["padding"].(int)

	inDims := in.Shape().Dims()
	N := inDims[0]
	Cin := inDims[1]
	L := inDims[2]
	wDims := w.Shape().Dims()
	Cout := wDims[0]
	kL := wDims[2]
	outDims := out.Shape().Dims()
	Lout := outDims[2]

	inS := asSlice[T](in)
	wS := asSlice[T](w)
	outS := asSlice[T](out)

	for n := range N {
		for co := range Cout {
			var bv T
			if hasBias {
				bv = biasS[co]
			}
			for o := range Lout {
				acc := bv
				for ci := range Cin {
					for k := range kL {
						pos := o*stride - pad + k
						if pos < 0 || pos >= L {
							continue
						}
						acc += inS[n*Cin*L+ci*L+pos] * wS[co*Cin*kL+ci*kL+k]
					}
				}
				outS[n*Cout*Lout+co*Lout+o] = acc
			}
		}
	}
	return nil
}

// upsampleNearestKernel applies nearest-neighbor upsampling on the trailing
// two dims of a (N, C, H, W) tensor by integer scale factors.
func upsampleNearestKernel[T dtype.Numeric](ctx *registry.Context) error {
	in := ctx.Inputs[0]
	out := ctx.Output
	scale := ctx.Attrs["scale"].([2]int)
	sH, sW := scale[0], scale[1]
	inDims := in.Shape().Dims()
	N := inDims[0]
	C := inDims[1]
	H := inDims[2]
	W := inDims[3]
	Hout := H * sH
	Wout := W * sW

	inS := asSlice[T](in)
	outS := asSlice[T](out)
	for n := range N {
		for c := range C {
			for h := range Hout {
				for x := range Wout {
					srcH := h / sH
					srcW := x / sW
					outS[n*C*Hout*Wout+c*Hout*Wout+h*Wout+x] =
						inS[n*C*H*W+c*H*W+srcH*W+srcW]
				}
			}
		}
	}
	return nil
}
