package cpu

import (
	"light/device"
	"light/dtype"
	"light/internal/parallel"
	"light/internal/registry"
	"light/internal/simd"
)

func init() {
	reg := func(op string, dt dtype.DType, k registry.Kernel) {
		registry.Register(registry.Key{Op: op, Dev: device.KindCPU, DType: dt}, k)
	}
	reg("conv2d", dtype.F32, conv2dF32)
	reg("conv2d", dtype.F64, conv2dKernel[float64])
	reg("maxpool2d", dtype.F32, maxPool2dKernel[float32])
	reg("maxpool2d", dtype.F64, maxPool2dKernel[float64])
	reg("avgpool2d", dtype.F32, avgPool2dKernel[float32])
	reg("avgpool2d", dtype.F64, avgPool2dKernel[float64])
}

// conv2dF32 is conv2dKernel as blocked im2col + the AVX2 x · Wᵀ tiles. The
// output (Cout, Hout · Wout) is cut into tiles of convBlock pixels by
// convCoBlock channels, spread across cores (small feature maps still fill
// them through the channel split). A tile lays its pixels out as rows of
// Cin · kH · kW patch values (zero padding included) - reused across the
// channel tiles of the same pixel block - and multiplies the weight rows
// against them straight into the output. Same products, different
// summation order.
func conv2dF32(ctx *registry.Context) error {
	const convBlock, convCoBlock = 48, 64
	in, w, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	stride := ctx.Attrs["stride"].([2]int)
	pad := ctx.Attrs["padding"].([2]int)
	sH, sW, pH, pW := stride[0], stride[1], pad[0], pad[1]
	N, Cin, H, W := in.Dim(0), in.Dim(1), in.Dim(2), in.Dim(3)
	Cout, kH, kW := w.Dim(0), w.Dim(2), w.Dim(3)
	Hout, Wout := out.Dim(2), out.Dim(3)
	K, P := Cin*kH*kW, Hout*Wout
	inS, wS, outS := asSlice[float32](in), asSlice[float32](w), asSlice[float32](out)
	var bias []float32
	if len(ctx.Inputs) > 2 && ctx.Inputs[2] != nil {
		bias = asSlice[float32](ctx.Inputs[2])
	}
	nP := (P + convBlock - 1) / convBlock
	nC := (Cout + convCoBlock - 1) / convCoBlock
	for n := range N {
		src := inS[n*Cin*H*W:]
		dst := outS[n*Cout*P : (n+1)*Cout*P]
		parallel.For(nP*nC, 1, func(lo, hi int) {
			col := make([]float32, convBlock*K)
			built := -1
			for id := lo; id < hi; id++ {
				pb, cb := id/nC, id%nC
				p0 := pb * convBlock
				pc := min(convBlock, P-p0)
				if pb != built {
					for r := range pc {
						oh, ow := (p0+r)/Wout, (p0+r)%Wout
						row := col[r*K : (r+1)*K]
						i := 0
						for ci := range Cin {
							plane := src[ci*H*W:]
							for ky := range kH {
								ih := oh*sH - pH + ky
								for kx := range kW {
									iw := ow*sW - pW + kx
									if ih < 0 || ih >= H || iw < 0 || iw >= W {
										row[i] = 0
									} else {
										row[i] = plane[ih*W+iw]
									}
									i++
								}
							}
						}
					}
					built = pb
				}
				c0 := cb * convCoBlock
				cc := min(convCoBlock, Cout-c0)
				simd.MatTB(wS[c0*K:], K, col, K, dst[c0*P+p0:], P, cc, pc, K)
				if bias != nil {
					for co := c0; co < c0+cc; co++ {
						o := dst[co*P+p0 : co*P+p0+pc]
						for j := range o {
							o[j] += bias[co]
						}
					}
				}
			}
		})
	}
	return nil
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
