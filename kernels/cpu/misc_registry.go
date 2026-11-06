package cpu

import (
	"math"

	"light/device"
	"light/dtype"
	"light/internal/registry"
	"math/rand/v2"
)

func init() {
	registry.Register(registry.Key{Op: "rope", Dev: device.KindCPU, DType: dtype.F32}, ropeF32CPU)
	registry.Register(registry.Key{Op: "rope", Dev: device.KindCPU, DType: dtype.F64}, ropeF64CPU)
	registry.Register(registry.Key{Op: "rope_i", Dev: device.KindCPU, DType: dtype.F32}, ropeIF32CPU)
	registry.Register(registry.Key{Op: "rope_i", Dev: device.KindCPU, DType: dtype.F64}, ropeIF64CPU)
	registry.Register(registry.Key{Op: "rope_i", Dev: device.KindCPU, DType: dtype.BF16}, ropeIBF16CPU)
	registry.Register(registry.Key{Op: "dropout", Dev: device.KindCPU, DType: dtype.F32}, dropoutF32CPU)
	registry.Register(registry.Key{Op: "dropout", Dev: device.KindCPU, DType: dtype.F64}, dropoutF64CPU)
	registry.Register(registry.Key{Op: "tril", Dev: device.KindCPU, DType: dtype.F32}, triangleCPU[float32](true))
	registry.Register(registry.Key{Op: "tril", Dev: device.KindCPU, DType: dtype.F64}, triangleCPU[float64](true))
	registry.Register(registry.Key{Op: "triu", Dev: device.KindCPU, DType: dtype.F32}, triangleCPU[float32](false))
	registry.Register(registry.Key{Op: "triu", Dev: device.KindCPU, DType: dtype.F64}, triangleCPU[float64](false))
}

// ropeF32CPU / ropeF64CPU apply RoPE in the rotate-half (GPT-NeoX / HF
// Llama) convention: dim i pairs with dim i+headDim/2 at angle
// pos * base^(-2i/headDim). This is what every HuggingFace Llama-family
// checkpoint (Llama, Qwen, Mistral, Phi, Gemma) was trained with; GPT-J
// style interleaved pairs (2i, 2i+1) are NOT supported.
func ropeF32CPU(ctx *registry.Context) error {
	x := ctx.Inputs[0]
	positions := ctx.Inputs[1]
	out := ctx.Output
	base := ctx.Attrs["base"].(float64)
	xS := asSlice[float32](x)
	posS := asSlice[int64](positions)
	outS := asSlice[float32](out)
	headDim := x.Dim(-1)
	seqLen := x.Dim(-2)
	batch := x.NumElements() / (seqLen * headDim)
	half := headDim / 2
	for b := range batch {
		for s := range seqLen {
			pos := float64(posS[s])
			rowBase := (b*seqLen + s) * headDim
			for i := 0; i < half; i++ {
				theta := pos * math.Pow(base, -float64(2*i)/float64(headDim))
				c := float32(math.Cos(theta))
				sn := float32(math.Sin(theta))
				x0 := xS[rowBase+i]
				x1 := xS[rowBase+i+half]
				outS[rowBase+i] = x0*c - x1*sn
				outS[rowBase+i+half] = x0*sn + x1*c
			}
		}
	}
	return nil
}

func ropeF64CPU(ctx *registry.Context) error {
	x := ctx.Inputs[0]
	positions := ctx.Inputs[1]
	out := ctx.Output
	base := ctx.Attrs["base"].(float64)
	xS := asSlice[float64](x)
	posS := asSlice[int64](positions)
	outS := asSlice[float64](out)
	headDim := x.Dim(-1)
	seqLen := x.Dim(-2)
	batch := x.NumElements() / (seqLen * headDim)
	half := headDim / 2
	for b := range batch {
		for s := range seqLen {
			pos := float64(posS[s])
			rowBase := (b*seqLen + s) * headDim
			for i := 0; i < half; i++ {
				theta := pos * math.Pow(base, -float64(2*i)/float64(headDim))
				c := math.Cos(theta)
				sn := math.Sin(theta)
				x0 := xS[rowBase+i]
				x1 := xS[rowBase+i+half]
				outS[rowBase+i] = x0*c - x1*sn
				outS[rowBase+i+half] = x0*sn + x1*c
			}
		}
	}
	return nil
}

func dropoutF32CPU(ctx *registry.Context) error {
	x, out := ctx.Inputs[0], ctx.Output
	p := ctx.Attrs["p"].(float64)
	scale := float32(ctx.Attrs["scale"].(float64))
	xS := asSlice[float32](x)
	outS := asSlice[float32](out)
	for i := range xS {
		if rand.Float64() < p {
			outS[i] = 0
		} else {
			outS[i] = xS[i] * scale
		}
	}
	return nil
}

func dropoutF64CPU(ctx *registry.Context) error {
	x, out := ctx.Inputs[0], ctx.Output
	p := ctx.Attrs["p"].(float64)
	scale := ctx.Attrs["scale"].(float64)
	xS := asSlice[float64](x)
	outS := asSlice[float64](out)
	for i := range xS {
		if rand.Float64() < p {
			outS[i] = 0
		} else {
			outS[i] = xS[i] * scale
		}
	}
	return nil
}

func triangleCPU[T dtype.Numeric](lower bool) registry.Kernel {
	return func(ctx *registry.Context) error {
		a, out := ctx.Inputs[0], ctx.Output
		k := ctx.Attrs["k"].(int)
		aS := asSlice[T](a)
		outS := asSlice[T](out)
		// Copy input to output.
		copy(outS, aS)
		rank := out.Rank()
		dims := out.Shape().Dims()
		rows := dims[rank-2]
		cols := dims[rank-1]
		batch := out.NumElements() / (rows * cols)
		matSize := rows * cols
		var zero T
		for b := range batch {
			base := b * matSize
			for i := range rows {
				for j := range cols {
					if lower {
						if j > i+k {
							outS[base+i*cols+j] = zero
						}
					} else {
						if j < i+k {
							outS[base+i*cols+j] = zero
						}
					}
				}
			}
		}
		return nil
	}
}

// ropeI* apply RoPE with GPT-J interleaved pairs: dims (2i, 2i+1) rotate by
// pos * base^(-2i/headDim). Used by GPT-J, CodeGen and Phi-1/2 checkpoints.
func ropeIF32CPU(ctx *registry.Context) error {
	x, positions, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	base := ctx.Attrs["base"].(float64)
	xS, posS, outS := asSlice[float32](x), asSlice[int64](positions), asSlice[float32](out)
	ropeIWalk(x.Dim(-1), x.Dim(-2), x.NumElements(), posS, base, func(off int, c, sn float64) {
		x0, x1 := float64(xS[off]), float64(xS[off+1])
		outS[off] = float32(x0*c - x1*sn)
		outS[off+1] = float32(x0*sn + x1*c)
	})
	return nil
}

func ropeIF64CPU(ctx *registry.Context) error {
	x, positions, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	base := ctx.Attrs["base"].(float64)
	xS, posS, outS := asSlice[float64](x), asSlice[int64](positions), asSlice[float64](out)
	ropeIWalk(x.Dim(-1), x.Dim(-2), x.NumElements(), posS, base, func(off int, c, sn float64) {
		x0, x1 := xS[off], xS[off+1]
		outS[off] = x0*c - x1*sn
		outS[off+1] = x0*sn + x1*c
	})
	return nil
}

func ropeIBF16CPU(ctx *registry.Context) error {
	x, positions, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	base := ctx.Attrs["base"].(float64)
	xS, posS, outS := asSlice[uint16](x), asSlice[int64](positions), asSlice[uint16](out)
	ropeIWalk(x.Dim(-1), x.Dim(-2), x.NumElements(), posS, base, func(off int, c, sn float64) {
		x0 := float64(dtype.BFloat16(xS[off]).ToF32())
		x1 := float64(dtype.BFloat16(xS[off+1]).ToF32())
		outS[off] = uint16(dtype.BF16FromF32(float32(x0*c - x1*sn)))
		outS[off+1] = uint16(dtype.BF16FromF32(float32(x0*sn + x1*c)))
	})
	return nil
}

// ropeIWalk visits every interleaved pair with its rotation.
func ropeIWalk(headDim, seqLen, numel int, pos []int64, base float64, visit func(off int, c, sn float64)) {
	rows := numel / headDim
	half := headDim / 2
	for r := range rows {
		p := float64(pos[r%seqLen])
		for i := range half {
			theta := p * math.Pow(base, -float64(2*i)/float64(headDim))
			visit(r*headDim+2*i, math.Cos(theta), math.Sin(theta))
		}
	}
}
