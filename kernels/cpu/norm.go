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
	reg("rmsnorm", dtype.F32, rmsnormF32)
	reg("rmsnorm", dtype.F64, rmsnormF64)
	reg("rmsnorm", dtype.BF16, rmsnormBF16)
	reg("layernorm", dtype.F32, layernormF32)
	reg("layernorm", dtype.F64, layernormF64)
}

// rmsnormF32: y = x * rsqrt(mean(x²) + eps) * weight, along the last axis.
// Input x is (..., cols); weight is (cols,); out has shape of x.
func rmsnormF32(ctx *registry.Context) error {
	x, out := ctx.Inputs[0], ctx.Output
	weight := ctx.Inputs[1]
	eps, _ := ctx.Attrs["eps"].(float64)

	xS := asSlice[float32](x)
	wS := asSlice[float32](weight)
	oS := asSlice[float32](out)

	cols := x.Dim(-1)
	rows := x.NumElements() / cols
	for r := range rows {
		rowX := xS[r*cols : (r+1)*cols]
		rowO := oS[r*cols : (r+1)*cols]
		var sumSq float32
		for _, v := range rowX {
			sumSq += v * v
		}
		inv := float32(1.0 / math.Sqrt(float64(sumSq/float32(cols))+eps))
		for i, v := range rowX {
			rowO[i] = v * inv * wS[i]
		}
	}
	return nil
}

// rmsnormBF16 is rmsnormF32 for BF16 x and weight: the statistics and the
// product are computed in F32 and each output is rounded to BF16 once, as
// transformers' RMSNorm does (x.float() … .to(input_dtype)).
func rmsnormBF16(ctx *registry.Context) error {
	x, out := ctx.Inputs[0], ctx.Output
	weight := ctx.Inputs[1]
	eps, _ := ctx.Attrs["eps"].(float64)

	xS := asSlice[uint16](x)
	wS := asSlice[uint16](weight)
	oS := asSlice[uint16](out)

	cols := x.Dim(-1)
	rows := x.NumElements() / cols
	w := make([]float32, cols)
	for i := range w {
		w[i] = dtype.BFloat16(wS[i]).ToF32()
	}
	row := make([]float32, cols)
	for r := range rows {
		var sumSq float64
		for i, v := range xS[r*cols : (r+1)*cols] {
			row[i] = dtype.BFloat16(v).ToF32()
			sumSq += float64(row[i]) * float64(row[i])
		}
		inv := float32(1.0 / math.Sqrt(sumSq/float64(cols)+eps))
		rowO := oS[r*cols : (r+1)*cols]
		for i, v := range row {
			rowO[i] = uint16(dtype.BF16FromF32(v * inv * w[i]))
		}
	}
	return nil
}

func rmsnormF64(ctx *registry.Context) error {
	x, out := ctx.Inputs[0], ctx.Output
	weight := ctx.Inputs[1]
	eps, _ := ctx.Attrs["eps"].(float64)

	xS := asSlice[float64](x)
	wS := asSlice[float64](weight)
	oS := asSlice[float64](out)

	cols := x.Dim(-1)
	rows := x.NumElements() / cols
	for r := range rows {
		rowX := xS[r*cols : (r+1)*cols]
		rowO := oS[r*cols : (r+1)*cols]
		var sumSq float64
		for _, v := range rowX {
			sumSq += v * v
		}
		inv := 1.0 / math.Sqrt(sumSq/float64(cols)+eps)
		for i, v := range rowX {
			rowO[i] = v * inv * wS[i]
		}
	}
	return nil
}

// layernormF32: y = gamma * (x - mean) / sqrt(var + eps) + beta, last axis.
func layernormF32(ctx *registry.Context) error {
	x, out := ctx.Inputs[0], ctx.Output
	gamma, beta := ctx.Inputs[1], ctx.Inputs[2]
	eps, _ := ctx.Attrs["eps"].(float64)

	xS := asSlice[float32](x)
	gS := asSlice[float32](gamma)
	bS := asSlice[float32](beta)
	oS := asSlice[float32](out)

	cols := x.Dim(-1)
	rows := x.NumElements() / cols
	colsF := float32(cols)
	for r := range rows {
		rowX := xS[r*cols : (r+1)*cols]
		rowO := oS[r*cols : (r+1)*cols]
		var sum float32
		for _, v := range rowX {
			sum += v
		}
		mean := sum / colsF
		var varSum float32
		for _, v := range rowX {
			d := v - mean
			varSum += d * d
		}
		inv := float32(1.0 / math.Sqrt(float64(varSum/colsF)+eps))
		for i, v := range rowX {
			rowO[i] = (v-mean)*inv*gS[i] + bS[i]
		}
	}
	return nil
}

func layernormF64(ctx *registry.Context) error {
	x, out := ctx.Inputs[0], ctx.Output
	gamma, beta := ctx.Inputs[1], ctx.Inputs[2]
	eps, _ := ctx.Attrs["eps"].(float64)

	xS := asSlice[float64](x)
	gS := asSlice[float64](gamma)
	bS := asSlice[float64](beta)
	oS := asSlice[float64](out)

	cols := x.Dim(-1)
	rows := x.NumElements() / cols
	colsF := float64(cols)
	for r := range rows {
		rowX := xS[r*cols : (r+1)*cols]
		rowO := oS[r*cols : (r+1)*cols]
		var sum float64
		for _, v := range rowX {
			sum += v
		}
		mean := sum / colsF
		var varSum float64
		for _, v := range rowX {
			d := v - mean
			varSum += d * d
		}
		inv := 1.0 / math.Sqrt(varSum/colsF+eps)
		for i, v := range rowX {
			rowO[i] = (v-mean)*inv*gS[i] + bS[i]
		}
	}
	return nil
}
