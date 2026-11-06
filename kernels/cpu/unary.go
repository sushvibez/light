package cpu

import (
	"math"

	"light/device"
	"light/dtype"
	"light/internal/registry"
)

// unaryOp runs a broadcasting-irrelevant unary kernel over a Numeric dtype.
// Unary ops never broadcast, but do respect strides (caller may pass a view).
func unaryOp[T dtype.Numeric](ctx *registry.Context, fn func(a T) T) error {
	a, out := ctx.Inputs[0], ctx.Output
	aS := asSlice[T](a)
	outS := asSlice[T](out)

	// Fast path: same-shape + contiguous + offset 0 for both.
	if a.IsContiguous() && a.Offset() == 0 && a.Shape().Equal(out.Shape()) {
		for i, v := range aS {
			outS[i] = fn(v)
		}
		return nil
	}

	dims := out.Shape().Dims()
	rank := len(dims)
	idx := make([]int, rank)
	aStr := a.Strides().Slice()
	aBase := a.Offset()
	n := out.NumElements()
	for i := range n {
		aOff := aBase
		for ax := range rank {
			aOff += idx[ax] * aStr[ax]
		}
		outS[i] = fn(aS[aOff])
		for ax := rank - 1; ax >= 0; ax-- {
			idx[ax]++
			if idx[ax] < dims[ax] {
				break
			}
			idx[ax] = 0
		}
	}
	return nil
}

// unaryHalfOp: unary on F16 via F32 internally.
func unaryHalfOp(ctx *registry.Context, fn func(a float32) float32) error {
	a, out := ctx.Inputs[0], ctx.Output
	aS := asSlice[uint16](a)
	outS := asSlice[uint16](out)
	dims := out.Shape().Dims()
	rank := len(dims)
	idx := make([]int, rank)
	aStr := a.Strides().Slice()
	aBase := a.Offset()
	n := out.NumElements()
	for i := range n {
		aOff := aBase
		for ax := range rank {
			aOff += idx[ax] * aStr[ax]
		}
		av := dtype.Half(aS[aOff]).ToF32()
		outS[i] = uint16(dtype.HalfFromF32(fn(av)))
		for ax := rank - 1; ax >= 0; ax-- {
			idx[ax]++
			if idx[ax] < dims[ax] {
				break
			}
			idx[ax] = 0
		}
	}
	return nil
}

// unaryBF16Op: unary on BF16 via F32 internally.
func unaryBF16Op(ctx *registry.Context, fn func(a float32) float32) error {
	a, out := ctx.Inputs[0], ctx.Output
	aS := asSlice[uint16](a)
	outS := asSlice[uint16](out)
	dims := out.Shape().Dims()
	rank := len(dims)
	idx := make([]int, rank)
	aStr := a.Strides().Slice()
	aBase := a.Offset()
	n := out.NumElements()
	for i := range n {
		aOff := aBase
		for ax := range rank {
			aOff += idx[ax] * aStr[ax]
		}
		av := dtype.BFloat16(aS[aOff]).ToF32()
		outS[i] = uint16(dtype.BF16FromF32(fn(av)))
		for ax := rank - 1; ax >= 0; ax-- {
			idx[ax]++
			if idx[ax] < dims[ax] {
				break
			}
			idx[ax] = 0
		}
	}
	return nil
}

// registerUnaryNumeric wires a unary op across *all* Numeric dtypes (including
// integer). Useful for ops like neg, abs.
func registerUnaryNumeric(
	op string,
	f32 func(float32) float32,
	f64 func(float64) float64,
	i32 func(int32) int32,
	i64 func(int64) int64,
	u8 func(uint8) uint8,
	i8 func(int8) int8,
) {
	reg := func(dt dtype.DType, k registry.Kernel) {
		registry.Register(registry.Key{Op: op, Dev: 0, DType: dt}, k)
	}
	reg(dtype.F32, func(ctx *registry.Context) error { return unaryOp(ctx, f32) })
	reg(dtype.F64, func(ctx *registry.Context) error { return unaryOp(ctx, f64) })
	reg(dtype.I32, func(ctx *registry.Context) error { return unaryOp(ctx, i32) })
	reg(dtype.I64, func(ctx *registry.Context) error { return unaryOp(ctx, i64) })
	reg(dtype.U8, func(ctx *registry.Context) error { return unaryOp(ctx, u8) })
	reg(dtype.I8, func(ctx *registry.Context) error { return unaryOp(ctx, i8) })
	reg(dtype.F16, func(ctx *registry.Context) error { return unaryHalfOp(ctx, f32) })
	reg(dtype.BF16, func(ctx *registry.Context) error { return unaryBF16Op(ctx, f32) })
}

// registerUnaryFloat wires a unary op across float dtypes only (F16/BF16/F32/F64).
// For transcendental ops (exp/log/sqrt/sin/…), integers make no sense.
func registerUnaryFloat(op string, fn func(float32) float32, fn64 func(float64) float64) {
	reg := func(dt dtype.DType, k registry.Kernel) {
		registry.Register(registry.Key{Op: op, Dev: 0, DType: dt}, k)
	}
	reg(dtype.F32, func(ctx *registry.Context) error { return unaryOp(ctx, fn) })
	reg(dtype.F64, func(ctx *registry.Context) error { return unaryOp(ctx, fn64) })
	reg(dtype.F16, func(ctx *registry.Context) error { return unaryHalfOp(ctx, fn) })
	reg(dtype.BF16, func(ctx *registry.Context) error { return unaryBF16Op(ctx, fn) })
}

func init() {
	registerUnaryNumeric("neg",
		func(a float32) float32 { return -a },
		func(a float64) float64 { return -a },
		func(a int32) int32 { return -a },
		func(a int64) int64 { return -a },
		func(a uint8) uint8 { return -a },
		func(a int8) int8 { return -a },
	)
	registerUnaryNumeric("abs",
		func(a float32) float32 {
			if a < 0 {
				return -a
			}
			return a
		},
		math.Abs,
		func(a int32) int32 {
			if a < 0 {
				return -a
			}
			return a
		},
		func(a int64) int64 {
			if a < 0 {
				return -a
			}
			return a
		},
		func(a uint8) uint8 { return a },
		func(a int8) int8 {
			if a < 0 {
				return -a
			}
			return a
		},
	)

	registerUnaryFloat("exp",
		func(a float32) float32 { return float32(math.Exp(float64(a))) },
		math.Exp,
	)
	registerUnaryFloat("log",
		func(a float32) float32 { return float32(math.Log(float64(a))) },
		math.Log,
	)
	registerUnaryFloat("sqrt",
		func(a float32) float32 { return float32(math.Sqrt(float64(a))) },
		math.Sqrt,
	)
	registerUnaryFloat("sin",
		func(a float32) float32 { return float32(math.Sin(float64(a))) },
		math.Sin,
	)
	registerUnaryFloat("cos",
		func(a float32) float32 { return float32(math.Cos(float64(a))) },
		math.Cos,
	)
	registerUnaryFloat("tanh",
		func(a float32) float32 { return float32(math.Tanh(float64(a))) },
		math.Tanh,
	)
	registerUnaryFloat("sigmoid",
		func(a float32) float32 { return 1.0 / (1.0 + float32(math.Exp(float64(-a)))) },
		func(a float64) float64 { return 1.0 / (1.0 + math.Exp(-a)) },
	)
	registerUnaryFloat("relu",
		func(a float32) float32 {
			if a < 0 {
				return 0
			}
			return a
		},
		func(a float64) float64 {
			if a < 0 {
				return 0
			}
			return a
		},
	)
	registerUnaryFloat("gelu",
		func(a float32) float32 { return float32(gelu64(float64(a))) },
		gelu64,
	)
	registerUnaryFloat("gelu_erf",
		func(a float32) float32 { return float32(geluErf64(float64(a))) },
		geluErf64,
	)
	registerUnaryFloat("silu",
		func(a float32) float32 { return a / (1.0 + float32(math.Exp(float64(-a)))) },
		func(a float64) float64 { return a / (1.0 + math.Exp(-a)) },
	)
	registerUnaryFloat("tan",
		func(a float32) float32 { return float32(math.Tan(float64(a))) },
		math.Tan,
	)
	registerUnaryFloat("floor",
		func(a float32) float32 { return float32(math.Floor(float64(a))) },
		math.Floor,
	)
	registerUnaryFloat("ceil",
		func(a float32) float32 { return float32(math.Ceil(float64(a))) },
		math.Ceil,
	)
	registerUnaryFloat("round",
		func(a float32) float32 { return float32(math.Round(float64(a))) },
		math.Round,
	)
}

// gelu64 implements the tanh approximation used by many transformer libs.
func gelu64(x float64) float64 {
	c := math.Sqrt(2.0 / math.Pi)
	return 0.5 * x * (1.0 + math.Tanh(c*(x+0.044715*x*x*x)))
}

// geluErf64 is the exact GELU: 0.5·x·(1+erf(x/√2)).
func geluErf64(x float64) float64 {
	return 0.5 * x * (1.0 + math.Erf(x/math.Sqrt2))
}

// f32Pow / f64Pow / intPow helpers for the Pow binary op.
func f32Pow(a, b float32) float32 { return float32(math.Pow(float64(a), float64(b))) }
func f64Pow(a, b float64) float64 { return math.Pow(a, b) }
func intPow(base, exp int32) int32 {
	if exp < 0 {
		return 0
	}
	r := int32(1)
	for range exp {
		r *= base
	}
	return r
}
func int64Pow(base, exp int64) int64 {
	if exp < 0 {
		return 0
	}
	r := int64(1)
	for range exp {
		r *= base
	}
	return r
}

// ---- sign / erf / recip / sqr / mish / hard_swish / hard_sigmoid and the
// parametrised clamp / elu / leaky_relu / powf / affine (2026-10-28). CUDA
// twins: kernels/cuda/src/elementwise.cu + bf16_ops.cu. ----

func init() {
	registerUnaryNumeric("sign",
		func(a float32) float32 { return sign32(a) },
		sign64,
		func(a int32) int32 {
			if a > 0 {
				return 1
			} else if a < 0 {
				return -1
			}
			return 0
		},
		func(a int64) int64 {
			if a > 0 {
				return 1
			} else if a < 0 {
				return -1
			}
			return 0
		},
		func(a uint8) uint8 {
			if a > 0 {
				return 1
			}
			return 0
		},
		func(a int8) int8 {
			if a > 0 {
				return 1
			} else if a < 0 {
				return -1
			}
			return 0
		},
	)
	registerUnaryFloat("erf", func(a float32) float32 { return float32(math.Erf(float64(a))) }, math.Erf)
	registerUnaryFloat("recip", func(a float32) float32 { return 1 / a }, func(a float64) float64 { return 1 / a })
	registerUnaryFloat("sqr", func(a float32) float32 { return a * a }, func(a float64) float64 { return a * a })
	registerUnaryFloat("mish", func(a float32) float32 { return float32(mish64(float64(a))) }, mish64)
	registerUnaryFloat("hard_sigmoid", func(a float32) float32 { return float32(hardSigmoid64(float64(a))) }, hardSigmoid64)
	registerUnaryFloat("hard_swish", func(a float32) float32 { return float32(float64(a) * hardSigmoid64(float64(a))) },
		func(a float64) float64 { return a * hardSigmoid64(a) })

	registerUnaryParam("clamp", func(p0, p1 float64) func(float64) float64 {
		return func(a float64) float64 { return math.Min(math.Max(a, p0), p1) }
	})
	registerUnaryParam("elu", func(p0, _ float64) func(float64) float64 {
		return func(a float64) float64 {
			if a > 0 {
				return a
			}
			return p0 * (math.Exp(a) - 1)
		}
	})
	registerUnaryParam("leaky_relu", func(p0, _ float64) func(float64) float64 {
		return func(a float64) float64 {
			if a > 0 {
				return a
			}
			return p0 * a
		}
	})
	registerUnaryParam("powf", func(p0, _ float64) func(float64) float64 {
		return func(a float64) float64 { return math.Pow(a, p0) }
	})
	registerUnaryParam("affine", func(p0, p1 float64) func(float64) float64 {
		return func(a float64) float64 { return a*p0 + p1 }
	})
}

func sign32(a float32) float32 {
	if a > 0 {
		return 1
	} else if a < 0 {
		return -1
	}
	return 0
}

func sign64(a float64) float64 {
	if a > 0 {
		return 1
	} else if a < 0 {
		return -1
	}
	return 0
}

// mish64 = x · tanh(softplus(x)); softplus saturates to x above 20 as the
// CUDA kernel does.
func mish64(x float64) float64 {
	sp := x
	if x <= 20 {
		sp = math.Log1p(math.Exp(x))
	}
	return x * math.Tanh(sp)
}

func hardSigmoid64(x float64) float64 {
	return math.Min(math.Max((x+3)/6, 0), 1)
}

// registerUnaryParam wires a unary op whose scalar parameters arrive in
// Attrs["p0"] / Attrs["p1"] (float64) across F32/F64/F16/BF16. `mk` builds
// the F64 kernel for a given parameter pair; F32 and halves go through it.
func registerUnaryParam(op string, mk func(p0, p1 float64) func(float64) float64) {
	reg := func(dt dtype.DType, k registry.Kernel) {
		registry.Register(registry.Key{Op: op, Dev: device.KindCPU, DType: dt}, k)
	}
	params := func(ctx *registry.Context) (float64, float64) {
		p0, _ := ctx.Attrs["p0"].(float64)
		p1, _ := ctx.Attrs["p1"].(float64)
		return p0, p1
	}
	f32 := func(ctx *registry.Context) func(float32) float32 {
		f := mk(params(ctx))
		return func(a float32) float32 { return float32(f(float64(a))) }
	}
	reg(dtype.F64, func(ctx *registry.Context) error { return unaryOp(ctx, mk(params(ctx))) })
	reg(dtype.F32, func(ctx *registry.Context) error { return unaryOp(ctx, f32(ctx)) })
	reg(dtype.F16, func(ctx *registry.Context) error { return unaryHalfOp(ctx, f32(ctx)) })
	reg(dtype.BF16, func(ctx *registry.Context) error { return unaryBF16Op(ctx, f32(ctx)) })
}
