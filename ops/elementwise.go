package ops

import (
	"fmt"

	"light/internal/registry"
	"light/tensor"
)

// Sub returns a - b, broadcasting.
func Sub(a, b *tensor.Tensor) (*tensor.Tensor, error) { return binaryDispatch("sub", a, b) }

// Mul returns a * b (elementwise), broadcasting.
func Mul(a, b *tensor.Tensor) (*tensor.Tensor, error) { return binaryDispatch("mul", a, b) }

// Div returns a / b, broadcasting. Integer div-by-zero returns 0 rather than
// panicking (matches Candle's convention of producing deterministic output).
func Div(a, b *tensor.Tensor) (*tensor.Tensor, error) { return binaryDispatch("div", a, b) }

// Pow returns a ** b, broadcasting.
func Pow(a, b *tensor.Tensor) (*tensor.Tensor, error) { return binaryDispatch("pow", a, b) }

// Neg returns -a.
func Neg(a *tensor.Tensor) (*tensor.Tensor, error) { return unaryDispatch("neg", a) }

// Abs returns |a|.
func Abs(a *tensor.Tensor) (*tensor.Tensor, error) { return unaryDispatch("abs", a) }

// Exp, Log, Sqrt, Sin, Cos, Tanh, Sigmoid, Relu, Gelu, SiLU operate on float
// dtypes only; other dtypes will produce a kernel-not-found error at dispatch.
func Exp(a *tensor.Tensor) (*tensor.Tensor, error)     { return unaryDispatch("exp", a) }
func Log(a *tensor.Tensor) (*tensor.Tensor, error)     { return unaryDispatch("log", a) }
func Sqrt(a *tensor.Tensor) (*tensor.Tensor, error)    { return unaryDispatch("sqrt", a) }
func Sin(a *tensor.Tensor) (*tensor.Tensor, error)     { return unaryDispatch("sin", a) }
func Cos(a *tensor.Tensor) (*tensor.Tensor, error)     { return unaryDispatch("cos", a) }
func Tanh(a *tensor.Tensor) (*tensor.Tensor, error)    { return unaryDispatch("tanh", a) }
func Sigmoid(a *tensor.Tensor) (*tensor.Tensor, error) { return unaryDispatch("sigmoid", a) }
func Relu(a *tensor.Tensor) (*tensor.Tensor, error)    { return unaryDispatch("relu", a) }
func Gelu(a *tensor.Tensor) (*tensor.Tensor, error)    { return unaryDispatch("gelu", a) }
func SiLU(a *tensor.Tensor) (*tensor.Tensor, error)    { return unaryDispatch("silu", a) }

// GeluErf is the exact Gaussian Error Linear Unit, 0.5·x·(1+erf(x/√2)) —
// PyTorch's default `F.gelu(x)` and HF BERT's `hidden_act: "gelu"`. `Gelu`
// above is the tanh approximation (GPT-2 / HF "gelu_new").
func GeluErf(a *tensor.Tensor) (*tensor.Tensor, error) { return unaryDispatch("gelu_erf", a) }

// Tan, Floor, Ceil, Round round out the usual unary float op set.
func Tan(a *tensor.Tensor) (*tensor.Tensor, error)   { return unaryDispatch("tan", a) }
func Floor(a *tensor.Tensor) (*tensor.Tensor, error) { return unaryDispatch("floor", a) }
func Ceil(a *tensor.Tensor) (*tensor.Tensor, error)  { return unaryDispatch("ceil", a) }
func Round(a *tensor.Tensor) (*tensor.Tensor, error) { return unaryDispatch("round", a) }

// binaryDispatch is the shared plumbing for every broadcasting binary op.
func binaryDispatch(op string, a, b *tensor.Tensor) (*tensor.Tensor, error) {
	outShape, err := validateBinaryOp(op, a, b)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(a.DType(), outShape.Dims(), a.Device())
	if err != nil {
		return nil, err
	}
	if err := dispatch(op, a.Device(), a.DType(), []*tensor.Tensor{a, b}, out); err != nil {
		return nil, err
	}
	return out, nil
}

// unaryDispatch is the shared plumbing for every unary op.
func unaryDispatch(op string, a *tensor.Tensor) (*tensor.Tensor, error) {
	out, err := tensor.Zeros(a.DType(), a.Shape().Dims(), a.Device())
	if err != nil {
		return nil, err
	}
	if err := dispatch(op, a.Device(), a.DType(), []*tensor.Tensor{a}, out); err != nil {
		return nil, err
	}
	return out, nil
}

// ---- Candle-parity unary / binary ops (2026-10-28). Kernels live in
// kernels/cpu/unary.go + binary.go and kernels/cuda/src/elementwise.cu +
// bf16_ops.cu. ----

// Sign returns -1, 0 or 1 per element (float and integer dtypes).
func Sign(a *tensor.Tensor) (*tensor.Tensor, error) { return unaryDispatch("sign", a) }

// Erf is the Gauss error function.
func Erf(a *tensor.Tensor) (*tensor.Tensor, error) { return unaryDispatch("erf", a) }

// Recip returns 1/a.
func Recip(a *tensor.Tensor) (*tensor.Tensor, error) { return unaryDispatch("recip", a) }

// Sqr returns a².
func Sqr(a *tensor.Tensor) (*tensor.Tensor, error) { return unaryDispatch("sqr", a) }

// Mish is x·tanh(softplus(x)).
func Mish(a *tensor.Tensor) (*tensor.Tensor, error) { return unaryDispatch("mish", a) }

// HardSwish is x·clamp((x+3)/6, 0, 1) (MobileNetV3).
func HardSwish(a *tensor.Tensor) (*tensor.Tensor, error) { return unaryDispatch("hard_swish", a) }

// HardSigmoid is clamp((x+3)/6, 0, 1).
func HardSigmoid(a *tensor.Tensor) (*tensor.Tensor, error) { return unaryDispatch("hard_sigmoid", a) }

// Clamp limits every element to [lo, hi].
func Clamp(a *tensor.Tensor, lo, hi float64) (*tensor.Tensor, error) {
	if lo > hi {
		return nil, fmt.Errorf("Clamp: lo %g > hi %g", lo, hi)
	}
	return unaryParamDispatch("clamp", a, lo, hi)
}

// ELU is x for x > 0, alpha·(eˣ-1) otherwise.
func ELU(a *tensor.Tensor, alpha float64) (*tensor.Tensor, error) {
	return unaryParamDispatch("elu", a, alpha, 0)
}

// LeakyReLU is x for x > 0, slope·x otherwise.
func LeakyReLU(a *tensor.Tensor, slope float64) (*tensor.Tensor, error) {
	return unaryParamDispatch("leaky_relu", a, slope, 0)
}

// PowScalar raises every element to the scalar exponent (Candle `powf`).
func PowScalar(a *tensor.Tensor, exponent float64) (*tensor.Tensor, error) {
	return unaryParamDispatch("powf", a, exponent, 0)
}

// Affine returns a·mul + add (Candle `affine`).
func Affine(a *tensor.Tensor, mul, add float64) (*tensor.Tensor, error) {
	return unaryParamDispatch("affine", a, mul, add)
}

func unaryParamDispatch(op string, a *tensor.Tensor, p0, p1 float64) (*tensor.Tensor, error) {
	src, err := ensureContig(a)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(src.DType(), src.Shape().Dims(), src.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: op, Dev: src.Device().Kind(), DType: src.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{src},
		Output: out,
		Attrs:  map[string]any{"p0": p0, "p1": p1},
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// Maximum returns the elementwise max of a and b, broadcasting.
func Maximum(a, b *tensor.Tensor) (*tensor.Tensor, error) { return binaryDispatch("maximum", a, b) }

// Minimum returns the elementwise min of a and b, broadcasting.
func Minimum(a, b *tensor.Tensor) (*tensor.Tensor, error) { return binaryDispatch("minimum", a, b) }

// RoundTo rounds every element to `decimals` decimal places (Candle
// `round_to`): round(x·10ᵈ)/10ᵈ.
func RoundTo(a *tensor.Tensor, decimals int) (*tensor.Tensor, error) {
	scale := 1.0
	for range decimals {
		scale *= 10
	}
	x, err := Affine(a, scale, 0)
	if err != nil {
		return nil, err
	}
	if x, err = Round(x); err != nil {
		return nil, err
	}
	return Affine(x, 1/scale, 0)
}
