//go:build !nocuda

package cuda

import (
	"fmt"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
)

// kernelName maps an (op, dtype) pair to the PTX function name we compiled.
// Names are unmangled in the .cu source (extern "C" __global__).
var kernelName = map[struct {
	op string
	dt dtype.DType
}]string{
	// add across widest dtype range
	{"add", dtype.F32}: "add_f32", {"add", dtype.F64}: "add_f64",
	{"add", dtype.I32}: "add_i32", {"add", dtype.I64}: "add_i64",
	{"add", dtype.F16}: "add_f16", {"add", dtype.BF16}: "add_bf16",
	// rest of binary
	{"sub", dtype.F32}: "sub_f32", {"sub", dtype.F64}: "sub_f64", {"sub", dtype.F16}: "sub_f16", {"sub", dtype.BF16}: "sub_bf16",
	{"mul", dtype.F32}: "mul_f32", {"mul", dtype.F64}: "mul_f64", {"mul", dtype.F16}: "mul_f16", {"mul", dtype.BF16}: "mul_bf16",
	{"div", dtype.F32}: "div_f32", {"div", dtype.F64}: "div_f64", {"div", dtype.F16}: "div_f16", {"div", dtype.BF16}: "div_bf16",
	// unary
	{"neg", dtype.F32}: "neg_f32", {"neg", dtype.F64}: "neg_f64", {"neg", dtype.F16}: "neg_f16", {"neg", dtype.BF16}: "neg_bf16",
	{"exp", dtype.F32}: "exp_f32", {"exp", dtype.F64}: "exp_f64", {"exp", dtype.F16}: "exp_f16", {"exp", dtype.BF16}: "exp_bf16",
	{"log", dtype.F32}: "log_f32", {"log", dtype.F64}: "log_f64", {"log", dtype.F16}: "log_f16", {"log", dtype.BF16}: "log_bf16",
	{"sqrt", dtype.F32}: "sqrt_f32", {"sqrt", dtype.F64}: "sqrt_f64", {"sqrt", dtype.F16}: "sqrt_f16", {"sqrt", dtype.BF16}: "sqrt_bf16",
	{"relu", dtype.F32}: "relu_f32", {"relu", dtype.F64}: "relu_f64", {"relu", dtype.F16}: "relu_f16", {"relu", dtype.BF16}: "relu_bf16",
	{"sigmoid", dtype.F32}: "sigmoid_f32", {"sigmoid", dtype.F64}: "sigmoid_f64", {"sigmoid", dtype.F16}: "sigmoid_f16", {"sigmoid", dtype.BF16}: "sigmoid_bf16",
	{"tanh", dtype.F32}: "tanh_f32", {"tanh", dtype.F64}: "tanh_f64", {"tanh", dtype.F16}: "tanh_f16", {"tanh", dtype.BF16}: "tanh_bf16",
	{"silu", dtype.F32}: "silu_f32", {"silu", dtype.F64}: "silu_f64", {"silu", dtype.F16}: "silu_f16", {"silu", dtype.BF16}: "silu_bf16",
	// Phase 2 additions.
	{"abs", dtype.F32}: "abs_f32", {"abs", dtype.F64}: "abs_f64",
	{"sin", dtype.F32}: "sin_f32", {"sin", dtype.F64}: "sin_f64",
	{"cos", dtype.F32}: "cos_f32", {"cos", dtype.F64}: "cos_f64",
	{"tan", dtype.F32}: "tan_f32", {"tan", dtype.F64}: "tan_f64",
	{"floor", dtype.F32}: "floor_f32", {"floor", dtype.F64}: "floor_f64",
	{"ceil", dtype.F32}: "ceil_f32", {"ceil", dtype.F64}: "ceil_f64",
	{"round", dtype.F32}: "round_f32", {"round", dtype.F64}: "round_f64",
	{"gelu", dtype.F32}: "gelu_f32", {"gelu", dtype.F64}: "gelu_f64", {"gelu", dtype.BF16}: "gelu_bf16", {"gelu", dtype.F16}: "gelu_f16",
	{"gelu_erf", dtype.F32}: "gelu_erf_f32", {"gelu_erf", dtype.F64}: "gelu_erf_f64", {"gelu_erf", dtype.BF16}: "gelu_erf_bf16", {"gelu_erf", dtype.F16}: "gelu_erf_f16",
	{"pow", dtype.F32}: "pow_f32", {"pow", dtype.F64}: "pow_f64",
}

func init() {
	binDT := []dtype.DType{dtype.F32, dtype.F64, dtype.F16, dtype.BF16}
	addDT := append([]dtype.DType{dtype.I32, dtype.I64}, binDT...)
	for _, op := range []string{"add"} {
		for _, dt := range addDT {
			registerBinary(op, dt)
		}
	}
	for _, op := range []string{"sub", "mul", "div"} {
		for _, dt := range binDT {
			registerBinary(op, dt)
		}
	}
	for _, op := range []string{"neg", "exp", "log", "sqrt", "relu", "sigmoid", "tanh", "silu"} {
		for _, dt := range binDT {
			registerUnary(op, dt)
		}
	}
	// Phase 2: F32/F64-only additions.
	for _, op := range []string{"abs", "sin", "cos", "tan", "floor", "ceil", "round"} {
		for _, dt := range []dtype.DType{dtype.F32, dtype.F64} {
			registerUnary(op, dt)
		}
	}
	// GELU (tanh and erf): F32, F64, BF16, F16 (half kernels compute in F32).
	for _, dt := range []dtype.DType{dtype.F32, dtype.F64, dtype.BF16, dtype.F16} {
		registerUnary("gelu", dt)
		registerUnary("gelu_erf", dt)
	}
	for _, dt := range []dtype.DType{dtype.F32, dtype.F64} {
		registerBinary("pow", dt)
	}
}

func registerBinary(op string, dt dtype.DType) {
	name, ok := kernelName[struct {
		op string
		dt dtype.DType
	}{op, dt}]
	if !ok {
		return
	}
	registry.Register(registry.Key{Op: op, Dev: device.KindCUDA, DType: dt}, makeBinaryLauncher(name))
}

func registerUnary(op string, dt dtype.DType) {
	name, ok := kernelName[struct {
		op string
		dt dtype.DType
	}{op, dt}]
	if !ok {
		return
	}
	registry.Register(registry.Key{Op: op, Dev: device.KindCUDA, DType: dt}, makeUnaryLauncher(name))
}

func makeBinaryLauncher(name string) registry.Kernel {
	return func(ctx *registry.Context) error {
		a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
		// Fast path: same-shape contiguous operands.
		if a.Shape().Equal(out.Shape()) && b.Shape().Equal(out.Shape()) &&
			a.IsContiguous() && b.IsContiguous() && a.Offset() == 0 && b.Offset() == 0 {
			if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
				return err
			}
			fn, err := GetKernel(name)
			if err != nil {
				return err
			}
			aPtr := a.Storage().CUDAPtr()
			bPtr := b.Storage().CUDAPtr()
			outPtr := out.Storage().CUDAPtr()
			n := uint64(out.NumElements())
			args := []unsafe.Pointer{
				unsafe.Pointer(&aPtr),
				unsafe.Pointer(&bPtr),
				unsafe.Pointer(&outPtr),
				unsafe.Pointer(&n),
			}
			if err := Launch1D(fn, n, args); err != nil {
				return err
			}
			return nil
		}
		// Strided / broadcast path: only F32/F64 in Phase 2 extended.
		return launchStridedBinary(name, ctx)
	}
}

func makeUnaryLauncher(name string) registry.Kernel {
	return func(ctx *registry.Context) error {
		a, out := ctx.Inputs[0], ctx.Output
		if !a.Shape().Equal(out.Shape()) {
			return fmt.Errorf("cuda %s: shape mismatch %v vs %v", name, a.Shape(), out.Shape())
		}
		if !a.IsContiguous() || a.Offset() != 0 {
			return fmt.Errorf("cuda %s: non-contiguous input not yet supported", name)
		}
		if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
			return err
		}
		fn, err := GetKernel(name)
		if err != nil {
			return err
		}
		aPtr := a.Storage().CUDAPtr()
		outPtr := out.Storage().CUDAPtr()
		n := uint64(out.NumElements())
		args := []unsafe.Pointer{
			unsafe.Pointer(&aPtr),
			unsafe.Pointer(&outPtr),
			unsafe.Pointer(&n),
		}
		if err := Launch1D(fn, n, args); err != nil {
			return err
		}
		return nil
	}
}

// ---- sign / erf / recip / sqr / mish / hard_swish / hard_sigmoid,
// parametrised clamp / elu / leaky_relu / powf / affine, and maximum /
// minimum (2026-10-28). Kernels: src/elementwise.cu (F32/F64/I32/I64) and
// src/bf16_ops.cu (BF16). ----
func init() {
	type key = struct {
		op string
		dt dtype.DType
	}
	floats := []dtype.DType{dtype.F32, dtype.F64, dtype.BF16}
	ints := []dtype.DType{dtype.I32, dtype.I64}
	suffix := map[dtype.DType]string{dtype.F32: "f32", dtype.F64: "f64", dtype.BF16: "bf16", dtype.I32: "i32", dtype.I64: "i64"}

	for _, op := range []string{"erf", "recip", "sqr", "mish", "hard_swish", "hard_sigmoid"} {
		for _, dt := range floats {
			kernelName[key{op, dt}] = op + "_" + suffix[dt]
			registerUnary(op, dt)
		}
	}
	for _, dt := range append(append([]dtype.DType{}, floats...), ints...) {
		kernelName[key{"sign", dt}] = "sign_" + suffix[dt]
		registerUnary("sign", dt)
	}
	// Parametrised unary (Attrs p0, p1).
	for _, op := range []string{"clamp", "elu", "leaky_relu", "powf", "affine"} {
		for _, dt := range floats {
			registry.Register(registry.Key{Op: op, Dev: device.KindCUDA, DType: dt}, makeUnaryParamLauncher(op+"_"+suffix[dt], dt))
		}
	}
	// maximum / minimum: plain kernel here, strided variant via stridedBinaryName.
	for _, op := range []string{"maximum", "minimum"} {
		for _, dt := range append(append([]dtype.DType{}, floats...), ints...) {
			kernelName[key{op, dt}] = op + "_" + suffix[dt]
			registerBinary(op, dt)
		}
	}
}

// makeUnaryParamLauncher launches `<op>_<dt>(a, out, n, p0, p1)` with the
// two scalars read from Attrs["p0"] / Attrs["p1"] (float64; missing = 0).
// The scalar width follows the dtype: double for F64, float otherwise.
func makeUnaryParamLauncher(name string, dt dtype.DType) registry.Kernel {
	return func(ctx *registry.Context) error {
		a, out := ctx.Inputs[0], ctx.Output
		if !a.Shape().Equal(out.Shape()) {
			return fmt.Errorf("cuda %s: shape mismatch %v vs %v", name, a.Shape(), out.Shape())
		}
		if !a.IsContiguous() || a.Offset() != 0 {
			return fmt.Errorf("cuda %s: non-contiguous input not yet supported", name)
		}
		p0, _ := ctx.Attrs["p0"].(float64)
		p1, _ := ctx.Attrs["p1"].(float64)
		if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
			return err
		}
		fn, err := GetKernel(name)
		if err != nil {
			return err
		}
		aPtr := a.Storage().CUDAPtr()
		outPtr := out.Storage().CUDAPtr()
		n := uint64(out.NumElements())
		args := []unsafe.Pointer{
			unsafe.Pointer(&aPtr),
			unsafe.Pointer(&outPtr),
			unsafe.Pointer(&n),
		}
		p0f, p1f := float32(p0), float32(p1)
		if dt == dtype.F64 {
			args = append(args, unsafe.Pointer(&p0), unsafe.Pointer(&p1))
		} else {
			args = append(args, unsafe.Pointer(&p0f), unsafe.Pointer(&p1f))
		}
		if err := Launch1D(fn, n, args); err != nil {
			return err
		}
		return nil
	}
}
