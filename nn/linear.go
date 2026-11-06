package nn

import (
	"fmt"
	"math"
	"unsafe"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// Linear implements `y = x @ W^T + b`.
//
//	W: (OutFeatures, InFeatures)
//	b: (OutFeatures,)  — optional
//	x: (..., InFeatures)
//	y: (..., OutFeatures)
//
// Bias is broadcast across the leading batch dims via the strided binary
// kernels on CUDA / CPU.
type Linear struct {
	W *tensor.Tensor
	B *tensor.Tensor // nil if no bias; same dtype as W (activation dtype for F8 weights)
	// WScale de-quantises an F8E4M3 weight: W ≈ codes · WScale (0 means 1).
	// Loaders set it from `weight_scale` / `weight_scale_inv` tensors.
	WScale      float32
	inFeatures  int
	outFeatures int
}

// NewLinear builds a Linear layer with Kaiming-uniform-initialised F32
// weights. If bias=true, the bias is initialised to zero.
func NewLinear(inFeatures, outFeatures int, bias bool, dev device.Device) (*Linear, error) {
	return NewLinearWithDType(inFeatures, outFeatures, bias, dtype.F32, dev)
}

// NewLinearWithDType builds a Linear with weights stored at `wdt`,
// Kaiming-uniform-initialised. Bias is zero-initialised in F32 regardless of
// `wdt` (loaders / forward will lazy-cast to BF16 when needed).
func NewLinearWithDType(inFeatures, outFeatures int, bias bool, wdt dtype.DType, dev device.Device) (*Linear, error) {
	return newLinearImpl(inFeatures, outFeatures, bias, wdt, dev, true)
}

// NewLinearUninit allocates the weight tensor with zeros instead of running
// Kaiming init. Use this when a loader is about to overwrite the values —
// for a 3B-param model this skips ~3B random draws + a per-tensor F32→BF16
// CPU cast, saving ~60 s of model construction.
func NewLinearUninit(inFeatures, outFeatures int, bias bool, wdt dtype.DType, dev device.Device) (*Linear, error) {
	return newLinearImpl(inFeatures, outFeatures, bias, wdt, dev, false)
}

// NewLinearPlaceholder builds a Linear whose weight is nil until a loader
// calls SetWeight. It allocates nothing for W (only an F32 zero bias when
// bias is true), so a quantized GGUF model can be constructed without ever
// materialising F32 weights. Forward errors until the weight is set.
func NewLinearPlaceholder(inFeatures, outFeatures int, bias bool, dev device.Device) (*Linear, error) {
	l := &Linear{inFeatures: inFeatures, outFeatures: outFeatures}
	if bias {
		b, err := tensor.Zeros(dtype.F32, []int{outFeatures}, dev)
		if err != nil {
			return nil, err
		}
		l.B = b
	}
	return l, nil
}

// SetWeight installs w as the layer weight. w must be (outFeatures,
// inFeatures); any dtype, including GGUF quantized formats, is accepted.
func (l *Linear) SetWeight(w *tensor.Tensor) error {
	if w == nil || w.Rank() != 2 || w.Dim(0) != l.outFeatures || w.Dim(1) != l.inFeatures {
		return fmt.Errorf("nn.Linear.SetWeight: want (%d, %d), got %v", l.outFeatures, l.inFeatures, w)
	}
	l.W = w
	return nil
}

// NewLinearTied builds a Linear that reuses an existing weight tensor — no
// allocation. Used for tied LM heads where the embedding's storage doubles
// as the projection weight; saves V*D*sizeof(dtype) of throwaway GPU memory
// (≈300 MB on Qwen2.5-3B BF16) that would otherwise sit pending GC.
//
//	weight: (outFeatures, inFeatures) — typically the token-embedding weight
func NewLinearTied(weight *tensor.Tensor, bias bool, dev device.Device) (*Linear, error) {
	if weight == nil {
		return nil, fmt.Errorf("nn.NewLinearTied: weight is nil")
	}
	if weight.Rank() != 2 {
		return nil, fmt.Errorf("nn.NewLinearTied: weight must be rank-2, got %v", weight.Shape())
	}
	l := &Linear{
		W:           weight,
		outFeatures: weight.Dim(0),
		inFeatures:  weight.Dim(1),
	}
	if bias {
		b, err := tensor.Zeros(dtype.F32, []int{l.outFeatures}, dev)
		if err != nil {
			return nil, err
		}
		l.B = b
	}
	return l, nil
}

func newLinearImpl(inFeatures, outFeatures int, bias bool, wdt dtype.DType, dev device.Device, kaiming bool) (*Linear, error) {
	if inFeatures <= 0 || outFeatures <= 0 {
		return nil, fmt.Errorf("nn.Linear: features must be positive (in=%d, out=%d)", inFeatures, outFeatures)
	}
	var w *tensor.Tensor
	var err error
	if kaiming {
		// Kaiming uniform: U(-k, +k) with k = 1/sqrt(inFeatures). Init in F32
		// on CPU, cast to wdt on CPU, then transfer — that way no transient
		// F32 GPU tensor is left behind (otherwise pending finalizers can
		// pile up gigabytes of pinned device memory for a 3B-param model).
		k := float32(1.0 / math.Sqrt(float64(inFeatures)))
		var wCPU *tensor.Tensor
		wCPU, err = tensor.Rand(dtype.F32, []int{outFeatures, inFeatures}, device.CPU())
		if err != nil {
			return nil, err
		}
		scaleUniform(wCPU, k)
		if wdt != dtype.F32 {
			wCPU, err = ops.Cast(wCPU, wdt)
			if err != nil {
				return nil, err
			}
		}
		w = wCPU
		if !dev.SameAs(device.CPU()) {
			w, err = wCPU.To(dev)
			if err != nil {
				return nil, err
			}
		}
	} else {
		// Allocate directly on `dev` at the target dtype. One cudaMalloc, no
		// CPU pass, no Kaiming init — the loader is expected to fill values
		// before any forward.
		w, err = tensor.Zeros(wdt, []int{outFeatures, inFeatures}, dev)
		if err != nil {
			return nil, err
		}
	}
	l := &Linear{W: w, inFeatures: inFeatures, outFeatures: outFeatures}
	if bias {
		// Bias matches W's dtype — no runtime cast needed during Forward.
		// Loader is responsible for filling the right values via the same
		// dtype path; if you need a Kaiming-init F32-bias-with-BF16-W
		// model for some training experiment, write it as a follow-up
		// rather than re-introducing the dtype split here.
		b, err := tensor.Zeros(wdt, []int{outFeatures}, dev)
		if err != nil {
			return nil, err
		}
		l.B = b
	}
	return l, nil
}

// Forward computes y = x @ W^T + b. Requires `x.DType() == l.W.DType()` —
// the loader is responsible for landing every parameter at the same dtype,
// which keeps this hot path free of runtime dtype checks. The matmul takes
// the cublasGemmEx BF16-tensor-core path automatically when W is BF16 (via
// `ops.MatMulMixed`); F32 / F64 weights route through the regular GEMM.
func (l *Linear) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	if l.W == nil {
		return nil, fmt.Errorf("nn.Linear: weight not loaded (placeholder layer %d→%d)", l.inFeatures, l.outFeatures)
	}
	if !x.Device().SameAs(l.W.Device()) {
		return nil, fmt.Errorf("nn.Linear: x on %v but W on %v", x.Device(), l.W.Device())
	}
	if x.Rank() < 1 || x.Dim(-1) != l.inFeatures {
		return nil, fmt.Errorf("nn.Linear: last dim of x (%d) != in_features (%d)", x.Dim(-1), l.inFeatures)
	}
	quantized := l.W.DType().IsQuantized()
	if quantized && x.DType() != dtype.F32 {
		return nil, fmt.Errorf("nn.Linear: quantized weight (%v) needs F32 activations, got %v", l.W.DType(), x.DType())
	}
	fp8 := l.W.DType() == dtype.F8E4M3
	if fp8 && x.DType() != dtype.F32 && x.DType() != dtype.BF16 && x.DType() != dtype.F16 {
		return nil, fmt.Errorf("nn.Linear: F8E4M3 weight needs F32/BF16/F16 activations, got %v", x.DType())
	}
	if !quantized && !fp8 && x.DType() != l.W.DType() {
		return nil, fmt.Errorf("nn.Linear: x dtype %v != W dtype %v — cast at the boundary, not inside the layer", x.DType(), l.W.DType())
	}

	// For rank > 2, flatten the batch dims: x: (..., in) → (B, in) where B = prod(batch dims).
	origShape := x.Shape().Dims()
	flat := x
	if x.Rank() > 2 {
		batch := 1
		for _, d := range origShape[:len(origShape)-1] {
			batch *= d
		}
		f, err := autograd.Reshape(x, batch, l.inFeatures)
		if err != nil {
			return nil, err
		}
		flat = f
	}

	var y *tensor.Tensor
	var err error
	if quantized {
		// GGUF weight: fused dequant-matmul (no autograd; inference only).
		flatC, err2 := flat.Contiguous()
		if err2 != nil {
			return nil, err2
		}
		y, err = ops.MatMulQ(flatC, l.W)
	} else if fp8 {
		// FP8 weight: quantise the activation per-tensor, run the cuBLASLt
		// FP8 GEMM (CPU reference otherwise) and land at the activation
		// dtype. Inference only.
		flatC, err2 := flat.Contiguous()
		if err2 != nil {
			return nil, err2
		}
		xq, xScale, err2 := ops.QuantizeF8(flatC)
		if err2 != nil {
			return nil, err2
		}
		wT, err2 := l.W.Transpose(0, 1)
		if err2 != nil {
			return nil, err2
		}
		wScale := l.WScale
		if wScale == 0 {
			wScale = 1
		}
		y, err = ops.MatMulF8(xq, wT, xScale, wScale, x.DType())
	} else if l.W.DType() == dtype.BF16 || l.W.DType() == dtype.F16 {
		// BF16 / F16 → cuBLAS GemmEx tensor-core path (16-bit in, F32
		// accumulate, 16-bit out). Caller has already arranged dtype uniformity, so the
		// transpose is a pure view and contig is needed once.
		wT, err2 := l.W.Transpose(0, 1)
		if err2 != nil {
			return nil, err2
		}
		flatC, err2 := flat.Contiguous()
		if err2 != nil {
			return nil, err2
		}
		y, err = ops.MatMulMixed(flatC, wT)
	} else {
		wT, err2 := autograd.Transpose(l.W, 0, 1)
		if err2 != nil {
			return nil, err2
		}
		y, err = autograd.MatMul(flat, wT)
	}
	if err != nil {
		return nil, err
	}
	if l.B != nil {
		y, err = autograd.Add(y, l.B)
		if err != nil {
			return nil, err
		}
	}

	// Restore leading batch dims if we flattened.
	if x.Rank() > 2 {
		newShape := make([]int, len(origShape))
		copy(newShape, origShape)
		newShape[len(newShape)-1] = l.outFeatures
		var err error
		y, err = autograd.Reshape(y, newShape...)
		if err != nil {
			return nil, err
		}
	}
	return y, nil
}

// Parameters returns [W] or [W, B].
func (l *Linear) Parameters() []*tensor.Tensor {
	if l.B != nil {
		return []*tensor.Tensor{l.W, l.B}
	}
	return []*tensor.Tensor{l.W}
}

// InFeatures returns the input dimension.
func (l *Linear) InFeatures() int { return l.inFeatures }

// OutFeatures returns the output dimension.
func (l *Linear) OutFeatures() int { return l.outFeatures }

// scaleUniform rescales a CPU F32 tensor whose values are Uniform[0,1) to
// Uniform[-k, +k). Used for Kaiming init since tensor.Rand currently only
// supports Uniform[0,1).
func scaleUniform(t *tensor.Tensor, k float32) {
	buf := t.Storage().CPUBytes()
	vals := unsafe.Slice((*float32)(unsafe.Pointer(&buf[0])), len(buf)/4)
	scale := 2 * k
	for i := range vals {
		vals[i] = vals[i]*scale - k
	}
}
