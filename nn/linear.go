package nn

import (
	"fmt"
	"math"
	"unsafe"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/ops"
	"light/quant"
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
	WScale float32
	// WBlockScale switches an F8E4M3 weight to block scaling (the "fp8"
	// fine-grained format of Qwen3.5 / DeepSeek-V3 FP8 checkpoints):
	// W[i, j] ≈ codes[i, j] · WBlockScale[i / WBlock[0], j / WBlock[1]], F32.
	// Forward runs ops.MatMulF8Block: a fused FP8 GEMV for decode, the
	// weight expanded to the activation dtype + the ordinary GEMM otherwise
	// (any GPU; the weight stays 1 byte / param at rest).
	WBlockScale *tensor.Tensor
	WBlock      [2]int
	// W4 holds a group-quantized 4-bit weight (GPTQ / AWQ, converted to
	// quant.Int4) instead of W; Forward runs ops.MatMulInt4.
	W4 *quant.Int4
	// Mixed16 lets F32 activations against a BF16 / F16 weight round to the
	// weight's type for a tensor-core GEMM (F32 accumulate and output) on
	// devices that have one (CUDA); elsewhere the product stays exact.
	// Set for Linears built under WithWeightStorage16Mixed.
	Mixed16     bool
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
	mixed := false
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
		wd, w16, m := weightDType16Mixed(wdt, inFeatures) // 16-bit storage loads
		mixed = m && w16
		w, err = tensor.Zeros(wd, []int{outFeatures, inFeatures}, dev)
		if err != nil {
			return nil, err
		}
	}
	l := &Linear{W: w, Mixed16: mixed, inFeatures: inFeatures, outFeatures: outFeatures}
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
	if l.W4 != nil {
		return l.forwardInt4(x)
	}
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
	w := l.W
	blockFP8 := fp8 && l.WBlockScale != nil
	if blockFP8 {
		fp8 = false // handled by ops.MatMulF8Block below
		if x.DType() == dtype.F16 {
			return nil, fmt.Errorf("nn.Linear: block-scaled FP8 needs F32 or BF16 activations, got F16")
		}
	}
	// 16-bit weight storage (WebGPU W16A32): F32 activations against a
	// BF16 / F16 weight widened exactly inside the matmul.
	w16 := x.DType() == dtype.F32 && (w.DType() == dtype.BF16 || w.DType() == dtype.F16)
	if !quantized && !fp8 && !blockFP8 && !w16 && x.DType() != w.DType() {
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
	biasDone := false
	if blockFP8 {
		// Block-scaled FP8 (inference only): fused GEMV for a few rows,
		// dequantize + GEMM otherwise.
		y, err = ops.MatMulF8Block(flat, l.W, l.WBlockScale, l.WBlock[0], l.WBlock[1])
	} else if w16 {
		flatC, err2 := flat.Contiguous()
		if err2 != nil {
			return nil, err2
		}
		if l.Mixed16 && device.IsCUDA(flatC.Device()) {
			wT, err2 := w.Transpose(0, 1)
			if err2 != nil {
				return nil, err2
			}
			y, err = ops.MatMulMixed(flatC, wT)
		} else {
			y, err = ops.MatMulW16(flatC, w)
		}
	} else if quantized {
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
	} else if w.DType() == dtype.BF16 || w.DType() == dtype.F16 {
		// BF16 / F16 → cuBLAS GemmEx tensor-core path (16-bit in, F32
		// accumulate, 16-bit out). Caller has already arranged dtype uniformity, so the
		// transpose is a pure view and contig is needed once.
		wT, err2 := w.Transpose(0, 1)
		if err2 != nil {
			return nil, err2
		}
		flatC, err2 := flat.Contiguous()
		if err2 != nil {
			return nil, err2
		}
		if l.B != nil && w.DType() == dtype.BF16 && flatC.DType() == dtype.BF16 {
			// x·Wᵀ + b with one rounding, as cuBLAS's bias epilogue does for
			// transformers: GEMM to F32, add the bias in F32, round once.
			// Rounding x·Wᵀ to BF16 first loses the small difference when it
			// nearly cancels a large bias (Qwen2-family k_proj biases reach
			// 55, where BF16 steps are 0.25: MiMo-7B's logits moved by up to
			// 14 before this).
			y, err = linearBF16Bias(flatC, wT, l.B)
			biasDone = true
		} else {
			y, err = ops.MatMulMixed(flatC, wT)
		}
	} else {
		wT, err2 := autograd.Transpose(w, 0, 1)
		if err2 != nil {
			return nil, err2
		}
		y, err = autograd.MatMul(flat, wT)
	}
	if err != nil {
		return nil, err
	}
	if l.B != nil && !biasDone {
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

// Mixed16Input rounds x for ForwardMixed16 when this layer computes at 16
// bits (Mixed16 with a BF16 / F16 weight, F32 x, on CUDA), so several
// layers reading the same activation (q / k / v, a gated MLP's two input
// projections) cast it once. It returns nil when Forward would not round.
func (l *Linear) Mixed16Input(x *tensor.Tensor) (*tensor.Tensor, error) {
	if !l.Mixed16 || l.W == nil || x.DType() != dtype.F32 || !device.IsCUDA(x.Device()) ||
		(l.W.DType() != dtype.BF16 && l.W.DType() != dtype.F16) {
		return nil, nil
	}
	return ops.Cast(x, l.W.DType())
}

// ForwardMixed16 is Forward for x already rounded by Mixed16Input (of this
// layer or one with the same weight dtype): F32 out, as Forward gives.
func (l *Linear) ForwardMixed16(xr *tensor.Tensor) (*tensor.Tensor, error) {
	if l.W == nil || xr.DType() != l.W.DType() || xr.Dim(-1) != l.inFeatures {
		return nil, fmt.Errorf("nn.Linear.ForwardMixed16: want a %v (..., %d) input, got %v %v", l.W.DType(), l.inFeatures, xr.DType(), xr.Shape())
	}
	dims := xr.Shape().Dims()
	flat, err := xr.Contiguous()
	if err != nil {
		return nil, err
	}
	if flat, err = flat.Reshape(xr.NumElements()/l.inFeatures, l.inFeatures); err != nil {
		return nil, err
	}
	wT, err := l.W.Transpose(0, 1)
	if err != nil {
		return nil, err
	}
	y, err := ops.MatMulMixedF32Out(flat, wT)
	if err != nil {
		return nil, err
	}
	if l.B != nil {
		b := l.B
		if b.DType() != dtype.F32 {
			if b, err = ops.Cast(b, dtype.F32); err != nil {
				return nil, err
			}
		}
		if y, err = ops.Add(y, b); err != nil {
			return nil, err
		}
	}
	out := append([]int(nil), dims...)
	out[len(out)-1] = l.outFeatures
	return y.Reshape(out...)
}

// Parameters returns [W] or [W, B].
func (l *Linear) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	if l.W4 != nil {
		// Packed codes (8 per element) plus per-group scales / zeros.
		ps = []*tensor.Tensor{l.W4.Codes, l.W4.Scales, l.W4.Zeros}
	} else {
		ps = []*tensor.Tensor{l.W}
	}
	if l.B != nil {
		ps = append(ps, l.B)
	}
	return ps
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

// linearBF16Bias returns round_bf16(x·wT + b) for BF16 x (M, K), BF16 wT
// (K, N) and a bias b (N,): the product is accumulated and stored in F32
// (the F32 × BF16 GEMM, which casts x back to BF16 exactly), the bias added
// in F32, and the sum rounded once.
func linearBF16Bias(x, wT, b *tensor.Tensor) (*tensor.Tensor, error) {
	x32, err := ops.Cast(x, dtype.F32)
	if err != nil {
		return nil, err
	}
	y, err := ops.MatMulMixed(x32, wT) // F32 out
	if err != nil {
		return nil, err
	}
	b32 := b
	if b.DType() != dtype.F32 {
		if b32, err = ops.Cast(b, dtype.F32); err != nil {
			return nil, err
		}
	}
	if y, err = ops.Add(y, b32); err != nil {
		return nil, err
	}
	return ops.Cast(y, dtype.BF16)
}

// SetInt4 installs a 4-bit weight of shape (outFeatures, inFeatures).
func (l *Linear) SetInt4(w quant.Int4) error {
	if w.N() != l.outFeatures || w.K() != l.inFeatures {
		return fmt.Errorf("nn.Linear.SetInt4: want (%d, %d), got (%d, %d)", l.outFeatures, l.inFeatures, w.N(), w.K())
	}
	l.W4, l.W = &w, nil
	return nil
}

// forwardInt4 is Forward for a 4-bit weight (inference only). With a bias
// and BF16 activations the product is formed in F32 and rounded once after
// the bias, as for BF16 weights (see linearBF16Bias).
func (l *Linear) forwardInt4(x *tensor.Tensor) (*tensor.Tensor, error) {
	if !x.Device().SameAs(l.W4.Codes.Device()) {
		return nil, fmt.Errorf("nn.Linear: x on %v but the 4-bit weight on %v", x.Device(), l.W4.Codes.Device())
	}
	if x.Rank() < 1 || x.Dim(-1) != l.inFeatures {
		return nil, fmt.Errorf("nn.Linear: last dim of x (%d) != in_features (%d)", x.Dim(-1), l.inFeatures)
	}
	dims := x.Shape().Dims()
	flat, err := autograd.Reshape(x, -1, l.inFeatures)
	if err != nil {
		return nil, err
	}
	in := flat
	if l.B != nil && x.DType() == dtype.BF16 {
		if in, err = ops.Cast(flat, dtype.F32); err != nil {
			return nil, err
		}
	}
	y, err := ops.MatMulInt4(in, *l.W4)
	if err != nil {
		return nil, err
	}
	if l.B != nil {
		b := l.B
		if b.DType() != y.DType() {
			if b, err = ops.Cast(b, y.DType()); err != nil {
				return nil, err
			}
		}
		if y, err = ops.Add(y, b); err != nil {
			return nil, err
		}
		if y.DType() != x.DType() {
			if y, err = ops.Cast(y, x.DType()); err != nil {
				return nil, err
			}
		}
	}
	dims[len(dims)-1] = l.outFeatures
	return autograd.Reshape(y, dims...)
}
