package autograd

import (
	"fmt"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// CrossEntropyLossFunction computes the mean negative-log-likelihood of the
// softmax of `logits` at the indices given by `target`. Equivalent to
// PyTorch's `F.cross_entropy(logits, target, reduction='mean')`.
//
// Shapes:
//
//	logits: [B, C] F32
//	target: [B]    I64   (class indices in [0, C))
//	loss:   scalar F32
//
// Implementation: the one-hot tensor for `target` is built on CPU and
// transferred to the logits' device. From there, the stable softmax, loss
// and backward all flow through composed device-agnostic ops, so both CPU
// and CUDA F32 work end-to-end.
type CrossEntropyLossFunction struct{}

func (CrossEntropyLossFunction) Name() string { return "CrossEntropyLoss" }

func (CrossEntropyLossFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("CrossEntropyLoss: want 2 inputs (logits, target), got %d", len(inputs))
	}
	logits, target := inputs[0], inputs[1]
	if err := validateCELoss(logits, target); err != nil {
		return nil, err
	}
	B := logits.Dim(0)
	C := logits.Dim(1)
	dev := logits.Device()

	oneHot, err := buildOneHot(target, B, C, dev)
	if err != nil {
		return nil, err
	}

	// Stable softmax / log-softmax via composed ops.
	shift, err := shiftByMax(logits)
	if err != nil {
		return nil, err
	}
	expShift, err := ops.Exp(shift)
	if err != nil {
		return nil, err
	}
	sumExp, err := ops.Sum(expShift, ops.ReduceOptions{Axes: []int{1}, KeepDims: true})
	if err != nil {
		return nil, err
	}
	probs, err := ops.Div(expShift, sumExp)
	if err != nil {
		return nil, err
	}
	logSumExp, err := ops.Log(sumExp)
	if err != nil {
		return nil, err
	}
	logSoftmax, err := ops.Sub(shift, logSumExp)
	if err != nil {
		return nil, err
	}

	// Per-sample NLL: -sum(one_hot * log_softmax, axis=1) → [B]
	picked, err := ops.Mul(oneHot, logSoftmax)
	if err != nil {
		return nil, err
	}
	perSample, err := ops.Sum(picked, ops.ReduceOptions{Axes: []int{1}})
	if err != nil {
		return nil, err
	}
	negPer, err := ops.Neg(perSample)
	if err != nil {
		return nil, err
	}
	loss, err := ops.Mean(negPer, ops.ReduceOptions{})
	if err != nil {
		return nil, err
	}

	ctx.SaveForBackward(probs, oneHot)
	ctx.Set("batch", B)
	return []*tensor.Tensor{loss}, nil
}

func (CrossEntropyLossFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	saved := ctx.Saved()
	probs, oneHot := saved[0], saved[1]
	Bany, _ := ctx.Get("batch")
	B := Bany.(int)

	// grad_logits = (probs - one_hot) * (g / B)
	diff, err := ops.Sub(probs, oneHot)
	if err != nil {
		return nil, err
	}
	invB, err := tensor.Full(1.0/float32(B), []int{1}, g.Device())
	if err != nil {
		return nil, err
	}
	scale, err := ops.Mul(g, invB)
	if err != nil {
		return nil, err
	}
	gradLogits, err := ops.Mul(diff, scale)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gradLogits, nil}, nil
}

// CrossEntropyLoss is the autograd-aware fused softmax cross-entropy.
func CrossEntropyLoss(logits, target *tensor.Tensor) (*tensor.Tensor, error) {
	outs, err := Apply(CrossEntropyLossFunction{}, []*tensor.Tensor{logits, target})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// buildOneHot constructs a (B, C) F32 one-hot tensor on `dev` from an I64
// target [B]. Done on CPU then transferred to keep the indexing logic
// simple; target is tiny (a few KB) so the H2D transfer is negligible.
func buildOneHot(target *tensor.Tensor, B, C int, dev device.Device) (*tensor.Tensor, error) {
	var targetCPU *tensor.Tensor
	if target.Device().Kind() == device.KindCPU {
		cont, err := target.Contiguous()
		if err != nil {
			return nil, err
		}
		targetCPU = cont
	} else {
		t, err := target.To(device.CPU())
		if err != nil {
			return nil, err
		}
		targetCPU = t
	}
	tgt := cpuI64(targetCPU)
	data := make([]float32, B*C)
	for b, cls := range tgt {
		if cls < 0 || int(cls) >= C {
			return nil, fmt.Errorf(
				"target[%d]=%d out of range [0, %d)", b, cls, C)
		}
		data[b*C+int(cls)] = 1
	}
	return tensor.New(data, []int{B, C}, dev)
}

// validateCELoss checks shapes, dtypes, and device agreement.
func validateCELoss(logits, target *tensor.Tensor) error {
	if logits.DType() != dtype.F32 {
		return fmt.Errorf("CrossEntropyLoss: logits must be F32 (got %v)", logits.DType())
	}
	if logits.Rank() != 2 {
		return fmt.Errorf("CrossEntropyLoss: logits must be rank-2 (got %d)", logits.Rank())
	}
	return validateClassTarget("CrossEntropyLoss", target, logits.Dim(0))
}

// validateClassTarget checks that target is an I64 rank-1 tensor of exactly
// batch class indices, before buildOneHot reinterprets its storage. (Class
// bounds are checked while building the one-hot.)
func validateClassTarget(name string, target *tensor.Tensor, batch int) error {
	if target.DType() != dtype.I64 {
		return fmt.Errorf("%s: target must be I64 (got %v)", name, target.DType())
	}
	if target.Rank() != 1 {
		return fmt.Errorf("%s: target must be rank-1 (got %d)", name, target.Rank())
	}
	if target.Dim(0) != batch {
		return fmt.Errorf("%s: batch mismatch (input %d vs target %d)",
			name, batch, target.Dim(0))
	}
	return nil
}

// shiftByMax returns `x - max(x, axis=-1, keepdims=True)` via ops.Max + Sub.
// Used for stable softmax.
func shiftByMax(x *tensor.Tensor) (*tensor.Tensor, error) {
	mx, err := ops.Max(x, ops.ReduceOptions{Axes: []int{-1}, KeepDims: true})
	if err != nil {
		return nil, err
	}
	return ops.Sub(x, mx)
}

// cpuF32 reinterprets a CPU F32 tensor's storage as []float32.
func cpuF32(t *tensor.Tensor) []float32 {
	if t.Device().Kind() != device.KindCPU {
		panic("autograd.cpuF32: not CPU")
	}
	buf := t.Storage().CPUBytes()
	if len(buf) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&buf[0])), len(buf)/4)
}

// cpuI64 reinterprets a CPU I64 tensor's storage as []int64.
func cpuI64(t *tensor.Tensor) []int64 {
	if t.Device().Kind() != device.KindCPU {
		panic("autograd.cpuI64: not CPU")
	}
	buf := t.Storage().CPUBytes()
	if len(buf) == 0 {
		return nil
	}
	return unsafe.Slice((*int64)(unsafe.Pointer(&buf[0])), len(buf)/8)
}

// ---- Losses (Candle candle_nn::loss) ----
// MSELoss is mean((pred - target)²).
func MSELoss(pred, target *tensor.Tensor) (*tensor.Tensor, error) {
	d, err := Sub(pred, target)
	if err != nil {
		return nil, err
	}
	sq, err := Sqr(d)
	if err != nil {
		return nil, err
	}
	return Mean(sq)
}

// NLLLoss is the negative log-likelihood of log-probabilities logp (B, C)
// at the I64 targets (B,), averaged over the batch. Pair with LogSoftmax.
func NLLLoss(logp, target *tensor.Tensor) (*tensor.Tensor, error) {
	if logp.Rank() != 2 {
		return nil, fmt.Errorf("NLLLoss: logp must be (B, C), got %v", logp.Shape())
	}
	B, C := logp.Dim(0), logp.Dim(1)
	if err := validateClassTarget("NLLLoss", target, B); err != nil {
		return nil, err
	}
	oneHot, err := buildOneHot(target, B, C, logp.Device())
	if err != nil {
		return nil, err
	}
	if oneHot.DType() != logp.DType() {
		if oneHot, err = ops.Cast(oneHot, logp.DType()); err != nil {
			return nil, err
		}
	}
	picked, err := Mul(logp, oneHot)
	if err != nil {
		return nil, err
	}
	s, err := Sum(picked)
	if err != nil {
		return nil, err
	}
	scale, err := scalarLike(-1/float32(B), logp)
	if err != nil {
		return nil, err
	}
	return Mul(s, scale)
}

// BCEWithLogitsLoss is mean(max(x,0) - x·t + log(1 + e^{-|x|})), the
// numerically stable binary cross-entropy on logits x against targets t.
//
// The forward is the stable composed formula; the backward is the analytic
// derivative of the smooth loss, (σ(x) - t)/N w.r.t. x and -x/N w.r.t. t.
// (Differentiating the composed ReLU and Abs separately picks a zero
// subgradient for both at x = 0 and gives -1 instead of σ(0) - 1 = -0.5.)
func BCEWithLogitsLoss(logits, target *tensor.Tensor) (*tensor.Tensor, error) {
	outs, err := Apply(BCEWithLogitsFunction{}, []*tensor.Tensor{logits, target})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// BCEWithLogitsFunction is the fused stable BCE-with-logits (mean reduction).
type BCEWithLogitsFunction struct{}

func (BCEWithLogitsFunction) Name() string { return "BCEWithLogitsLoss" }

func (BCEWithLogitsFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("BCEWithLogitsLoss: want 2 inputs (logits, target), got %d", len(inputs))
	}
	x, t := inputs[0], inputs[1]
	pos, err := ops.Relu(x)
	if err != nil {
		return nil, err
	}
	xt, err := ops.Mul(x, t)
	if err != nil {
		return nil, err
	}
	ab, err := ops.Abs(x)
	if err != nil {
		return nil, err
	}
	negAbs, err := ops.Neg(ab)
	if err != nil {
		return nil, err
	}
	e, err := ops.Exp(negAbs)
	if err != nil {
		return nil, err
	}
	one, err := scalarLike(1, x)
	if err != nil {
		return nil, err
	}
	if e, err = ops.Add(e, one); err != nil {
		return nil, err
	}
	lg, err := ops.Log(e)
	if err != nil {
		return nil, err
	}
	per, err := ops.Sub(pos, xt)
	if err != nil {
		return nil, err
	}
	if per, err = ops.Add(per, lg); err != nil {
		return nil, err
	}
	loss, err := ops.Mean(per, ops.ReduceOptions{})
	if err != nil {
		return nil, err
	}
	ctx.SaveForBackward(x, t)
	ctx.Set("n", per.NumElements())
	ctx.Set("outShape", per.Shape().Dims())
	return []*tensor.Tensor{loss}, nil
}

func (BCEWithLogitsFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	saved := ctx.Saved()
	x, t := saved[0], saved[1]
	nAny, _ := ctx.Get("n")
	outAny, _ := ctx.Get("outShape")
	n := nAny.(int)
	outShape := outAny.([]int)

	// scale = g / N, broadcast against the elementwise loss shape.
	var invN *tensor.Tensor
	var err error
	if g.DType() == dtype.F32 {
		invN, err = tensor.Full(1.0/float32(n), []int{1}, g.Device())
	} else {
		invN, err = fullLike(1/float64(n), []int{1}, g)
	}
	if err != nil {
		return nil, err
	}
	scale, err := ops.Mul(g, invN)
	if err != nil {
		return nil, err
	}

	sig, err := ops.Sigmoid(x)
	if err != nil {
		return nil, err
	}
	d, err := ops.Sub(sig, t)
	if err != nil {
		return nil, err
	}
	gxRaw, err := ops.Mul(d, scale)
	if err != nil {
		return nil, err
	}
	gx, err := broadcastReduce(gxRaw, outShape, x.Shape().Dims())
	if err != nil {
		return nil, err
	}

	negX, err := ops.Neg(x)
	if err != nil {
		return nil, err
	}
	gtRaw, err := ops.Mul(negX, scale)
	if err != nil {
		return nil, err
	}
	gt, err := broadcastReduce(gtRaw, outShape, t.Shape().Dims())
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gx, gt}, nil
}

// broadcastReduce expands g to full (when an operand of a broadcasting
// forward was smaller than the result) and then sums it back down to want.
func broadcastReduce(g *tensor.Tensor, full, want []int) (*tensor.Tensor, error) {
	if !equalDims(g.Shape().Dims(), full) {
		z, err := tensor.Zeros(g.DType(), full, g.Device())
		if err != nil {
			return nil, err
		}
		if g, err = ops.Add(z, g); err != nil {
			return nil, err
		}
	}
	return reduceToShape(g, want)
}

// HuberLoss is mean(0.5·d² for |d| ≤ delta, delta·(|d| - 0.5·delta) otherwise).
func HuberLoss(pred, target *tensor.Tensor, delta float64) (*tensor.Tensor, error) {
	d, err := Sub(pred, target)
	if err != nil {
		return nil, err
	}
	ad, err := Abs(d)
	if err != nil {
		return nil, err
	}
	q, err := Clamp(ad, 0, delta) // min(|d|, delta)
	if err != nil {
		return nil, err
	}
	qq, err := Sqr(q)
	if err != nil {
		return nil, err
	}
	half, err := scalarLike(0.5, pred)
	if err != nil {
		return nil, err
	}
	quad, err := Mul(qq, half)
	if err != nil {
		return nil, err
	}
	lin, err := Sub(ad, q) // |d| - min(|d|, delta)
	if err != nil {
		return nil, err
	}
	dl, err := scalarLike(float32(delta), pred)
	if err != nil {
		return nil, err
	}
	if lin, err = Mul(lin, dl); err != nil {
		return nil, err
	}
	tot, err := Add(quad, lin)
	if err != nil {
		return nil, err
	}
	return Mean(tot)
}
