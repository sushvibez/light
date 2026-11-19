package training

import (
	"fmt"
	"math"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/parallel"
	"light/ops"
	"light/tensor"
)

// AdamW is the decoupled-weight-decay variant of Adam (Loshchilov & Hutter,
// 2019) — the standard optimizer for modern transformer training.
//
// Update rule per step t:
//
//	m_t = β1 * m_{t-1} + (1 - β1) * g
//	v_t = β2 * v_{t-1} + (1 - β2) * g²
//	m̂ = m_t / (1 - β1^t)
//	v̂ = v_t / (1 - β2^t)
//	p  ← p - lr * (m̂ / (√v̂ + ε) + wd * p)
//
// Weight decay is applied directly to the parameter (not rolled into the
// gradient estimate); this is the key behavioural difference from Adam.
// Device-agnostic: every update is expressed via ops.* so parameters on CPU
// or CUDA all take the same code path.
type AdamW struct {
	params []*tensor.Tensor
	lr     float32
	beta1  float32
	beta2  float32
	eps    float32
	wd     float32

	m []*tensor.Tensor
	v []*tensor.Tensor
	// steps[i] counts the updates parameter i has actually received; bias
	// correction uses it rather than a global step, so a parameter that
	// skipped steps (no gradient) starts its correction at t = 1.
	steps []int

	forceGeneric bool // tests: always take the tensor-op path
}

// AdamWOptions controls hyperparameters. Defaults match PyTorch's
// torch.optim.AdamW where sensible.
type AdamWOptions struct {
	LR          float32
	Beta1       float32
	Beta2       float32
	Eps         float32
	WeightDecay float32
}

func (o *AdamWOptions) applyDefaults() {
	if o.LR == 0 {
		o.LR = 1e-3
	}
	if o.Beta1 == 0 {
		o.Beta1 = 0.9
	}
	if o.Beta2 == 0 {
		o.Beta2 = 0.999
	}
	if o.Eps == 0 {
		o.Eps = 1e-8
	}
}

// validate rejects hyperparameters that make the update meaningless or
// non-finite: LR and Eps must be finite and positive, both betas in [0, 1),
// and WeightDecay finite and non-negative.
func (o *AdamWOptions) validate() error {
	finite := func(x float32) bool {
		f := float64(x)
		return !math.IsNaN(f) && !math.IsInf(f, 0)
	}
	if !finite(o.LR) || o.LR <= 0 {
		return fmt.Errorf("training.AdamW: lr must be finite and positive (got %v)", o.LR)
	}
	if !finite(o.Eps) || o.Eps <= 0 {
		return fmt.Errorf("training.AdamW: eps must be finite and positive (got %v)", o.Eps)
	}
	if !finite(o.Beta1) || o.Beta1 < 0 || o.Beta1 >= 1 {
		return fmt.Errorf("training.AdamW: beta1 must be in [0, 1) (got %v)", o.Beta1)
	}
	if !finite(o.Beta2) || o.Beta2 < 0 || o.Beta2 >= 1 {
		return fmt.Errorf("training.AdamW: beta2 must be in [0, 1) (got %v)", o.Beta2)
	}
	if !finite(o.WeightDecay) || o.WeightDecay < 0 {
		return fmt.Errorf("training.AdamW: weight decay must be finite and non-negative (got %v)", o.WeightDecay)
	}
	return nil
}

// NewAdamW constructs an AdamW optimizer.
func NewAdamW(params []*tensor.Tensor, opts AdamWOptions) (*AdamW, error) {
	opts.applyDefaults()
	if err := opts.validate(); err != nil {
		return nil, err
	}
	for i, p := range params {
		if p == nil {
			return nil, fmt.Errorf("training.AdamW: params[%d] is nil", i)
		}
		if p.DType() != dtype.F32 {
			return nil, fmt.Errorf("training.AdamW: params[%d] dtype %v, only F32 supported", i, p.DType())
		}
		p.SetRequiresGrad(true)
	}
	return &AdamW{
		params: params,
		lr:     opts.LR,
		beta1:  opts.Beta1,
		beta2:  opts.Beta2,
		eps:    opts.Eps,
		wd:     opts.WeightDecay,
	}, nil
}

func (o *AdamW) Parameters() []*tensor.Tensor { return o.params }

// SetLR updates the learning rate (for LR schedulers).
func (o *AdamW) SetLR(lr float32) { o.lr = lr }

func (o *AdamW) ZeroGrad() {
	for _, p := range o.params {
		p.SetGrad(nil)
	}
}

// Step applies one AdamW update to every parameter that has a gradient.
func (o *AdamW) Step() error {
	if o.m == nil {
		o.m = make([]*tensor.Tensor, len(o.params))
		o.v = make([]*tensor.Tensor, len(o.params))
		o.steps = make([]int, len(o.params))
	}

	for i, p := range o.params {
		g := p.Grad()
		if g == nil {
			continue
		}
		if !p.Device().SameAs(g.Device()) {
			return fmt.Errorf("training.AdamW: param %d on %v but grad on %v",
				i, p.Device(), g.Device())
		}
		if o.m[i] == nil {
			var err error
			o.m[i], err = tensor.Zeros(dtype.F32, p.Shape().Dims(), p.Device())
			if err != nil {
				return err
			}
			o.v[i], err = tensor.Zeros(dtype.F32, p.Shape().Dims(), p.Device())
			if err != nil {
				return err
			}
		}

		o.steps[i]++
		t := float64(o.steps[i])
		biasCorr1 := 1 - float32(math.Pow(float64(o.beta1), t))
		biasCorr2 := 1 - float32(math.Pow(float64(o.beta2), t))

		decay := 1 - o.lr*o.wd
		if !o.forceGeneric && o.stepCPU(i, g, biasCorr1, biasCorr2, decay) {
			p.SetGradFn(nil, 0)
			continue
		}

		dev := p.Device()
		sc, err := scalarsOnDevice(dev, o.beta1, o.beta2, 1-o.beta1, 1-o.beta2,
			biasCorr1, biasCorr2, o.eps, o.lr)
		if err != nil {
			return fmt.Errorf("training.AdamW: param %d: %w", i, err)
		}
		b1T, b2T, oneMinusB1, oneMinusB2 := sc[0], sc[1], sc[2], sc[3]
		bc1T, bc2T, epsT, lrT := sc[4], sc[5], sc[6], sc[7]

		// m = β1*m + (1-β1)*g
		mPart1, err := ops.Mul(o.m[i], b1T)
		if err != nil {
			return err
		}
		mPart2, err := ops.Mul(g, oneMinusB1)
		if err != nil {
			return err
		}
		mNew, err := ops.Add(mPart1, mPart2)
		if err != nil {
			return err
		}
		if err := o.m[i].CopyFrom(mNew); err != nil {
			return err
		}

		// v = β2*v + (1-β2)*g²
		gSq, err := ops.Mul(g, g)
		if err != nil {
			return err
		}
		vPart1, err := ops.Mul(o.v[i], b2T)
		if err != nil {
			return err
		}
		vPart2, err := ops.Mul(gSq, oneMinusB2)
		if err != nil {
			return err
		}
		vNew, err := ops.Add(vPart1, vPart2)
		if err != nil {
			return err
		}
		if err := o.v[i].CopyFrom(vNew); err != nil {
			return err
		}

		// m_hat = m / bc1; v_hat = v / bc2
		mHat, err := ops.Div(o.m[i], bc1T)
		if err != nil {
			return err
		}
		vHat, err := ops.Div(o.v[i], bc2T)
		if err != nil {
			return err
		}
		vHatSqrt, err := ops.Sqrt(vHat)
		if err != nil {
			return err
		}
		denom, err := ops.Add(vHatSqrt, epsT)
		if err != nil {
			return err
		}
		adamStep, err := ops.Div(mHat, denom)
		if err != nil {
			return err
		}

		// Weight decay: p ← p - lr*wd*p  ⇒ p ← p * (1 - lr*wd)
		if o.wd != 0 {
			decayT, err := scalarOnDevice(decay, dev)
			if err != nil {
				return err
			}
			pDecayed, err := ops.Mul(p, decayT)
			if err != nil {
				return err
			}
			if err := p.CopyFrom(pDecayed); err != nil {
				return err
			}
		}

		// p ← p - lr * adamStep
		scaled, err := ops.Mul(adamStep, lrT)
		if err != nil {
			return err
		}
		pNew, err := ops.Sub(p, scaled)
		if err != nil {
			return err
		}
		if err := p.CopyFrom(pNew); err != nil {
			return err
		}
		p.SetGradFn(nil, 0)
	}
	return nil
}

// stepCPU is Step's update for parameter i fused into one pass, when p, its
// gradient and both moments are contiguous F32 CPU tensors of one shape;
// it reports false (nothing written) otherwise. Each element goes through
// the same float32 operations, in the same order and with the same
// rounding, as the tensor-op path (the float32() conversions keep the
// compiler from fusing a multiply-add), so results are bit-identical; it
// just skips the eleven parameter-sized intermediates and eight scalar
// tensors.
func (o *AdamW) stepCPU(i int, g *tensor.Tensor, bc1, bc2, decay float32) bool {
	p := o.params[i]
	ps, ok1 := cpuF32(p)
	gs, ok2 := cpuF32(g)
	ms, ok3 := cpuF32(o.m[i])
	vs, ok4 := cpuF32(o.v[i])
	if !ok1 || !ok2 || !ok3 || !ok4 || !g.Shape().Equal(p.Shape()) {
		return false
	}
	// In-place writes, as CopyFrom: invalidate saved autograd values.
	p.Storage().BumpVersion()
	o.m[i].Storage().BumpVersion()
	o.v[i].Storage().BumpVersion()
	b1, b2, omb1, omb2 := o.beta1, o.beta2, 1-o.beta1, 1-o.beta2
	eps, lr, wd := o.eps, o.lr, o.wd != 0
	parallel.For(len(ps), parallel.Grain, func(lo, hi int) {
		for j := lo; j < hi; j++ {
			gj := gs[j]
			m := float32(ms[j]*b1) + float32(gj*omb1)
			v := float32(vs[j]*b2) + float32(float32(gj*gj)*omb2)
			ms[j], vs[j] = m, v
			den := float32(math.Sqrt(float64(v/bc2))) + eps
			step := (m / bc1) / den
			pj := ps[j]
			if wd {
				pj = float32(pj * decay)
			}
			ps[j] = pj - float32(step*lr)
		}
	})
	return true
}

// cpuF32 returns t's elements as a mutable slice when t is a contiguous,
// zero-offset F32 tensor on the CPU.
func cpuF32(t *tensor.Tensor) ([]float32, bool) {
	if !device.IsCPU(t.Device()) || t.DType() != dtype.F32 || !t.IsContiguous() || t.Offset() != 0 {
		return nil, false
	}
	n := t.NumElements()
	if n == 0 {
		return nil, true
	}
	buf := t.Storage().CPUBytes()
	return unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(buf))), n), true
}
