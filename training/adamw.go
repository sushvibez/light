package training

import (
	"fmt"
	"math"

	"light/device"
	"light/dtype"
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

	m    []*tensor.Tensor
	v    []*tensor.Tensor
	step int
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

// NewAdamW constructs an AdamW optimizer.
func NewAdamW(params []*tensor.Tensor, opts AdamWOptions) (*AdamW, error) {
	opts.applyDefaults()
	if opts.LR <= 0 {
		return nil, fmt.Errorf("training.AdamW: lr must be positive")
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
	}
	o.step++
	t := float32(o.step)
	biasCorr1 := 1 - float32(math.Pow(float64(o.beta1), float64(t)))
	biasCorr2 := 1 - float32(math.Pow(float64(o.beta2), float64(t)))

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

		dev := p.Device()
		b1T, _ := scalarOnDevice(o.beta1, dev)
		b2T, _ := scalarOnDevice(o.beta2, dev)
		oneMinusB1, _ := scalarOnDevice(1-o.beta1, dev)
		oneMinusB2, _ := scalarOnDevice(1-o.beta2, dev)

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
		bc1T, _ := scalarOnDevice(biasCorr1, dev)
		bc2T, _ := scalarOnDevice(biasCorr2, dev)
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
		epsT, _ := scalarOnDevice(o.eps, dev)
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
			decayT, err := scalarOnDevice(1-o.lr*o.wd, dev)
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
		lrT, _ := scalarOnDevice(o.lr, dev)
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

// Ensure device import stays meaningful.
var _ = device.CPU
