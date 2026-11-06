// Package training holds optimizers (SGD, AdamW, Adafactor), learning-rate
// schedulers (cosine, linear, warmup variants), gradient clipping utilities,
// mixed-precision helpers, LoRA adapters, and a minimal DataLoader primitive.
package training

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// Optimizer is the common interface for parameter-update algorithms. Step
// consumes each parameter's accumulated .Grad() and writes new values into
// the parameter's storage in place via tensor.CopyFrom; ZeroGrad clears all
// gradients.
type Optimizer interface {
	Step() error
	ZeroGrad()
	Parameters() []*tensor.Tensor
}

// SGD is vanilla stochastic gradient descent with optional momentum and
// weight decay. Device-agnostic: parameters may live on CPU or CUDA, and
// updates are computed via device-routed ops + `tensor.CopyFrom` for the
// in-place write.
type SGD struct {
	params      []*tensor.Tensor
	lr          float32
	momentum    float32
	weightDecay float32
	velocity    []*tensor.Tensor // same shape as each param; nil until first Step
}

// SGDOptions controls SGD hyperparameters.
type SGDOptions struct {
	LR          float32 // required, must be > 0
	Momentum    float32 // if > 0, use heavy-ball momentum
	WeightDecay float32 // adds wd*p to the effective gradient
}

// NewSGD constructs an SGD optimizer over the given parameters.
func NewSGD(params []*tensor.Tensor, opts SGDOptions) (*SGD, error) {
	if opts.LR <= 0 {
		return nil, fmt.Errorf("training.SGD: lr must be positive (got %v)", opts.LR)
	}
	for i, p := range params {
		if p == nil {
			return nil, fmt.Errorf("training.SGD: params[%d] is nil", i)
		}
		if p.DType() != dtype.F32 {
			return nil, fmt.Errorf("training.SGD: params[%d] dtype %v, only F32 supported", i, p.DType())
		}
		p.SetRequiresGrad(true)
	}
	return &SGD{
		params:      params,
		lr:          opts.LR,
		momentum:    opts.Momentum,
		weightDecay: opts.WeightDecay,
	}, nil
}

func (o *SGD) Parameters() []*tensor.Tensor { return o.params }

// SetLR updates the learning rate (for LR schedulers).
func (o *SGD) SetLR(lr float32) { o.lr = lr }

func (o *SGD) ZeroGrad() {
	for _, p := range o.params {
		p.SetGrad(nil)
	}
}

// Step applies one parameter update per parameter that has a gradient.
// Mathematically: eff_grad = grad + wd*p; v = momentum*v + eff_grad;
// p_new = p - lr * (v if momentum else eff_grad). p is updated in-place.
func (o *SGD) Step() error {
	if o.momentum != 0 && o.velocity == nil {
		o.velocity = make([]*tensor.Tensor, len(o.params))
	}
	for i, p := range o.params {
		g := p.Grad()
		if g == nil {
			continue
		}
		if !p.Device().SameAs(g.Device()) {
			return fmt.Errorf("training.SGD: param %d on %v but grad on %v",
				i, p.Device(), g.Device())
		}

		effGrad := g
		if o.weightDecay != 0 {
			wdT, err := scalarOnDevice(o.weightDecay, p.Device())
			if err != nil {
				return err
			}
			wdP, err := ops.Mul(p, wdT) // scalar broadcasts
			if err != nil {
				return err
			}
			effGrad, err = ops.Add(effGrad, wdP)
			if err != nil {
				return err
			}
		}

		step := effGrad
		if o.momentum != 0 {
			if o.velocity[i] == nil {
				v, err := tensor.Zeros(dtype.F32, p.Shape().Dims(), p.Device())
				if err != nil {
					return err
				}
				o.velocity[i] = v
			}
			// v_new = momentum * v + effGrad
			muT, err := scalarOnDevice(o.momentum, p.Device())
			if err != nil {
				return err
			}
			muV, err := ops.Mul(o.velocity[i], muT)
			if err != nil {
				return err
			}
			vNew, err := ops.Add(muV, effGrad)
			if err != nil {
				return err
			}
			if err := o.velocity[i].CopyFrom(vNew); err != nil {
				return err
			}
			step = o.velocity[i]
		}

		// p_new = p - lr * step
		lrT, err := scalarOnDevice(o.lr, p.Device())
		if err != nil {
			return err
		}
		lrStep, err := ops.Mul(step, lrT)
		if err != nil {
			return err
		}
		pNew, err := ops.Sub(p, lrStep)
		if err != nil {
			return err
		}
		if err := p.CopyFrom(pNew); err != nil {
			return err
		}
		// Clear any stale gradFn left on p by prior autograd.
		p.SetGradFn(nil, 0)
	}
	return nil
}

// scalarOnDevice returns a [1]-shaped F32 tensor holding v on dev.
func scalarOnDevice(v float32, dev device.Device) (*tensor.Tensor, error) {
	return tensor.Full(v, []int{1}, dev)
}
