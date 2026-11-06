package nn

import (
	"light/autograd"
	"light/tensor"
)

// Dropout applies inverted dropout with probability `P`. Matches PyTorch's
// nn.Dropout: in training mode each element is zeroed with probability P and
// survivors are scaled by 1/(1-P); in eval mode the input passes through
// unchanged. Toggle via `SetTraining(bool)`.
type Dropout struct {
	P        float32
	Training bool
}

// NewDropout constructs a Dropout module with probability p. Starts in
// training mode (Training=true).
func NewDropout(p float32) *Dropout {
	return &Dropout{P: p, Training: true}
}

func (d *Dropout) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	return autograd.Dropout(x, d.P, d.Training)
}

func (d *Dropout) Parameters() []*tensor.Tensor { return nil }

// SetTraining toggles train/eval mode. In eval mode Dropout is a no-op.
func (d *Dropout) SetTraining(v bool) { d.Training = v }
