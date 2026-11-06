package nn

import "light/tensor"

// Module is the common interface for composable neural-network building
// blocks. Concrete modules may own parameters (trainable `*Tensor`s, returned
// by Parameters) and implement Forward.
//
// Phase 4 will extend this with `Backward` / `ZeroGrad` once the autograd
// engine lands; for Phase 2-3 inference usage, Forward + Parameters is enough.
type Module interface {
	Forward(x *tensor.Tensor) (*tensor.Tensor, error)
	Parameters() []*tensor.Tensor
}

// Sequential runs a chain of Modules left-to-right.
type Sequential struct {
	modules []Module
}

// NewSequential composes the given modules into a single Module.
func NewSequential(modules ...Module) *Sequential {
	return &Sequential{modules: modules}
}

// Forward passes x through each module in order.
func (s *Sequential) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	var err error
	for _, m := range s.modules {
		x, err = m.Forward(x)
		if err != nil {
			return nil, err
		}
	}
	return x, nil
}

// Parameters returns the concatenated parameters of every sub-module.
func (s *Sequential) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, m := range s.modules {
		ps = append(ps, m.Parameters()...)
	}
	return ps
}

// Len returns the number of sub-modules.
func (s *Sequential) Len() int { return len(s.modules) }

// At returns the i-th sub-module.
func (s *Sequential) At(i int) Module { return s.modules[i] }
