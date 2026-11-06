package training

import "math"

// LRSetter is satisfied by any optimizer whose base learning rate can be
// updated in place. Our SGD and AdamW both implement it.
type LRSetter interface {
	SetLR(lr float32)
}

// CosineLR schedules the learning rate with optional linear warmup:
//
//	step <  warmup:  lr = base * step / warmup
//	step >= warmup:  lr = min_lr + 0.5 * (base - min_lr) *
//	                         (1 + cos(π · (step - warmup) / (total - warmup)))
//
// Matches PyTorch's `torch.optim.lr_scheduler.CosineAnnealingLR` with a
// LambdaLR warmup prefix.
type CosineLR struct {
	Base    float32 // peak learning rate
	Min     float32 // floor after annealing
	Warmup  int     // linear-warmup steps
	Total   int     // total training steps
	optim   LRSetter
	current int
}

// NewCosineLR builds a schedule over `total` training steps with `warmup`
// linear-ramp steps leading to `base`, annealing down to `minLR` thereafter.
func NewCosineLR(opt LRSetter, base, minLR float32, warmup, total int) *CosineLR {
	s := &CosineLR{Base: base, Min: minLR, Warmup: warmup, Total: total, optim: opt}
	s.apply()
	return s
}

// Step advances the internal step counter and updates the optimizer LR.
func (s *CosineLR) Step() {
	s.current++
	s.apply()
}

// LR returns the learning rate scheduled for the current step.
func (s *CosineLR) LR() float32 {
	if s.current < s.Warmup {
		if s.Warmup == 0 {
			return s.Base
		}
		return s.Base * float32(s.current) / float32(s.Warmup)
	}
	if s.current >= s.Total {
		return s.Min
	}
	progress := float64(s.current-s.Warmup) / float64(s.Total-s.Warmup)
	cos := 0.5 * (1 + math.Cos(math.Pi*progress))
	return s.Min + (s.Base-s.Min)*float32(cos)
}

func (s *CosineLR) apply() { s.optim.SetLR(s.LR()) }
