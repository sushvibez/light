package models

import "light/tensor"

// SetFluxDebug installs (or clears, with nil) the intermediate hook of the
// first double block; for tests comparing against a traced reference.
func SetFluxDebug(f func(name string, t *tensor.Tensor)) { fluxDebug = f }
