// Package ops contains the public-facing tensor operations (Add, MatMul,
// Softmax, LayerNorm, …). Each function validates shapes/dtypes/devices,
// allocates an output tensor, then dispatches to the appropriate kernel via
// the registry in internal/registry.
//
// Operations that participate in autograd wrap their forward implementation in
// an autograd.Function and record a FunctionNode when any input has
// RequiresGrad set.
package ops
