// Package autograd implements PyTorch-style reverse-mode automatic
// differentiation. Every op that may participate in gradient computation is
// an autograd.Function with static Forward and Backward methods; applying a
// Function records a FunctionNode into the dynamic DAG that Backward() walks
// in reverse topological order.
//
// v1 is first-order, reverse-mode only. In-place ops are tracked via a version
// counter on the Tensor; saved-tensor mutation produces a descriptive error.
// Hooks (register_hook equivalents) are deferred to Phase 8.
package autograd
