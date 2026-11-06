package autograd

import (
	"fmt"

	"light/tensor"
)

// Function is the unit of differentiable computation. Forward runs the op
// and records anything Backward will need (via Context.SaveForBackward /
// Context.Set); Backward receives one gradient per forward output and returns
// one gradient per forward input (nil entries for inputs that do not require
// gradients).
type Function interface {
	// Name returns a short identifier used in error messages.
	Name() string
	// Forward runs the op on inputs, returning its outputs. Must be pure
	// w.r.t. the graph state — do not attach gradFns here.
	Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error)
	// Backward receives one gradient per forward output (nil means "zero")
	// and returns one gradient per forward input (nil to skip an input).
	Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error)
}

// Context carries per-node state from Forward to Backward.
type Context struct {
	saved []*tensor.Tensor
	meta  map[string]any
}

// SaveForBackward stores tensors the Backward pass will need (typically the
// inputs themselves, or intermediate results). They are held by the
// FunctionNode until the node is collected.
func (c *Context) SaveForBackward(ts ...*tensor.Tensor) {
	c.saved = append(c.saved, ts...)
}

// Saved returns the list of tensors previously saved.
func (c *Context) Saved() []*tensor.Tensor { return c.saved }

// Set stores an arbitrary piece of metadata, e.g. a shape or axis the
// backward pass needs.
func (c *Context) Set(key string, v any) {
	if c.meta == nil {
		c.meta = make(map[string]any)
	}
	c.meta[key] = v
}

// Get retrieves metadata stored with Set. The boolean indicates presence.
func (c *Context) Get(key string) (any, bool) {
	v, ok := c.meta[key]
	return v, ok
}

// FunctionNode is one node in the dynamic autograd DAG. It holds the
// Function, the Context, the inputs that produced the node (so Backward can
// route gradients to leaf tensors), and the arity of outputs.
type FunctionNode struct {
	fn       Function
	ctx      *Context
	inputs   []*tensor.Tensor // forward inputs, in order
	nOutputs int
}

// Inputs returns the forward inputs (shared — do not mutate).
func (n *FunctionNode) Inputs() []*tensor.Tensor { return n.inputs }

// Function returns the underlying Function (used by error messages).
func (n *FunctionNode) Function() Function { return n.fn }

// Apply runs a Function on the given inputs. If any input has
// RequiresGrad, each output is wired into the autograd DAG: its GradFn is
// set to a fresh FunctionNode over fn with the inputs saved. Outputs always
// share the shape/dtype/device chosen by fn.Forward.
func Apply(fn Function, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	needsGrad := false
	for _, in := range inputs {
		if in != nil && (in.RequiresGrad() || in.GradFn() != nil) {
			needsGrad = true
			break
		}
	}
	ctx := &Context{}
	outputs, err := fn.Forward(ctx, inputs)
	if err != nil {
		return nil, fmt.Errorf("autograd %s: forward: %w", fn.Name(), err)
	}
	if !needsGrad {
		return outputs, nil
	}
	node := &FunctionNode{
		fn:       fn,
		ctx:      ctx,
		inputs:   inputs,
		nOutputs: len(outputs),
	}
	for i, out := range outputs {
		out.SetRequiresGrad(true)
		out.SetGradFn(node, i)
	}
	return outputs, nil
}
