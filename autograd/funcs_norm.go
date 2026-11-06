package autograd

import (
	"fmt"

	"light/ops"
	"light/tensor"
)

// LayerNormFunction applies LayerNorm over the last axis:
//
//	μ = mean(x, axis=-1, keepdims=True)
//	σ² = mean((x-μ)², axis=-1, keepdims=True)
//	σ = sqrt(σ² + ε)
//	x̂ = (x - μ) / σ
//	y = x̂ * γ + β
//
// Backward (per-element along the last axis, with D = last-axis size):
//
//	g' = g · γ                       (shape matches x)
//	dL/dβ = Σ over leading axes of g
//	dL/dγ = Σ over leading axes of (g · x̂)
//	dL/dx = (1/σ) · (g' - mean(g', -1, kd=T) - x̂ · mean(g' · x̂, -1, kd=T))
//
// The mean is over the last axis; gamma/beta are rank-1 of length D.
type LayerNormFunction struct {
	eps float32
}

func (LayerNormFunction) Name() string { return "LayerNorm" }

func (f LayerNormFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 3 {
		return nil, fmt.Errorf("LayerNorm: need (x, gamma, beta)")
	}
	x, gamma, beta := inputs[0], inputs[1], inputs[2]
	if gamma.Rank() != 1 || beta.Rank() != 1 {
		return nil, fmt.Errorf("LayerNorm: gamma/beta must be rank-1")
	}
	if gamma.Dim(0) != x.Dim(-1) {
		return nil, fmt.Errorf("LayerNorm: gamma length %d != x last dim %d",
			gamma.Dim(0), x.Dim(-1))
	}

	mu, err := ops.Mean(x, ops.ReduceOptions{Axes: []int{-1}, KeepDims: true})
	if err != nil {
		return nil, err
	}
	xMu, err := ops.Sub(x, mu)
	if err != nil {
		return nil, err
	}
	xMuSq, err := ops.Mul(xMu, xMu)
	if err != nil {
		return nil, err
	}
	variance, err := ops.Mean(xMuSq, ops.ReduceOptions{Axes: []int{-1}, KeepDims: true})
	if err != nil {
		return nil, err
	}
	epsT, err := tensor.Full(f.eps, []int{1}, x.Device())
	if err != nil {
		return nil, err
	}
	varPlusEps, err := ops.Add(variance, epsT)
	if err != nil {
		return nil, err
	}
	sigma, err := ops.Sqrt(varPlusEps)
	if err != nil {
		return nil, err
	}
	xHat, err := ops.Div(xMu, sigma)
	if err != nil {
		return nil, err
	}
	scaled, err := ops.Mul(xHat, gamma) // broadcasts gamma across leading axes
	if err != nil {
		return nil, err
	}
	y, err := ops.Add(scaled, beta)
	if err != nil {
		return nil, err
	}

	ctx.SaveForBackward(xHat, sigma, gamma)
	return []*tensor.Tensor{y}, nil
}

func (LayerNormFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	saved := ctx.Saved()
	xHat, sigma, gamma := saved[0], saved[1], saved[2]

	// Collapse leading axes for beta/gamma grads.
	rank := g.Rank()
	leadingAxes := make([]int, rank-1)
	for i := range rank - 1 {
		leadingAxes[i] = i
	}
	// dL/dbeta = sum over leading axes of g.
	gBeta, err := ops.Sum(g, ops.ReduceOptions{Axes: leadingAxes})
	if err != nil {
		return nil, err
	}
	// dL/dgamma = sum over leading axes of g * x_hat.
	gxHat, err := ops.Mul(g, xHat)
	if err != nil {
		return nil, err
	}
	gGamma, err := ops.Sum(gxHat, ops.ReduceOptions{Axes: leadingAxes})
	if err != nil {
		return nil, err
	}

	// dL/dx_hat = g * gamma (broadcasts).
	gPrime, err := ops.Mul(g, gamma)
	if err != nil {
		return nil, err
	}
	// mean(g', -1, kd=T) and mean(g' * x_hat, -1, kd=T).
	meanGp, err := ops.Mean(gPrime, ops.ReduceOptions{Axes: []int{-1}, KeepDims: true})
	if err != nil {
		return nil, err
	}
	gpXh, err := ops.Mul(gPrime, xHat)
	if err != nil {
		return nil, err
	}
	meanGpXh, err := ops.Mean(gpXh, ops.ReduceOptions{Axes: []int{-1}, KeepDims: true})
	if err != nil {
		return nil, err
	}
	// inner = g' - meanGp - x_hat * meanGpXh
	inner1, err := ops.Sub(gPrime, meanGp)
	if err != nil {
		return nil, err
	}
	xhTimesMean, err := ops.Mul(xHat, meanGpXh)
	if err != nil {
		return nil, err
	}
	inner, err := ops.Sub(inner1, xhTimesMean)
	if err != nil {
		return nil, err
	}
	gX, err := ops.Div(inner, sigma)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gX, gGamma, gBeta}, nil
}

// LayerNorm is the autograd-aware LayerNorm over the last axis. gamma and
// beta are rank-1 of length x.Dim(-1).
func LayerNorm(x, gamma, beta *tensor.Tensor, eps float32) (*tensor.Tensor, error) {
	outs, err := Apply(LayerNormFunction{eps: eps}, []*tensor.Tensor{x, gamma, beta})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// RMSNormFunction applies Llama-style RMSNorm over the last axis:
//
//	σ² = mean(x², axis=-1, keepdims=True)
//	r  = 1 / sqrt(σ² + ε)
//	y  = x * r * γ
//
// No mean subtraction, no β. Backward (with D = last-axis size):
//
//	g'  = g · γ
//	dL/dγ = Σ over leading axes of (g · x · r)
//	dL/dx = r · (g' - x · r² · mean(g' · x, -1, kd=True))
type RMSNormFunction struct {
	eps float32
}

func (RMSNormFunction) Name() string { return "RMSNorm" }

func (f RMSNormFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("RMSNorm: need (x, gamma)")
	}
	x, gamma := inputs[0], inputs[1]
	if gamma.Rank() != 1 || gamma.Dim(0) != x.Dim(-1) {
		return nil, fmt.Errorf("RMSNorm: gamma shape mismatch")
	}
	xSq, err := ops.Mul(x, x)
	if err != nil {
		return nil, err
	}
	variance, err := ops.Mean(xSq, ops.ReduceOptions{Axes: []int{-1}, KeepDims: true})
	if err != nil {
		return nil, err
	}
	epsT, err := tensor.Full(f.eps, []int{1}, x.Device())
	if err != nil {
		return nil, err
	}
	varPlusEps, err := ops.Add(variance, epsT)
	if err != nil {
		return nil, err
	}
	sigma, err := ops.Sqrt(varPlusEps)
	if err != nil {
		return nil, err
	}
	one, err := tensor.Full[float32](1.0, []int{1}, x.Device())
	if err != nil {
		return nil, err
	}
	r, err := ops.Div(one, sigma)
	if err != nil {
		return nil, err
	}
	xr, err := ops.Mul(x, r)
	if err != nil {
		return nil, err
	}
	y, err := ops.Mul(xr, gamma)
	if err != nil {
		return nil, err
	}
	ctx.SaveForBackward(x, gamma, r)
	return []*tensor.Tensor{y}, nil
}

func (RMSNormFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	saved := ctx.Saved()
	x, gamma, r := saved[0], saved[1], saved[2]

	// dL/dgamma = sum over leading axes of g * x * r.
	xr, err := ops.Mul(x, r)
	if err != nil {
		return nil, err
	}
	gxr, err := ops.Mul(g, xr)
	if err != nil {
		return nil, err
	}
	rank := g.Rank()
	leading := make([]int, rank-1)
	for i := range rank - 1 {
		leading[i] = i
	}
	gGamma, err := ops.Sum(gxr, ops.ReduceOptions{Axes: leading})
	if err != nil {
		return nil, err
	}

	// dL/dx = r * (g' - x * r² * mean(g' * x, -1, kd=T)).
	gPrime, err := ops.Mul(g, gamma)
	if err != nil {
		return nil, err
	}
	gpX, err := ops.Mul(gPrime, x)
	if err != nil {
		return nil, err
	}
	meanGPX, err := ops.Mean(gpX, ops.ReduceOptions{Axes: []int{-1}, KeepDims: true})
	if err != nil {
		return nil, err
	}
	rSq, err := ops.Mul(r, r)
	if err != nil {
		return nil, err
	}
	xR2, err := ops.Mul(x, rSq)
	if err != nil {
		return nil, err
	}
	correction, err := ops.Mul(xR2, meanGPX)
	if err != nil {
		return nil, err
	}
	inner, err := ops.Sub(gPrime, correction)
	if err != nil {
		return nil, err
	}
	gX, err := ops.Mul(inner, r)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gX, gGamma}, nil
}

// RMSNorm is the autograd-aware RMSNorm over the last axis.
func RMSNorm(x, gamma *tensor.Tensor, eps float32) (*tensor.Tensor, error) {
	outs, err := Apply(RMSNormFunction{eps: eps}, []*tensor.Tensor{x, gamma})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}
