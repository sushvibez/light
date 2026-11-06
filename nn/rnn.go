package nn

import (
	"fmt"
	"math"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/tensor"
)

// LSTM is a single-layer, unidirectional long short-term memory cell with
// PyTorch's parameterisation and gate order (i, f, g, o):
//
//	gates = x·Wihᵀ + bih + h·Whhᵀ + bhh          (B, 4H)
//	c' = σ(f)·c + σ(i)·tanh(g);  h' = σ(o)·tanh(c')
//
// Weights are stored exactly as `torch.nn.LSTM` / Candle `candle_nn::lstm`
// ship them (Wih (4H, I), Whh (4H, H), biases (4H)), so checkpoints load
// without reshuffling. Everything is built from autograd ops, so the cell
// trains and runs on CPU or CUDA.
type LSTM struct {
	Wih, Whh *tensor.Tensor // (4H, I), (4H, H)
	Bih, Bhh *tensor.Tensor // (4H) or nil
	Hidden   int
}

// LSTMState is (h, c), each (B, H).
type LSTMState struct{ H, C *tensor.Tensor }

// NewLSTM allocates an LSTM with uniform(-1/√H, 1/√H) init, as PyTorch.
func NewLSTM(inputSize, hidden int, bias bool, dev device.Device) (*LSTM, error) {
	wih, err := uniformInit(4*hidden, inputSize, hidden, dev)
	if err != nil {
		return nil, err
	}
	whh, err := uniformInit(4*hidden, hidden, hidden, dev)
	if err != nil {
		return nil, err
	}
	l := &LSTM{Wih: wih, Whh: whh, Hidden: hidden}
	if bias {
		if l.Bih, err = uniformInit(4*hidden, 0, hidden, dev); err != nil {
			return nil, err
		}
		if l.Bhh, err = uniformInit(4*hidden, 0, hidden, dev); err != nil {
			return nil, err
		}
	}
	return l, nil
}

// uniformInit returns a (rows, cols) — or (rows,) when cols == 0 — tensor
// drawn from U(-1/√hidden, 1/√hidden) with requires_grad set.
func uniformInit(rows, cols, hidden int, dev device.Device) (*tensor.Tensor, error) {
	dims := []int{rows}
	if cols > 0 {
		dims = []int{rows, cols}
	}
	t, err := tensor.Rand(dtype.F32, dims, device.CPU())
	if err != nil {
		return nil, err
	}
	k := float32(1 / math.Sqrt(float64(hidden)))
	data := tensor.MustSlice[float32](t)
	for i, v := range data {
		data[i] = (2*v - 1) * k
	}
	out, err := tensor.New(data, dims, dev)
	if err != nil {
		return nil, err
	}
	out.SetRequiresGrad(true)
	return out, nil
}

// Parameters returns the trainable tensors.
func (l *LSTM) Parameters() []*tensor.Tensor {
	ps := []*tensor.Tensor{l.Wih, l.Whh}
	if l.Bih != nil {
		ps = append(ps, l.Bih, l.Bhh)
	}
	return ps
}

// ZeroState returns zero (h, c) for batch B on the weights' device.
func (l *LSTM) ZeroState(B int) (LSTMState, error) {
	h, err := tensor.Zeros(l.Wih.DType(), []int{B, l.Hidden}, l.Wih.Device())
	if err != nil {
		return LSTMState{}, err
	}
	c, err := tensor.Zeros(l.Wih.DType(), []int{B, l.Hidden}, l.Wih.Device())
	if err != nil {
		return LSTMState{}, err
	}
	return LSTMState{H: h, C: c}, nil
}

// Step advances one time step on x (B, I).
func (l *LSTM) Step(x *tensor.Tensor, st LSTMState) (LSTMState, error) {
	gates, err := affineGates(x, st.H, l.Wih, l.Whh, l.Bih, l.Bhh)
	if err != nil {
		return LSTMState{}, err
	}
	H := l.Hidden
	chunk := func(k int) (*tensor.Tensor, error) { return autograd.Narrow(gates, 1, k*H, H) }
	i, err := chunk(0)
	if err != nil {
		return LSTMState{}, err
	}
	f, err := chunk(1)
	if err != nil {
		return LSTMState{}, err
	}
	g, err := chunk(2)
	if err != nil {
		return LSTMState{}, err
	}
	o, err := chunk(3)
	if err != nil {
		return LSTMState{}, err
	}
	if i, err = autograd.Sigmoid(i); err != nil {
		return LSTMState{}, err
	}
	if f, err = autograd.Sigmoid(f); err != nil {
		return LSTMState{}, err
	}
	if g, err = autograd.Tanh(g); err != nil {
		return LSTMState{}, err
	}
	if o, err = autograd.Sigmoid(o); err != nil {
		return LSTMState{}, err
	}
	fc, err := autograd.Mul(f, st.C)
	if err != nil {
		return LSTMState{}, err
	}
	ig, err := autograd.Mul(i, g)
	if err != nil {
		return LSTMState{}, err
	}
	c, err := autograd.Add(fc, ig)
	if err != nil {
		return LSTMState{}, err
	}
	tc, err := autograd.Tanh(c)
	if err != nil {
		return LSTMState{}, err
	}
	h, err := autograd.Mul(o, tc)
	if err != nil {
		return LSTMState{}, err
	}
	return LSTMState{H: h, C: c}, nil
}

// Forward runs the whole sequence x (B, T, I) from a zero state and returns
// the hidden states (B, T, H) plus the final state.
func (l *LSTM) Forward(x *tensor.Tensor) (*tensor.Tensor, LSTMState, error) {
	if x.Rank() != 3 {
		return nil, LSTMState{}, fmt.Errorf("LSTM.Forward: want (B, T, I), got %v", x.Shape())
	}
	st, err := l.ZeroState(x.Dim(0))
	if err != nil {
		return nil, LSTMState{}, err
	}
	outs := make([]*tensor.Tensor, x.Dim(1))
	for t := range outs {
		xt, err := stepInput(x, t)
		if err != nil {
			return nil, LSTMState{}, err
		}
		if st, err = l.Step(xt, st); err != nil {
			return nil, LSTMState{}, err
		}
		if outs[t], err = autograd.Reshape(st.H, x.Dim(0), 1, l.Hidden); err != nil {
			return nil, LSTMState{}, err
		}
	}
	seq, err := autograd.Concat(outs, 1)
	if err != nil {
		return nil, LSTMState{}, err
	}
	return seq, st, nil
}

// GRU is a single-layer gated recurrent unit with PyTorch's gate order
// (r, z, n):
//
//	r = σ(x·Wirᵀ + bir + h·Whrᵀ + bhr)
//	z = σ(x·Wizᵀ + biz + h·Whzᵀ + bhz)
//	n = tanh(x·Winᵀ + bin + r·(h·Whnᵀ + bhn))
//	h' = (1 - z)·n + z·h
//
// Weights: Wih (3H, I), Whh (3H, H), biases (3H), as torch.nn.GRU.
type GRU struct {
	Wih, Whh *tensor.Tensor
	Bih, Bhh *tensor.Tensor
	Hidden   int
}

// NewGRU allocates a GRU with uniform(-1/√H, 1/√H) init.
func NewGRU(inputSize, hidden int, bias bool, dev device.Device) (*GRU, error) {
	wih, err := uniformInit(3*hidden, inputSize, hidden, dev)
	if err != nil {
		return nil, err
	}
	whh, err := uniformInit(3*hidden, hidden, hidden, dev)
	if err != nil {
		return nil, err
	}
	g := &GRU{Wih: wih, Whh: whh, Hidden: hidden}
	if bias {
		if g.Bih, err = uniformInit(3*hidden, 0, hidden, dev); err != nil {
			return nil, err
		}
		if g.Bhh, err = uniformInit(3*hidden, 0, hidden, dev); err != nil {
			return nil, err
		}
	}
	return g, nil
}

// Parameters returns the trainable tensors.
func (g *GRU) Parameters() []*tensor.Tensor {
	ps := []*tensor.Tensor{g.Wih, g.Whh}
	if g.Bih != nil {
		ps = append(ps, g.Bih, g.Bhh)
	}
	return ps
}

// ZeroState returns a zero hidden state (B, H).
func (g *GRU) ZeroState(B int) (*tensor.Tensor, error) {
	return tensor.Zeros(g.Wih.DType(), []int{B, g.Hidden}, g.Wih.Device())
}

// Step advances one time step on x (B, I) with hidden h (B, H).
func (g *GRU) Step(x, h *tensor.Tensor) (*tensor.Tensor, error) {
	H := g.Hidden
	xi, err := linearT(x, g.Wih, g.Bih) // (B, 3H)
	if err != nil {
		return nil, err
	}
	hh, err := linearT(h, g.Whh, g.Bhh) // (B, 3H)
	if err != nil {
		return nil, err
	}
	part := func(t *tensor.Tensor, k int) (*tensor.Tensor, error) { return autograd.Narrow(t, 1, k*H, H) }
	xr, err := part(xi, 0)
	if err != nil {
		return nil, err
	}
	hr, err := part(hh, 0)
	if err != nil {
		return nil, err
	}
	r, err := autograd.Add(xr, hr)
	if err != nil {
		return nil, err
	}
	if r, err = autograd.Sigmoid(r); err != nil {
		return nil, err
	}
	xz, err := part(xi, 1)
	if err != nil {
		return nil, err
	}
	hz, err := part(hh, 1)
	if err != nil {
		return nil, err
	}
	z, err := autograd.Add(xz, hz)
	if err != nil {
		return nil, err
	}
	if z, err = autograd.Sigmoid(z); err != nil {
		return nil, err
	}
	xn, err := part(xi, 2)
	if err != nil {
		return nil, err
	}
	hn, err := part(hh, 2)
	if err != nil {
		return nil, err
	}
	rhn, err := autograd.Mul(r, hn)
	if err != nil {
		return nil, err
	}
	n, err := autograd.Add(xn, rhn)
	if err != nil {
		return nil, err
	}
	if n, err = autograd.Tanh(n); err != nil {
		return nil, err
	}
	// h' = n + z·(h - n)
	diff, err := autograd.Sub(h, n)
	if err != nil {
		return nil, err
	}
	zd, err := autograd.Mul(z, diff)
	if err != nil {
		return nil, err
	}
	return autograd.Add(n, zd)
}

// Forward runs x (B, T, I) from a zero state and returns (B, T, H) plus the
// final hidden state.
func (g *GRU) Forward(x *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	if x.Rank() != 3 {
		return nil, nil, fmt.Errorf("GRU.Forward: want (B, T, I), got %v", x.Shape())
	}
	h, err := g.ZeroState(x.Dim(0))
	if err != nil {
		return nil, nil, err
	}
	outs := make([]*tensor.Tensor, x.Dim(1))
	for t := range outs {
		xt, err := stepInput(x, t)
		if err != nil {
			return nil, nil, err
		}
		if h, err = g.Step(xt, h); err != nil {
			return nil, nil, err
		}
		if outs[t], err = autograd.Reshape(h, x.Dim(0), 1, g.Hidden); err != nil {
			return nil, nil, err
		}
	}
	seq, err := autograd.Concat(outs, 1)
	if err != nil {
		return nil, nil, err
	}
	return seq, h, nil
}

// stepInput extracts x[:, t, :] as (B, I) differentiably.
func stepInput(x *tensor.Tensor, t int) (*tensor.Tensor, error) {
	xt, err := autograd.Narrow(x, 1, t, 1)
	if err != nil {
		return nil, err
	}
	return autograd.Reshape(xt, x.Dim(0), x.Dim(2))
}

// linearT computes x·Wᵀ + b for W stored (out, in).
func linearT(x, w, b *tensor.Tensor) (*tensor.Tensor, error) {
	wt, err := autograd.Transpose(w, 0, 1)
	if err != nil {
		return nil, err
	}
	if wt, err = autograd.Contiguous(wt); err != nil {
		return nil, err
	}
	y, err := autograd.MatMul(x, wt)
	if err != nil {
		return nil, err
	}
	if b == nil {
		return y, nil
	}
	return autograd.Add(y, b)
}

// affineGates computes x·Wihᵀ + bih + h·Whhᵀ + bhh.
func affineGates(x, h, wih, whh, bih, bhh *tensor.Tensor) (*tensor.Tensor, error) {
	xi, err := linearT(x, wih, bih)
	if err != nil {
		return nil, err
	}
	hh, err := linearT(h, whh, bhh)
	if err != nil {
		return nil, err
	}
	return autograd.Add(xi, hh)
}
