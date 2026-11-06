// Example 11 — train a tiny 2-layer MLP on XOR using autograd + manual SGD.
//
// This is the first example that learns: it demonstrates that the forward
// graph, backward pass, and parameter updates all wire together correctly.
// No optimizer package yet — the SGD step is hand-written for clarity.
package main

import (
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"runtime"
	"unsafe"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// dev is the autodetected device (CUDA when available, CPU otherwise).
// Helpers that allocate fresh tensors read this global so the call sites
// stay compact.
var dev device.Device

func main() {
	runtime.LockOSThread()
	dev = device.Auto()
	fmt.Printf("device: %v\n", dev)

	// XOR data: 4 samples, 2 features, 1 target.
	x, _ := tensor.New([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2}, dev)
	target, _ := tensor.New([]float32{0, 1, 1, 0}, []int{4, 1}, dev)

	// Two-layer MLP: 2 → 8 → 1 with Tanh hidden activation + sigmoid output.
	W1 := initParam(2, 8)
	b1 := initBias(8)
	W2 := initParam(8, 1)
	b2 := initBias(1)

	params := []*tensor.Tensor{W1, b1, W2, b2}
	for _, p := range params {
		p.SetRequiresGrad(true)
	}

	lr := float32(0.5)
	steps := 4000

	forward := func() (*tensor.Tensor, error) {
		z1, err := autograd.MatMul(x, W1)
		if err != nil {
			return nil, err
		}
		h1, err := autograd.Add(z1, b1)
		if err != nil {
			return nil, err
		}
		a1, err := autograd.Tanh(h1)
		if err != nil {
			return nil, err
		}
		z2, err := autograd.MatMul(a1, W2)
		if err != nil {
			return nil, err
		}
		y, err := autograd.Add(z2, b2)
		if err != nil {
			return nil, err
		}
		yhat, err := autograd.Sigmoid(y)
		if err != nil {
			return nil, err
		}
		diff, err := autograd.Sub(yhat, target)
		if err != nil {
			return nil, err
		}
		sq, err := autograd.Mul(diff, diff)
		if err != nil {
			return nil, err
		}
		return autograd.Mean(sq)
	}

	for step := range steps {
		// Zero grads.
		for _, p := range params {
			p.SetGrad(nil)
		}
		loss, err := forward()
		if err != nil {
			log.Fatal(err)
		}
		if err := autograd.Backward(loss, nil); err != nil {
			log.Fatal(err)
		}
		// SGD: p ← p - lr * grad.
		for _, p := range params {
			g := p.Grad()
			scaled, err := scaleTensor(g, lr)
			if err != nil {
				log.Fatal(err)
			}
			updated, err := ops.Sub(p, scaled)
			if err != nil {
				log.Fatal(err)
			}
			copyInto(p, updated)
		}
		if step%500 == 0 || step == steps-1 {
			lossCPU, _ := loss.To(device.CPU())
			fmt.Printf("step %4d  loss = %.5f\n", step, tensor.MustSlice[float32](lossCPU)[0])
		}
	}

	// Report final predictions.
	y, _ := evalForward(W1, b1, W2, b2, x)
	fmt.Println("\nInputs → predictions (trained MLP):")
	xCPU, _ := x.To(device.CPU())
	yCPU, _ := y.To(device.CPU())
	tCPU, _ := target.To(device.CPU())
	xSlice := tensor.MustSlice[float32](xCPU)
	ySlice := tensor.MustSlice[float32](yCPU)
	tSlice := tensor.MustSlice[float32](tCPU)
	for i := range 4 {
		fmt.Printf("  [%.0f %.0f] → %.4f  (target %.0f)\n",
			xSlice[i*2], xSlice[i*2+1], ySlice[i], tSlice[i])
	}
}

// initParam returns a small random [in, out] weight matrix scaled by 1/sqrt(in).
func initParam(in, out int) *tensor.Tensor {
	data := make([]float32, in*out)
	s := float32(1.0 / math.Sqrt(float64(in)))
	for i := range data {
		data[i] = (rand.Float32()*2 - 1) * s
	}
	t, _ := tensor.New(data, []int{in, out}, dev)
	return t
}

func initBias(n int) *tensor.Tensor {
	data := make([]float32, n)
	t, _ := tensor.New(data, []int{n}, dev)
	return t
}

// scaleTensor returns t * c as a new tensor. Uses broadcasting against a 1-elem tensor.
func scaleTensor(t *tensor.Tensor, c float32) (*tensor.Tensor, error) {
	s, err := tensor.Full(c, []int{1}, t.Device())
	if err != nil {
		return nil, err
	}
	return ops.Mul(t, s)
}

// copyInto overwrites dst's storage with src's contents (same shape, F32).
// Uses the device-aware tensor.CopyFrom so the same code path works on
// CPU and CUDA without an explicit branch.
func copyInto(dst, src *tensor.Tensor) {
	if err := dst.CopyFrom(src); err != nil {
		log.Fatal(err)
	}
	dst.SetGradFn(nil, 0)
	_ = unsafe.Pointer(nil)
	_ = dtype.F32
}

// evalForward runs the trained network without building a gradient graph.
func evalForward(W1, b1, W2, b2, x *tensor.Tensor) (*tensor.Tensor, error) {
	z1, err := ops.MatMul(x, W1)
	if err != nil {
		return nil, err
	}
	h1, err := ops.Add(z1, b1)
	if err != nil {
		return nil, err
	}
	a1, err := ops.Tanh(h1)
	if err != nil {
		return nil, err
	}
	z2, err := ops.MatMul(a1, W2)
	if err != nil {
		return nil, err
	}
	y, err := ops.Add(z2, b2)
	if err != nil {
		return nil, err
	}
	return ops.Sigmoid(y)
}
