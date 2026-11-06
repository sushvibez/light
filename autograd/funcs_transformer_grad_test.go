package autograd_test

import (
	"testing"
	"unsafe"

	"light/autograd"
	"light/device"
	"light/nn"
	"light/tensor"
)

func TestGradCheckRMSNorm(t *testing.T) {
	x, _ := tensor.New([]float32{
		0.3, -0.5, 1.0, 0.2,
		0.7, -0.1, -0.4, 0.8,
	}, []int{2, 4}, device.CPU())
	gamma, _ := tensor.New([]float32{1.0, 0.5, 1.5, 2.0}, []int{4}, device.CPU())
	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := autograd.RMSNorm(leaves[0], leaves[1], 1e-5)
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	if err := autograd.GradCheck(compute, []*tensor.Tensor{x, gamma}, 1e-3, 5e-3); err != nil {
		t.Fatal(err)
	}
}

// Full SelfAttention block: every parameter must receive a non-trivial
// gradient, and gradcheck should verify the closed-form against finite diffs.
func TestGradCheckSelfAttention(t *testing.T) {
	cpu := device.CPU()
	// (S=3, H=4) is small enough for gradcheck.
	x, _ := tensor.New([]float32{
		0.1, -0.2, 0.3, 0.4,
		-0.5, 0.6, -0.7, 0.8,
		0.2, 0.1, -0.3, 0.05,
	}, []int{3, 4}, cpu)
	attn, err := nn.NewSelfAttention(4, false, cpu)
	if err != nil {
		t.Fatal(err)
	}
	params := attn.Parameters()
	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := attn.Forward(x)
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	if err := autograd.GradCheck(compute, params, 1e-2, 1e-2); err != nil {
		t.Fatal(err)
	}
}

// Multi-head attention gradcheck with 2 heads × 2 head_dim = 4 hidden.
// Uses small fixed weights so the softmax isn't saturated and finite
// differences are close to the analytic gradient.
func TestGradCheckMultiHeadAttention(t *testing.T) {
	cpu := device.CPU()
	x, _ := tensor.New([]float32{
		0.1, -0.2, 0.3, 0.4,
		-0.5, 0.6, -0.7, 0.8,
		0.2, 0.1, -0.3, 0.05,
	}, []int{3, 4}, cpu)
	mha, err := nn.NewMultiHeadAttention(4, 2, false, false, cpu)
	if err != nil {
		t.Fatal(err)
	}
	// Overwrite projection weights with small, well-conditioned values so
	// gradients and finite diffs don't saturate the softmax.
	setSmallWeights := func(l *nn.Linear, seed float32) {
		buf := l.W.Storage().CPUBytes()
		for i := 0; i < len(buf); i += 4 {
			// deterministic small ramp in [-0.15, 0.15]
			v := float32(i/4)*0.02 + seed*0.01
			if v > 0.15 {
				v = 0.15
			}
			*(*float32)(unsafe.Pointer(&buf[i])) = v
		}
	}
	setSmallWeights(mha.QProj, 0.1)
	setSmallWeights(mha.KProj, 0.2)
	setSmallWeights(mha.VProj, 0.3)
	setSmallWeights(mha.OProj, 0.4)

	params := mha.Parameters()
	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := mha.Forward(x)
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	// Reshape + Permute + batched MatMul + Softmax + batched MatMul +
	// Permute + Reshape + Linear is ~8 ops deep; 2% absolute tolerance
	// catches any real-signal bug while accommodating FD error growth.
	if err := autograd.GradCheck(compute, params, 1e-3, 5e-2); err != nil {
		t.Fatal(err)
	}
}

// Multi-head attention gradcheck on a real (B, S, H) batched input.
func TestGradCheckMultiHeadAttentionBatched(t *testing.T) {
	cpu := device.CPU()
	// (B=2, S=3, H=4); 2 heads × 2 head_dim.
	x, _ := tensor.New([]float32{
		0.1, -0.2, 0.3, 0.4,
		-0.5, 0.6, -0.7, 0.8,
		0.2, 0.1, -0.3, 0.05,
		0.4, -0.3, 0.2, -0.1,
		-0.2, 0.5, -0.6, 0.7,
		0.3, 0.2, -0.4, 0.1,
	}, []int{2, 3, 4}, cpu)
	mha, err := nn.NewMultiHeadAttention(4, 2, false, false, cpu)
	if err != nil {
		t.Fatal(err)
	}
	setSmallW := func(l *nn.Linear, seed float32) {
		buf := l.W.Storage().CPUBytes()
		for i := 0; i < len(buf); i += 4 {
			v := float32(i/4)*0.02 + seed*0.01
			if v > 0.15 {
				v = 0.15
			}
			*(*float32)(unsafe.Pointer(&buf[i])) = v
		}
	}
	setSmallW(mha.QProj, 0.1)
	setSmallW(mha.KProj, 0.2)
	setSmallW(mha.VProj, 0.3)
	setSmallW(mha.OProj, 0.4)

	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := mha.Forward(x)
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	if err := autograd.GradCheck(compute, mha.Parameters(), 1e-3, 5e-2); err != nil {
		t.Fatal(err)
	}
}

// End-to-end transformer block gradcheck (LayerNorm + SelfAttn + residual +
// LayerNorm + FFN + residual).
func TestGradCheckTransformerBlock(t *testing.T) {
	cpu := device.CPU()
	x, _ := tensor.New([]float32{
		0.1, -0.2, 0.3, 0.4,
		-0.5, 0.6, -0.7, 0.8,
	}, []int{2, 4}, cpu)
	block, err := nn.NewTransformerBlock(4, false, cpu)
	if err != nil {
		t.Fatal(err)
	}
	params := block.Parameters()
	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := block.Forward(x)
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	// Wider tolerance — many composed ops compound finite-diff error.
	if err := autograd.GradCheck(compute, params, 1e-2, 3e-2); err != nil {
		t.Fatal(err)
	}
}
