package autograd_test

import (
	"testing"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/tensor"
)

func TestGradCheckPermute(t *testing.T) {
	x, _ := tensor.New([]float32{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
	}, []int{2, 3, 2}, device.CPU())
	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := autograd.Permute(leaves[0], 2, 0, 1)
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	if err := autograd.GradCheck(compute, []*tensor.Tensor{x}, 1e-3, 1e-3); err != nil {
		t.Fatal(err)
	}
}

func TestGradCheckBatchedMatMul(t *testing.T) {
	// 3-D batched matmul: A [2,3,4] @ B [2,4,2] -> C [2,3,2].
	a, _ := tensor.New([]float32{
		0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
		-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2,
	}, []int{2, 3, 4}, device.CPU())
	b, _ := tensor.New([]float32{
		0.3, 0.1, 0.2, -0.3, 0.5, 0.1, -0.2, 0.4,
		-0.3, 0.2, 0.1, -0.4, 0.5, -0.1, 0.3, 0.2,
	}, []int{2, 4, 2}, device.CPU())
	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		c, err := autograd.MatMul(leaves[0], leaves[1])
		if err != nil {
			return nil, err
		}
		return autograd.Sum(c)
	}
	if err := autograd.GradCheck(compute, []*tensor.Tensor{a, b}, 1e-3, 5e-3); err != nil {
		t.Fatal(err)
	}
}

func TestGradCheckIndexSelect(t *testing.T) {
	src, _ := tensor.New([]float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	}, []int{4, 3}, device.CPU())
	indices, _ := tensor.New([]int64{0, 2, 1, 2, 0}, []int{5}, device.CPU())

	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := autograd.IndexSelect(leaves[0], indices)
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	// Gradcheck only on src (indices are not differentiable).
	if err := autograd.GradCheck(compute, []*tensor.Tensor{src}, 1e-3, 5e-3); err != nil {
		t.Fatal(err)
	}
}

func TestGradCheckLayerNorm(t *testing.T) {
	x, _ := tensor.New([]float32{
		0.2, -0.5, 1.3, 0.1,
		0.7, -0.2, -0.3, 0.4,
	}, []int{2, 4}, device.CPU())
	gamma, _ := tensor.New([]float32{0.5, 1.0, 1.5, 2.0}, []int{4}, device.CPU())
	beta, _ := tensor.New([]float32{0.1, -0.1, 0.2, -0.2}, []int{4}, device.CPU())
	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := autograd.LayerNorm(leaves[0], leaves[1], leaves[2], 1e-5)
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	if err := autograd.GradCheck(compute, []*tensor.Tensor{x, gamma, beta}, 1e-3, 5e-3); err != nil {
		t.Fatal(err)
	}
}

// IndexSelect + CrossEntropyLoss — the common "token → classification" pipeline.
// Training signal flows through the embedding table.
func TestGradCheckAxisSum(t *testing.T) {
	x, _ := tensor.New([]float32{
		0.1, 0.2, 0.3, 0.4,
		0.5, 0.6, 0.7, 0.8,
		-0.1, -0.2, -0.3, -0.4,
	}, []int{3, 4}, device.CPU())
	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := autograd.AxisSum(leaves[0], []int{1}, false)
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	if err := autograd.GradCheck(compute, []*tensor.Tensor{x}, 1e-3, 1e-3); err != nil {
		t.Fatal(err)
	}
}

func TestGradCheckAxisMean(t *testing.T) {
	x, _ := tensor.New([]float32{
		0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
		0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
	}, []int{2, 3, 2}, device.CPU())
	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := autograd.AxisMean(leaves[0], []int{1}, false)
		if err != nil {
			return nil, err
		}
		return autograd.Sum(y)
	}
	if err := autograd.GradCheck(compute, []*tensor.Tensor{x}, 1e-3, 1e-3); err != nil {
		t.Fatal(err)
	}
}

func TestEmbeddingTraining(t *testing.T) {
	cpu := device.CPU()
	// Vocab=4, embedding dim = 3, classes = 2.
	W, _ := tensor.Rand(dtype.F32, []int{4, 3}, cpu)
	W.SetRequiresGrad(true)

	classifier, _ := tensor.New([]float32{
		0.3, -0.2,
		0.4, 0.1,
		-0.5, 0.6,
	}, []int{3, 2}, cpu)
	classifier.SetRequiresGrad(true)

	tokens, _ := tensor.New([]int64{0, 1, 2, 3}, []int{4}, cpu)
	target, _ := tensor.New([]int64{1, 0, 1, 0}, []int{4}, cpu)

	compute := func(leaves []*tensor.Tensor) (*tensor.Tensor, error) {
		emb, err := autograd.IndexSelect(leaves[0], tokens)
		if err != nil {
			return nil, err
		}
		logits, err := autograd.MatMul(emb, leaves[1])
		if err != nil {
			return nil, err
		}
		return autograd.CrossEntropyLoss(logits, target)
	}
	if err := autograd.GradCheck(compute, []*tensor.Tensor{W, classifier}, 1e-3, 5e-3); err != nil {
		t.Fatal(err)
	}
}
