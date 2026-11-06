package autograd

import (
	"fmt"

	"light/ops"
	"light/tensor"
)

// Backward runs reverse-mode autograd starting from `root`. The initial
// gradient w.r.t. `root` defaults to a tensor of ones matching root's shape
// if `gradOutput` is nil (matching the PyTorch convention where
// loss.backward() implies an outer grad of 1).
//
// After Backward returns, any leaf tensor (no GradFn) that has
// RequiresGrad=true will have its .Grad() populated with the accumulated
// gradient.
func Backward(root *tensor.Tensor, gradOutput *tensor.Tensor) error {
	if root == nil {
		return fmt.Errorf("autograd.Backward: root is nil")
	}
	g := gradOutput
	if g == nil {
		var err error
		g, err = tensor.Ones(root.DType(), root.Shape().Dims(), root.Device())
		if err != nil {
			return fmt.Errorf("autograd.Backward: alloc ones: %w", err)
		}
	}

	// Root-as-leaf: no gradFn means root was a leaf; accumulate directly.
	if root.GradFn() == nil {
		if root.RequiresGrad() {
			return accumulateGrad(root, g)
		}
		return nil
	}
	rootNode := root.GradFn().(*FunctionNode)

	order, err := topoOrder(rootNode)
	if err != nil {
		return err
	}

	// gradBuf[node] holds the accumulated gradient for each of node's outputs.
	gradBuf := make(map[*FunctionNode][]*tensor.Tensor)
	initNode := func(n *FunctionNode) {
		if _, ok := gradBuf[n]; !ok {
			gradBuf[n] = make([]*tensor.Tensor, n.nOutputs)
		}
	}
	initNode(rootNode)
	gradBuf[rootNode][root.OutputNr()] = g

	// Process nodes in reverse topological order (outputs before inputs).
	for i := len(order) - 1; i >= 0; i-- {
		node := order[i]
		outGrads := gradBuf[node]
		if outGrads == nil {
			continue
		}
		inputGrads, err := node.fn.Backward(node.ctx, outGrads)
		if err != nil {
			return fmt.Errorf("autograd %s: backward: %w", node.fn.Name(), err)
		}
		if len(inputGrads) != len(node.inputs) {
			return fmt.Errorf(
				"autograd %s: backward returned %d grads for %d inputs",
				node.fn.Name(), len(inputGrads), len(node.inputs))
		}
		for j, inGrad := range inputGrads {
			if inGrad == nil {
				continue
			}
			in := node.inputs[j]
			if !in.RequiresGrad() && in.GradFn() == nil {
				continue // input doesn't need a gradient
			}
			if in.GradFn() != nil {
				parent := in.GradFn().(*FunctionNode)
				initNode(parent)
				idx := in.OutputNr()
				if gradBuf[parent][idx] == nil {
					gradBuf[parent][idx] = inGrad
				} else {
					sum, err := ops.Add(gradBuf[parent][idx], inGrad)
					if err != nil {
						return fmt.Errorf(
							"autograd %s: accumulate: %w", node.fn.Name(), err)
					}
					gradBuf[parent][idx] = sum
				}
			} else if in.RequiresGrad() {
				if err := accumulateGrad(in, inGrad); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// topoOrder returns a topological ordering of the autograd DAG reachable
// from root. The order is such that parents (inputs) come before children
// (outputs); reverse-iterating this slice walks outputs → inputs.
func topoOrder(root *FunctionNode) ([]*FunctionNode, error) {
	visited := make(map[*FunctionNode]bool)
	var order []*FunctionNode
	var visit func(n *FunctionNode)
	visit = func(n *FunctionNode) {
		if visited[n] {
			return
		}
		visited[n] = true
		for _, in := range n.inputs {
			if gfn := in.GradFn(); gfn != nil {
				visit(gfn.(*FunctionNode))
			}
		}
		order = append(order, n)
	}
	visit(root)
	return order, nil
}

// accumulateGrad adds g to leaf.grad (creating leaf.grad if nil).
func accumulateGrad(leaf *tensor.Tensor, g *tensor.Tensor) error {
	cur := leaf.Grad()
	if cur == nil {
		leaf.SetGrad(g)
		return nil
	}
	sum, err := ops.Add(cur, g)
	if err != nil {
		return fmt.Errorf("autograd accumulate: %w", err)
	}
	leaf.SetGrad(sum)
	return nil
}
