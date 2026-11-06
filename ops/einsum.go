package ops

import (
	"fmt"
	"strings"

	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// einsumPlan is the parsed, validated form of an einsum notation string.
// Letters are mapped to dense ids in first-appearance order.
type einsumPlan struct {
	subs   [][]int // per operand: letter id at each axis
	out    []int   // letter ids in output order
	sum    []int   // contracted letter ids (first-appearance order)
	sizes  []int   // size per letter id
	inOut  []bool  // per letter id: appears in output
	repeat bool    // some operand repeats a letter (diagonal/trace)
}

func parseEinsum(notation string, operands []*tensor.Tensor) (*einsumPlan, error) {
	lhs, rhs, ok := strings.Cut(notation, "->")
	if !ok {
		return nil, fmt.Errorf("Einsum: notation %q must contain '->'", notation)
	}
	lhs = strings.ReplaceAll(lhs, " ", "")
	rhs = strings.ReplaceAll(rhs, " ", "")
	opSubs := strings.Split(lhs, ",")
	if len(opSubs) != len(operands) {
		return nil, fmt.Errorf("Einsum: notation %q names %d operands but %d were given", notation, len(opSubs), len(operands))
	}
	p := &einsumPlan{}
	id := map[byte]int{}
	letter := func(c byte) (int, error) {
		if c < 'a' || c > 'z' {
			if c < 'A' || c > 'Z' {
				return 0, fmt.Errorf("Einsum: invalid index character %q in %q", c, notation)
			}
		}
		if l, ok := id[c]; ok {
			return l, nil
		}
		l := len(p.sizes)
		id[c] = l
		p.sizes = append(p.sizes, -1)
		return l, nil
	}
	for i, sub := range opSubs {
		dims := operands[i].Shape().Dims()
		if len(sub) != len(dims) {
			return nil, fmt.Errorf("Einsum: operand %d has rank %d but subscript %q has %d indices", i, len(dims), sub, len(sub))
		}
		seen := map[int]bool{}
		axes := make([]int, len(sub))
		for ax := range len(sub) {
			l, err := letter(sub[ax])
			if err != nil {
				return nil, err
			}
			if p.sizes[l] == -1 {
				p.sizes[l] = dims[ax]
			} else if p.sizes[l] != dims[ax] {
				return nil, fmt.Errorf("Einsum: index %q has inconsistent sizes %d vs %d", sub[ax], p.sizes[l], dims[ax])
			}
			if seen[l] {
				p.repeat = true
			}
			seen[l] = true
			axes[ax] = l
		}
		p.subs = append(p.subs, axes)
	}
	p.inOut = make([]bool, len(p.sizes))
	outSeen := map[int]bool{}
	for i := range len(rhs) {
		c := rhs[i]
		l, ok := id[c]
		if !ok {
			return nil, fmt.Errorf("Einsum: output index %q does not appear in any operand", c)
		}
		if outSeen[l] {
			return nil, fmt.Errorf("Einsum: output index %q repeated", c)
		}
		outSeen[l] = true
		p.inOut[l] = true
		p.out = append(p.out, l)
	}
	for l := range p.sizes {
		if !p.inOut[l] {
			p.sum = append(p.sum, l)
		}
	}
	return p, nil
}

func (p *einsumPlan) outDims() []int {
	dims := make([]int, len(p.out))
	for i, l := range p.out {
		dims[i] = p.sizes[l]
	}
	return dims
}

// Einsum evaluates an Einstein-summation expression over F32/F64 operands.
// The notation must spell the output explicitly ("ij,jk->ik"); letters
// a-z/A-Z name axes, a letter missing from the output is summed over, and a
// letter repeated inside one operand selects the diagonal.
//
// Supported forms include matrix / batched products, transposes, outer
// products, element-wise products, traces and diagonals, and reductions.
//
// Execution:
//   - CPU: a single general kernel (any operand count, repeated letters).
//   - CUDA: one- and two-operand expressions without repeated letters are
//     lowered to Permute/Contiguous + Sum + batched MatMul, all of which
//     have CUDA kernels. Anything else (3+ operands, diagonals) is computed
//     on the host and the result copied back — a deliberate, documented
//     fallback rather than a device kernel.
func Einsum(notation string, operands ...*tensor.Tensor) (*tensor.Tensor, error) {
	if len(operands) == 0 {
		return nil, fmt.Errorf("Einsum: at least one operand required")
	}
	dt := operands[0].DType()
	dev := operands[0].Device()
	if dt != dtype.F32 && dt != dtype.F64 {
		return nil, fmt.Errorf("Einsum: dtype %v not supported (F32/F64 only)", dt)
	}
	for i, t := range operands[1:] {
		if t.DType() != dt {
			return nil, fmt.Errorf("Einsum: operand %d dtype %v != %v", i+1, t.DType(), dt)
		}
		if !t.Device().SameAs(dev) {
			return nil, fmt.Errorf("Einsum: operand %d on %v, expected %v", i+1, t.Device(), dev)
		}
	}
	plan, err := parseEinsum(notation, operands)
	if err != nil {
		return nil, err
	}
	contig := make([]*tensor.Tensor, len(operands))
	for i, t := range operands {
		if contig[i], err = ensureContig(t); err != nil {
			return nil, err
		}
	}
	if device.IsCUDA(dev) {
		if dt == dtype.F32 {
			return einsumCUDA(plan, contig)
		}
		// Host fallback: the batched CUDA matmul is F32-only.
		hosts := make([]*tensor.Tensor, len(contig))
		for i, t := range contig {
			if hosts[i], err = t.To(device.CPU()); err != nil {
				return nil, err
			}
		}
		out, err := einsumCPU(plan, hosts)
		if err != nil {
			return nil, err
		}
		return out.To(dev)
	}
	return einsumCPU(plan, contig)
}

func einsumCPU(plan *einsumPlan, operands []*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := tensor.Zeros(operands[0].DType(), plan.outDims(), operands[0].Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "einsum", Dev: device.KindCPU, DType: operands[0].DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: operands,
		Output: out,
		Attrs: map[string]any{
			"subs":  plan.subs,
			"out":   plan.out,
			"sum":   plan.sum,
			"sizes": plan.sizes,
		},
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// einsumPreSum sums away every letter of operand `op` that appears neither
// in the output nor in any other operand. Returns the reduced tensor and its
// remaining letters (in axis order).
func einsumPreSum(plan *einsumPlan, op int, t *tensor.Tensor, other []int) (*tensor.Tensor, []int, error) {
	inOther := map[int]bool{}
	for _, l := range other {
		inOther[l] = true
	}
	var axes []int
	var keep []int
	for ax, l := range plan.subs[op] {
		if !plan.inOut[l] && !inOther[l] {
			axes = append(axes, ax)
		} else {
			keep = append(keep, l)
		}
	}
	if len(axes) == 0 {
		return t, append([]int(nil), plan.subs[op]...), nil
	}
	if len(axes) == t.Rank() {
		// Full reduction to a scalar; keep a rank-0 tensor.
		s, err := Sum(t, ReduceOptions{})
		return s, nil, err
	}
	s, err := Sum(t, ReduceOptions{Axes: axes})
	return s, keep, err
}

// permuteTo reorders t (whose axes carry letters `have`) so the axes come
// out in `want` order, then materialises the result.
func permuteTo(t *tensor.Tensor, have, want []int) (*tensor.Tensor, error) {
	if len(have) != len(want) {
		return nil, fmt.Errorf("Einsum: internal axis mismatch %v vs %v", have, want)
	}
	perm := make([]int, len(want))
	identity := true
	for i, l := range want {
		pos := -1
		for j, h := range have {
			if h == l {
				pos = j
				break
			}
		}
		if pos < 0 {
			return nil, fmt.Errorf("Einsum: internal missing axis %d", l)
		}
		perm[i] = pos
		if pos != i {
			identity = false
		}
	}
	if identity {
		return ensureContig(t)
	}
	p, err := t.Permute(perm...)
	if err != nil {
		return nil, err
	}
	return p.Contiguous()
}

func prodSizes(plan *einsumPlan, letters []int) int {
	n := 1
	for _, l := range letters {
		n *= plan.sizes[l]
	}
	return n
}

// einsumCUDA runs any F32 expression on the device: repeated letters are
// first materialised as diagonal views (AsStrided + Contiguous), then
// three or more operands are contracted pairwise left to right, each pair
// through the batched-matmul lowering below. Letters that no later operand
// or the output needs are summed out at the pair that last uses them.
func einsumCUDA(plan *einsumPlan, operands []*tensor.Tensor) (*tensor.Tensor, error) {
	subs := make([][]int, len(plan.subs))
	for i, s := range plan.subs {
		subs[i] = append([]int(nil), s...)
	}
	ops := append([]*tensor.Tensor(nil), operands...)
	if plan.repeat {
		for i, t := range ops {
			d, letters, err := einsumDiagonal(t, subs[i])
			if err != nil {
				return nil, err
			}
			ops[i], subs[i] = d, letters
		}
	}
	if len(ops) <= 2 {
		p := *plan
		p.subs, p.repeat = subs, false
		return einsumCUDALowered(&p, ops)
	}
	// Pairwise: contract ops[0], ops[1] → intermediate keeping every letter
	// that the output or a later operand still needs.
	needed := map[int]bool{}
	for _, l := range plan.out {
		needed[l] = true
	}
	for _, s := range subs[2:] {
		for _, l := range s {
			needed[l] = true
		}
	}
	var inter []int
	seen := map[int]bool{}
	for _, s := range subs[:2] {
		for _, l := range s {
			if needed[l] && !seen[l] {
				inter = append(inter, l)
				seen[l] = true
			}
		}
	}
	notation := einsumLetters(subs[0]) + "," + einsumLetters(subs[1]) + "->" + einsumLetters(inter)
	c, err := Einsum(notation, ops[0], ops[1])
	if err != nil {
		return nil, err
	}
	rest := make([]string, 0, len(subs)-1)
	rest = append(rest, einsumLetters(inter))
	for _, s := range subs[2:] {
		rest = append(rest, einsumLetters(s))
	}
	next := append([]*tensor.Tensor{c}, ops[2:]...)
	return Einsum(strings.Join(rest, ",")+"->"+einsumLetters(plan.out), next...)
}

const einsumAlphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

// einsumLetters renders letter ids as a subscript string for a sub-call.
func einsumLetters(ids []int) string {
	b := make([]byte, len(ids))
	for i, id := range ids {
		b[i] = einsumAlphabet[id%len(einsumAlphabet)]
	}
	return string(b)
}

// einsumDiagonal collapses repeated letters of one operand into a single
// axis whose stride is the sum of the repeated axes' strides, and
// materialises the view. Returns the new tensor and its letter list.
func einsumDiagonal(t *tensor.Tensor, letters []int) (*tensor.Tensor, []int, error) {
	first := map[int]int{}
	repeated := false
	for ax, l := range letters {
		if _, ok := first[l]; ok {
			repeated = true
		} else {
			first[l] = ax
		}
	}
	if !repeated {
		return t, letters, nil
	}
	st := t.Strides().Slice()
	var dims, strides, out []int
	for ax, l := range letters {
		if first[l] != ax {
			continue
		}
		s := 0
		for ax2, l2 := range letters {
			if l2 == l {
				s += st[ax2]
			}
		}
		dims = append(dims, t.Dim(ax))
		strides = append(strides, s)
		out = append(out, l)
	}
	v, err := t.AsStrided(dims, strides, t.Offset())
	if err != nil {
		return nil, nil, fmt.Errorf("Einsum: diagonal view: %w", err)
	}
	c, err := v.Contiguous()
	if err != nil {
		return nil, nil, err
	}
	return c, out, nil
}

// einsumCUDALowered handles 1- and 2-operand expressions without repeated
// letters by composing existing device ops.
func einsumCUDALowered(plan *einsumPlan, operands []*tensor.Tensor) (*tensor.Tensor, error) {
	if len(operands) == 1 {
		t, have, err := einsumPreSum(plan, 0, operands[0], nil)
		if err != nil {
			return nil, err
		}
		return permuteTo(t, have, plan.out)
	}
	a, haveA, err := einsumPreSum(plan, 0, operands[0], plan.subs[1])
	if err != nil {
		return nil, err
	}
	b, haveB, err := einsumPreSum(plan, 1, operands[1], plan.subs[0])
	if err != nil {
		return nil, err
	}
	inA := map[int]bool{}
	for _, l := range haveA {
		inA[l] = true
	}
	inB := map[int]bool{}
	for _, l := range haveB {
		inB[l] = true
	}
	var batch, mIdx, nIdx, kIdx []int
	for _, l := range haveA {
		switch {
		case inB[l] && plan.inOut[l]:
			batch = append(batch, l)
		case inB[l]:
			kIdx = append(kIdx, l)
		default:
			mIdx = append(mIdx, l)
		}
	}
	for _, l := range haveB {
		if !inA[l] {
			nIdx = append(nIdx, l)
		}
	}
	// A → (batch, M, K), B → (batch, K, N).
	aOrder := append(append(append([]int{}, batch...), mIdx...), kIdx...)
	bOrder := append(append(append([]int{}, batch...), kIdx...), nIdx...)
	aP, err := permuteTo(a, haveA, aOrder)
	if err != nil {
		return nil, err
	}
	bP, err := permuteTo(b, haveB, bOrder)
	if err != nil {
		return nil, err
	}
	Bsz, Msz, Ksz, Nsz := prodSizes(plan, batch), prodSizes(plan, mIdx), prodSizes(plan, kIdx), prodSizes(plan, nIdx)
	a3, err := aP.Reshape(Bsz, Msz, Ksz)
	if err != nil {
		return nil, err
	}
	b3, err := bP.Reshape(Bsz, Ksz, Nsz)
	if err != nil {
		return nil, err
	}
	c3, err := MatMul(a3, b3) // (Bsz, Msz, Nsz)
	if err != nil {
		return nil, err
	}
	cOrder := append(append(append([]int{}, batch...), mIdx...), nIdx...)
	cDims := make([]int, 0, len(cOrder))
	for _, l := range cOrder {
		cDims = append(cDims, plan.sizes[l])
	}
	c, err := c3.Reshape(cDims...)
	if err != nil {
		return nil, err
	}
	return permuteTo(c, cOrder, plan.out)
}
