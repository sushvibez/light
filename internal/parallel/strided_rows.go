package parallel

// Coalesce merges adjacent axes of a row-major walk over dims whenever
// both operands step through them as one (stride[i] == stride[i+1] ·
// dims[i+1] for a and b; broadcast zero strides merge with zero strides),
// and drops size-1 axes. The returned slices are new; the element order
// is unchanged.
func Coalesce(dims, sa, sb []int) (d, a, b []int) {
	for i := range dims {
		if dims[i] == 1 {
			continue
		}
		if n := len(d); n > 0 && a[n-1] == sa[i]*dims[i] && b[n-1] == sb[i]*dims[i] {
			d[n-1] *= dims[i]
			a[n-1], b[n-1] = sa[i], sb[i]
			continue
		}
		d, a, b = append(d, dims[i]), append(a, sa[i]), append(b, sb[i])
	}
	if len(d) == 0 {
		return []int{1}, []int{0}, []int{0}
	}
	return d, a, b
}

// StridedRuns visits the row-major elements [lo, hi) of an array of shape
// dims (coalesced: see Coalesce) in runs along the innermost axis: run
// receives the first element index, the operands' offsets for it and the
// run length; within a run a and b advance by their last strides (sa[r-1],
// sb[r-1], possibly 0). Offsets are computed once per run.
func StridedRuns(dims, sa, sb []int, baseA, baseB, lo, hi int, run func(i, ao, bo, n int)) {
	rank := len(dims)
	inner := dims[rank-1]
	idx := make([]int, rank)
	ao, bo := baseA, baseB
	r := lo
	for ax := rank - 1; ax >= 0; ax-- {
		idx[ax] = r % dims[ax]
		r /= dims[ax]
		ao += idx[ax] * sa[ax]
		bo += idx[ax] * sb[ax]
	}
	for i := lo; i < hi; {
		n := min(inner-idx[rank-1], hi-i)
		run(i, ao, bo, n)
		i += n
		// Advance n elements along the innermost axis, carrying outward.
		idx[rank-1] += n
		ao += n * sa[rank-1]
		bo += n * sb[rank-1]
		for ax := rank - 1; ax > 0 && idx[ax] >= dims[ax]; ax-- {
			ao -= dims[ax] * sa[ax]
			bo -= dims[ax] * sb[ax]
			idx[ax] = 0
			idx[ax-1]++
			ao += sa[ax-1]
			bo += sb[ax-1]
		}
	}
}
