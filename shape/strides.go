package shape

// Strides represents per-axis step sizes (in elements, not bytes). Strides
// along with an offset and a Shape fully describe how logical indices map
// to storage indices.
type Strides struct {
	strides []int
}

// NewStrides constructs Strides from an explicit slice (defensively copied).
func NewStrides(s ...int) Strides {
	out := make([]int, len(s))
	copy(out, s)
	return Strides{strides: out}
}

// DefaultStrides returns the row-major (C-order) strides for a Shape. Axis 0
// varies slowest, last axis varies fastest with stride 1.
//
//	DefaultStrides([2,3,4]) → [12, 4, 1]
func DefaultStrides(s Shape) Strides {
	r := s.Rank()
	if r == 0 {
		return Strides{strides: nil}
	}
	out := make([]int, r)
	out[r-1] = 1
	for i := r - 2; i >= 0; i-- {
		out[i] = out[i+1] * s.Dim(i+1)
	}
	return Strides{strides: out}
}

// Rank returns the number of stride entries.
func (s Strides) Rank() int { return len(s.strides) }

// Get returns the stride for axis i (supports negative indexing).
func (s Strides) Get(i int) int {
	if i < 0 {
		i += len(s.strides)
	}
	return s.strides[i]
}

// Slice returns a copy of the strides as a []int.
func (s Strides) Slice() []int {
	out := make([]int, len(s.strides))
	copy(out, s.strides)
	return out
}

// Equal reports whether two Strides have identical values.
func (s Strides) Equal(other Strides) bool {
	if len(s.strides) != len(other.strides) {
		return false
	}
	for i, v := range s.strides {
		if v != other.strides[i] {
			return false
		}
	}
	return true
}

// IsContiguous reports whether the given Strides describe a row-major
// contiguous layout for the given Shape. Empty / rank-0 shapes are contiguous.
func IsContiguous(s Shape, st Strides) bool {
	return st.Equal(DefaultStrides(s))
}

// ContigWithDim0Broadcast reports contiguity ignoring axis 0 if its stride is
// 0 (common for expanded batch dimensions). Useful for elementwise kernels
// that can linearize over the last N axes.
func ContigWithDim0Broadcast(s Shape, st Strides) bool {
	if s.Rank() == 0 {
		return true
	}
	ds := DefaultStrides(s)
	// Check strides[1:] match default; strides[0] may be 0 or match.
	for i := 1; i < len(st.strides); i++ {
		if st.strides[i] != ds.strides[i] {
			return false
		}
	}
	return st.strides[0] == ds.strides[0] || st.strides[0] == 0
}

// Permute returns new Strides where axis i comes from source axis perm[i].
func (s Strides) Permute(perm []int) Strides {
	out := make([]int, len(perm))
	for i, p := range perm {
		out[i] = s.strides[p]
	}
	return Strides{strides: out}
}
