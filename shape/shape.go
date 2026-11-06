package shape

import (
	"fmt"
	"strings"
)

// Shape describes the dimension sizes of a tensor. It is an immutable value
// type: operations (Reshape, Permute, etc.) return new Shape values. Dims
// stores sizes in "row-major" order — axis 0 is outermost.
//
// The zero-value Shape is a scalar (rank 0, 1 element).
type Shape struct {
	dims []int
}

// New constructs a Shape from a slice of dimension sizes. The input is
// defensively copied. Negative sizes are rejected.
func New(dims ...int) Shape {
	out := make([]int, len(dims))
	for i, d := range dims {
		if d < 0 {
			panic(fmt.Sprintf("shape: negative dim %d at axis %d", d, i))
		}
		out[i] = d
	}
	return Shape{dims: out}
}

// Rank returns the number of dimensions.
func (s Shape) Rank() int { return len(s.dims) }

// Dim returns the size of axis i. Negative axes count from the end
// (Dim(-1) == last axis). Panics on out-of-range.
func (s Shape) Dim(i int) int {
	if i < 0 {
		i += len(s.dims)
	}
	if i < 0 || i >= len(s.dims) {
		panic(fmt.Sprintf("shape: axis %d out of range for rank %d", i, len(s.dims)))
	}
	return s.dims[i]
}

// Dims returns a copy of the dimension sizes.
func (s Shape) Dims() []int {
	out := make([]int, len(s.dims))
	copy(out, s.dims)
	return out
}

// NumElements returns the total element count (product of all dims). A rank-0
// shape has 1 element.
func (s Shape) NumElements() int {
	n := 1
	for _, d := range s.dims {
		n *= d
	}
	return n
}

// Equal reports whether two shapes have the same rank and same dim sizes.
func (s Shape) Equal(other Shape) bool {
	if len(s.dims) != len(other.dims) {
		return false
	}
	for i, d := range s.dims {
		if d != other.dims[i] {
			return false
		}
	}
	return true
}

// Clone returns a deep copy.
func (s Shape) Clone() Shape {
	return Shape{dims: s.Dims()}
}

// String returns a human-readable form like "(2, 3, 4)" or "()" for scalar.
func (s Shape) String() string {
	var b strings.Builder
	b.WriteByte('(')
	for i, d := range s.dims {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%d", d)
	}
	b.WriteByte(')')
	return b.String()
}

// IsScalar reports whether this shape describes a scalar (rank 0).
func (s Shape) IsScalar() bool { return len(s.dims) == 0 }

// NormalizeAxis resolves a possibly-negative axis against this shape's rank.
// Returns the positive axis and an error on out-of-range.
func (s Shape) NormalizeAxis(i int) (int, error) {
	if i < 0 {
		i += len(s.dims)
	}
	if i < 0 || i >= len(s.dims) {
		return 0, fmt.Errorf("shape: axis %d out of range for rank %d", i, len(s.dims))
	}
	return i, nil
}
