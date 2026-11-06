// Package tensor defines the core *Tensor type: a view into a storage.Storage
// with a Shape, Strides, and an offset, optionally annotated for autograd.
//
// Constructors (New, Zeros, Ones, Full, Arange, Rand, RandN) are generic over
// dtype.Numeric for compile-time type safety. Shape operations (Reshape,
// Permute, Transpose, Contiguous, Broadcast, Narrow, Slice) return new Tensors
// that share underlying storage where possible.
package tensor
