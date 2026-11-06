// Package dtype defines the element-type enumeration used across Light tensors,
// the Numeric generic constraint for type-safe user-facing APIs, and software
// emulation of half-precision (F16) and bfloat16 (BF16) types for CPU.
//
// Every tensor carries a runtime DType enum for internal kernel dispatch, while
// public constructors and accessors (tensor.New, tensor.ToSlice, etc.) are
// generic over Numeric so compile-time type checking catches most misuse.
package dtype
