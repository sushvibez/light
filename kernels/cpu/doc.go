// Package cpu holds pure-Go CPU kernel implementations for every tensor op.
// Kernels are generic over dtype.Numeric where possible; F16 and BF16 kernels
// upcast to F32, compute, then downcast (software emulation matches Candle).
//
// Each kernel registers itself with internal/registry in package init so that
// ops.<Op>() calls find it without import cycles.
package cpu
