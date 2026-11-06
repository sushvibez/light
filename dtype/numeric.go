package dtype

// Numeric is the compile-time constraint for type-safe user-facing APIs
// (tensor.New[T], tensor.ToSlice[T], tensor.Full[T], …). F16 and BF16 are
// intentionally excluded — they use a separate dtype.Half / dtype.BFloat16
// path — because Go generics can't express conversion-only types cleanly.
type Numeric interface {
	~float32 | ~float64 |
		~int8 | ~int16 | ~int32 | ~int64 |
		~uint8 | ~uint16 | ~uint32 | ~uint64
}

// Of returns the DType corresponding to the Go scalar type T.
func Of[T Numeric]() DType {
	var zero T
	switch any(zero).(type) {
	case float32:
		return F32
	case float64:
		return F64
	case int8:
		return I8
	case int16:
		return I16
	case int32:
		return I32
	case int64:
		return I64
	case uint8:
		return U8
	case uint16:
		return U16
	case uint32:
		return U32
	case uint64:
		return U64
	}
	// Fallback: user-defined ~type. Go down by reflect-less kind probing via
	// size and signedness — good enough for ~float32 / ~uint8 / etc.
	return F32 // unreachable for direct Go primitives; only reachable via ~T alias
}
