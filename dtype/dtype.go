package dtype

import "fmt"

// DType enumerates the element types a tensor can hold.
type DType uint8

// Scalar dtypes come first. Quantized dtypes follow so ranges can be tested
// via ordering (IsQuantized uses >= Q4_0).
const (
	F16 DType = iota
	BF16
	F32
	F64
	U8
	I8
	U16
	I16
	U32
	I32
	U64
	I64
	F8E4M3 // 1-byte float: 1 sign, 4 exponent, 3 mantissa (fn: no inf, NaN=0x7F)

	// Quantized dtypes (GGUF / llama.cpp semantics). Sizes are block-based.
	Q4_0
	Q4_1
	Q5_0
	Q5_1
	Q8_0
	Q8_1
	Q2_K
	Q3_K
	Q4_K
	Q5_K
	Q6_K
	Q8_K

	numDTypes // sentinel; must stay last
)

// names is indexed by DType value.
var names = [numDTypes]string{
	"F16", "BF16", "F32", "F64",
	"U8", "I8", "U16", "I16", "U32", "I32", "U64", "I64", "F8E4M3",
	"Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1",
	"Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K",
}

// blockInfo[dt] = [elements per block, bytes per block].
// Scalar dtypes have {1, bytesPerElem}. Quantized dtypes use the GGUF layout
// as implemented by ggml. K-quant block counts are always 256.
var blockInfo = [numDTypes][2]int{
	{1, 2}, {1, 2}, {1, 4}, {1, 8}, // F16, BF16, F32, F64
	{1, 1}, {1, 1}, {1, 2}, {1, 2}, // U8, I8, U16, I16
	{1, 4}, {1, 4}, {1, 8}, {1, 8}, // U32, I32, U64, I64
	{1, 1},                                                     // F8E4M3
	{32, 18}, {32, 20}, {32, 22}, {32, 24}, {32, 34}, {32, 36}, // Q4_0..Q8_1
	{256, 84}, {256, 110}, {256, 144}, {256, 176}, {256, 210}, {256, 292}, // Q2K..Q8K
}

// Size returns bytes per scalar element. For quantized dtypes, where storage
// is block-based and not integer bytes per element, Size returns 0 — callers
// should use ByteSize(n) for allocation sizing instead.
func (d DType) Size() int {
	if d >= numDTypes {
		return 0
	}
	info := blockInfo[d]
	if info[0] == 1 {
		return info[1]
	}
	return 0
}

// BlockElems returns how many logical elements pack into one storage block.
// Scalar dtypes return 1; quantized dtypes return their block size.
func (d DType) BlockElems() int {
	if d >= numDTypes {
		return 0
	}
	return blockInfo[d][0]
}

// BlockBytes returns how many storage bytes one block occupies.
func (d DType) BlockBytes() int {
	if d >= numDTypes {
		return 0
	}
	return blockInfo[d][1]
}

// ByteSize returns the number of bytes required to store n elements of this
// dtype, rounded up to a whole number of blocks for quantized dtypes.
func (d DType) ByteSize(n int) int {
	if d >= numDTypes || n < 0 {
		return 0
	}
	info := blockInfo[d]
	blocks := (n + info[0] - 1) / info[0]
	return blocks * info[1]
}

// String returns the canonical name of the dtype ("F32", "BF16", "Q4_0", …).
func (d DType) String() string {
	if d >= numDTypes {
		return fmt.Sprintf("DType(%d)", d)
	}
	return names[d]
}

// IsFloat reports whether the dtype is a floating-point scalar type.
func (d DType) IsFloat() bool { return d <= F64 || d == F8E4M3 }

// IsInt reports whether the dtype is an integer scalar type (signed or unsigned).
func (d DType) IsInt() bool { return d >= U8 && d <= I64 }

// IsSigned reports whether the dtype is a signed scalar integer or a float.
func (d DType) IsSigned() bool {
	switch d {
	case F16, BF16, F32, F64, I8, I16, I32, I64, F8E4M3:
		return true
	default:
		return false
	}
}

// IsQuantized reports whether the dtype is a block-based quantized type.
func (d DType) IsQuantized() bool { return d >= Q4_0 && d < numDTypes }

// Valid reports whether the dtype is a known value in the enum range.
func (d DType) Valid() bool { return d < numDTypes }
