package gguf

import (
	"fmt"

	"light/dtype"
)

// magic is the "GGUF" file magic in little-endian uint32 form.
const magic uint32 = 0x46554747

// SupportedVersions lists the GGUF versions this reader handles.
const (
	MinVersion = 2
	MaxVersion = 3
)

// Value types per the GGUF spec. Used both to decode metadata KVs and inside
// GGUF_TYPE_ARRAY entries.
const (
	TypeUInt8   uint32 = 0
	TypeInt8    uint32 = 1
	TypeUInt16  uint32 = 2
	TypeInt16   uint32 = 3
	TypeUInt32  uint32 = 4
	TypeInt32   uint32 = 5
	TypeFloat32 uint32 = 6
	TypeBool    uint32 = 7
	TypeString  uint32 = 8
	TypeArray   uint32 = 9
	TypeUInt64  uint32 = 10
	TypeInt64   uint32 = 11
	TypeFloat64 uint32 = 12
)

// ggmlType is the per-tensor dtype enum from ggml. Only the subset we support
// is listed here.
const (
	ggmlF32  uint32 = 0
	ggmlF16  uint32 = 1
	ggmlQ4_0 uint32 = 2
	ggmlQ4_1 uint32 = 3
	ggmlQ5_0 uint32 = 6
	ggmlQ5_1 uint32 = 7
	ggmlQ8_0 uint32 = 8
	ggmlQ8_1 uint32 = 9
	ggmlQ2_K uint32 = 10
	ggmlQ3_K uint32 = 11
	ggmlQ4_K uint32 = 12
	ggmlQ5_K uint32 = 13
	ggmlQ6_K uint32 = 14
	ggmlQ8_K uint32 = 15
	ggmlI8   uint32 = 24
	ggmlI16  uint32 = 25
	ggmlI32  uint32 = 26
	ggmlI64  uint32 = 27
	ggmlF64  uint32 = 28
	ggmlBF16 uint32 = 30
)

// ggmlToDType maps a ggml type code to the equivalent light/dtype.DType.
func ggmlToDType(t uint32) (dtype.DType, error) {
	switch t {
	case ggmlF32:
		return dtype.F32, nil
	case ggmlF16:
		return dtype.F16, nil
	case ggmlBF16:
		return dtype.BF16, nil
	case ggmlF64:
		return dtype.F64, nil
	case ggmlI8:
		return dtype.I8, nil
	case ggmlI16:
		return dtype.I16, nil
	case ggmlI32:
		return dtype.I32, nil
	case ggmlI64:
		return dtype.I64, nil
	case ggmlQ4_0:
		return dtype.Q4_0, nil
	case ggmlQ4_1:
		return dtype.Q4_1, nil
	case ggmlQ5_0:
		return dtype.Q5_0, nil
	case ggmlQ5_1:
		return dtype.Q5_1, nil
	case ggmlQ8_0:
		return dtype.Q8_0, nil
	case ggmlQ8_1:
		return dtype.Q8_1, nil
	case ggmlQ2_K:
		return dtype.Q2_K, nil
	case ggmlQ3_K:
		return dtype.Q3_K, nil
	case ggmlQ4_K:
		return dtype.Q4_K, nil
	case ggmlQ5_K:
		return dtype.Q5_K, nil
	case ggmlQ6_K:
		return dtype.Q6_K, nil
	case ggmlQ8_K:
		return dtype.Q8_K, nil
	}
	return 0, fmt.Errorf("gguf: unsupported ggml type %d", t)
}

// dtypeToGGML is the inverse mapping, used when writing test fixtures.
func dtypeToGGML(dt dtype.DType) (uint32, error) {
	switch dt {
	case dtype.F32:
		return ggmlF32, nil
	case dtype.F16:
		return ggmlF16, nil
	case dtype.BF16:
		return ggmlBF16, nil
	case dtype.F64:
		return ggmlF64, nil
	case dtype.I8:
		return ggmlI8, nil
	case dtype.I16:
		return ggmlI16, nil
	case dtype.I32:
		return ggmlI32, nil
	case dtype.I64:
		return ggmlI64, nil
	case dtype.Q4_0:
		return ggmlQ4_0, nil
	case dtype.Q4_1:
		return ggmlQ4_1, nil
	case dtype.Q5_0:
		return ggmlQ5_0, nil
	case dtype.Q5_1:
		return ggmlQ5_1, nil
	case dtype.Q8_0:
		return ggmlQ8_0, nil
	case dtype.Q8_1:
		return ggmlQ8_1, nil
	case dtype.Q2_K:
		return ggmlQ2_K, nil
	case dtype.Q3_K:
		return ggmlQ3_K, nil
	case dtype.Q4_K:
		return ggmlQ4_K, nil
	case dtype.Q5_K:
		return ggmlQ5_K, nil
	case dtype.Q6_K:
		return ggmlQ6_K, nil
	case dtype.Q8_K:
		return ggmlQ8_K, nil
	}
	return 0, fmt.Errorf("gguf: dtype %v has no ggml equivalent", dt)
}

// GGMLType returns the ggml tensor-type code for a dtype (for Writer.AddTensor).
func GGMLType(dt dtype.DType) (uint32, error) { return dtypeToGGML(dt) }
