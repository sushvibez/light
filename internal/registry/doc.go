// Package registry is the op-dispatch table. Each kernel registers itself
// under a (OpName, DeviceKind, DType) key at package init; public op functions
// in package ops look up and invoke the correct kernel without importing any
// backend-specific code directly.
package registry
