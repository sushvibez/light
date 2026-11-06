// Package shape holds the value types that describe a tensor's logical layout:
// Shape (dimension list), Strides (byte-stride per axis), and broadcasting
// utilities that mirror NumPy / PyTorch semantics.
//
// All types are immutable value types; operations return new values rather than
// mutating in place. Strides are expressed in elements (not bytes) and computed
// lazily from a Shape.
package shape
