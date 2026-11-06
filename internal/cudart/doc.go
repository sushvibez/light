// Package cudart wraps the CUDA Runtime API, cuBLAS, and cuDNN via CGo.
//
// All real functionality is behind the "cuda" build tag. Builds without the
// tag see only this package declaration and the stubs in nocuda.go, keeping
// the rest of the tree compilable CPU-only.
package cudart
