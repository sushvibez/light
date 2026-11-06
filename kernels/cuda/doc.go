// Package cuda holds the GPU kernel launch plumbing: PTX loading (via
// cuModuleLoadDataEx), per-kernel cuFunction caching, and registry wiring that
// connects ops.<Op>() dispatch to the underlying CUDA kernels.
//
// All functionality is behind the "cuda" build tag. Builds without the tag
// still see this package declaration but contain no code, so the rest of the
// tree compiles CPU-only.
package cuda
