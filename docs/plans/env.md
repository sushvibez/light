# Development environment (discovered 2026-04-17)

| Tool            | Version / value                              |
|-----------------|----------------------------------------------|
| Go              | go1.26.2 linux/amd64                         |
| nvcc            | 13.1 V13.1.80 (built 2025-11-07)             |
| GPU             | NVIDIA GeForce RTX 5060 Ti                   |
| Driver          | 590.48.01                                    |
| Compute cap     | 12.0 (sm_120, Blackwell)                     |
| CUDA_PATH (env) | unset → default `/usr/local/cuda`            |
| cuda_runtime.h  | `/usr/local/cuda/include/cuda_runtime.h` ✓   |

**Implications:**
- CUDA 13.1 supports PTX targets up to `compute_120` natively. We will keep `compute_80` as the broad-compat baseline and add `compute_90` for Hopper-only features; a `compute_120` variant is optional and only worth building for Blackwell-only tensor core paths (deferred, not in v1 plan).
- Driver 590.xx ≥ 535 → driver-JIT cache for PTX works out of the box.
- CGo `#cgo LDFLAGS` will use `-L/usr/local/cuda/lib64` (overridable via env-expansion shim).
