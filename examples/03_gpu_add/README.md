# Example 03 — first real GPU op

Allocates 1M-element F32 tensors, uploads to GPU, runs `Add` and `SiLU` on GPU,
downloads the result. Asserts the device is CUDA at startup (no silent CPU
fallback).

## Build & run

```bash
# once: produce PTX (needs nvcc; output is //go:embed'd into the binary)
bash scripts/build_ptx.sh

# run
go run -tags cuda ./examples/03_gpu_add
```

## Expected output (approximate, timings will vary)

```
device=CUDA(0)
uploading tensors... done (1.5ms)
add on GPU... done (0.3ms)
silu(sum) on GPU... done (0.2ms)
downloading to CPU... done (1.1ms)
first 5: [...]
 last 5: [...]
dtype=F32 shape=(1048576) total elements=1048576
```
