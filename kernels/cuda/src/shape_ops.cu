// Index-mapping shape kernels (F32): pad, flip, roll.
//
// Every kernel decomposes the row-major tensor around the dim it acts on as
// (outer, size, inner) — flat = (o * size + d) * inner + i — and runs one
// thread per OUTPUT element, so there are no races and no zero-fill pass.

#include <cuda_runtime.h>

// out[(o*outSize + d)*inner + i] = (left <= d < left+size) ? in[(o*size + d-left)*inner + i] : 0
extern "C" __global__ void pad_f32(const float *in, float *out,
                                   unsigned long outer, unsigned long size,
                                   unsigned long outSize, unsigned long inner,
                                   long left) {
  unsigned long idx = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long n = outer * outSize * inner;
  if (idx >= n) return;
  unsigned long i = idx % inner;
  unsigned long rest = idx / inner;
  long d = (long)(rest % outSize);
  unsigned long o = rest / outSize;
  long src = d - left;
  if (src < 0 || src >= (long)size) {
    out[idx] = 0.0f;
  } else {
    out[idx] = in[(o * size + (unsigned long)src) * inner + i];
  }
}

// out[(o*size + d)*inner + i] = in[(o*size + (size-1-d))*inner + i]
extern "C" __global__ void flip_f32(const float *in, float *out,
                                    unsigned long outer, unsigned long size,
                                    unsigned long inner) {
  unsigned long idx = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long n = outer * size * inner;
  if (idx >= n) return;
  unsigned long i = idx % inner;
  unsigned long rest = idx / inner;
  unsigned long d = rest % size;
  unsigned long o = rest / size;
  out[idx] = in[(o * size + (size - 1 - d)) * inner + i];
}

// out[(o*size + d)*inner + i] = in[(o*size + (d - shift mod size))*inner + i]
// shift is pre-normalised by the host into [0, size).
extern "C" __global__ void roll_f32(const float *in, float *out,
                                    unsigned long outer, unsigned long size,
                                    unsigned long inner, unsigned long shift) {
  unsigned long idx = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long n = outer * size * inner;
  if (idx >= n) return;
  unsigned long i = idx % inner;
  unsigned long rest = idx / inner;
  unsigned long d = rest % size;
  unsigned long o = rest / size;
  unsigned long src = (d + size - shift) % size;
  out[idx] = in[(o * size + src) * inner + i];
}
