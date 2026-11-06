// Index ops (2026-10-28): gather, scatter (set), scatter_add and index_add
// for F32 / F64 / I32 / I64 / BF16 / F16, one thread per source element.
//
// Layout contract (shared with kernels/cpu/gather.go and the Go launchers):
// the indexed axis `dim` splits a contiguous tensor into (outer, axis,
// inner); element (o, j, i) lives at o·axis·inner + j·inner + i.
//
//   gather:      out[o, j, i]           = src[o, idx[o, j, i], i]
//   scatter:     out[o, idx[o, j, i], i] = src[o, j, i]        (last writer wins)
//   scatter_add: out[o, idx[o, j, i], i] += src[o, j, i]       (atomic)
//   index_add:   out[o, idx[j], i]      += src[o, j, i]        (atomic, idx is 1-D)
//
// gather_f32 / scatter_add_f32 predate this file and live in elementwise.cu.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define GATHER(NAME, T)                                                                   \
  extern "C" __global__ void NAME(const T *src, const long long *indices, T *out,         \
                                  unsigned long outer, unsigned long outAxis,             \
                                  unsigned long srcAxis, unsigned long inner) {           \
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;               \
    if (i >= outer * outAxis * inner) return;                                             \
    unsigned long innerIdx = i % inner;                                                   \
    unsigned long outerIdx = i / (outAxis * inner);                                       \
    out[i] = src[outerIdx * srcAxis * inner + (unsigned long)indices[i] * inner + innerIdx]; \
  }

#define SCATTER(NAME, T, STORE)                                                           \
  extern "C" __global__ void NAME(const T *src, const long long *indices, T *out,         \
                                  unsigned long outer, unsigned long srcAxis,             \
                                  unsigned long outAxis, unsigned long inner) {           \
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;               \
    if (i >= outer * srcAxis * inner) return;                                             \
    unsigned long innerIdx = i % inner;                                                   \
    unsigned long outerIdx = i / (srcAxis * inner);                                       \
    unsigned long off = outerIdx * outAxis * inner + (unsigned long)indices[i] * inner + innerIdx; \
    STORE;                                                                                \
  }

#define INDEX_ADD(NAME, T, ADD)                                                           \
  extern "C" __global__ void NAME(const T *src, const long long *indices, T *out,         \
                                  unsigned long outer, unsigned long srcAxis,             \
                                  unsigned long outAxis, unsigned long inner) {           \
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;               \
    if (i >= outer * srcAxis * inner) return;                                             \
    unsigned long innerIdx = i % inner;                                                   \
    unsigned long j = (i / inner) % srcAxis;                                              \
    unsigned long outerIdx = i / (srcAxis * inner);                                       \
    unsigned long off = outerIdx * outAxis * inner + (unsigned long)indices[j] * inner + innerIdx; \
    ADD;                                                                                  \
  }

#define SET out[off] = src[i]
#define ATOMIC atomicAdd(&out[off], src[i])
#define ATOMIC_I64 atomicAdd((unsigned long long *)&out[off], (unsigned long long)src[i])

GATHER(gather_f64, double)
GATHER(gather_i32, int)
GATHER(gather_i64, long long)
GATHER(gather_bf16, __nv_bfloat16)
GATHER(gather_f16, __half)

SCATTER(scatter_f32, float, SET)
SCATTER(scatter_f64, double, SET)
SCATTER(scatter_i32, int, SET)
SCATTER(scatter_i64, long long, SET)
SCATTER(scatter_bf16, __nv_bfloat16, SET)
SCATTER(scatter_f16, __half, SET)

SCATTER(scatter_add_f64, double, ATOMIC)
SCATTER(scatter_add_i32, int, ATOMIC)
SCATTER(scatter_add_i64, long long, ATOMIC_I64)
SCATTER(scatter_add_bf16, __nv_bfloat16, ATOMIC)
SCATTER(scatter_add_f16, __half, ATOMIC)

INDEX_ADD(index_add_f32, float, ATOMIC)
INDEX_ADD(index_add_f64, double, ATOMIC)
INDEX_ADD(index_add_i32, int, ATOMIC)
INDEX_ADD(index_add_i64, long long, ATOMIC_I64)
INDEX_ADD(index_add_bf16, __nv_bfloat16, ATOMIC)
INDEX_ADD(index_add_f16, __half, ATOMIC)

// Row gather on raw bytes (block-quantized embedding tables and dtypes
// without a typed kernel): out[i] = src[indices[i]], rows of row_bytes.
extern "C" __global__ void index_select_bytes(const unsigned char *src, const long long *indices,
                                              unsigned char *out, unsigned long row_bytes,
                                              unsigned long total) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  unsigned long row = i / row_bytes, off = i - row * row_bytes;
  out[i] = src[(unsigned long)indices[row] * row_bytes + off];
}
