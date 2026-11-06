// Row-wise sort along the last axis (values + original indices), bitonic.
//
// ops.Sort pads every row to n2 = next power of two in scratch buffers
// (values T, indices long long), runs the bitonic network, then gathers the
// first `cols` entries back. Padding carries the sentinel that sorts last
// for the requested direction and index LLONG_MAX, so it never leaks into
// the output. Ties are broken by original index (ascending) in both
// directions, which reproduces Go's stable sort on the CPU path.
//
// Two paths: rows with n2 <= 1024 sort in shared memory with one block per
// row; larger rows use the global-memory network, one launch per (k, j)
// stage (log2(n2)·(log2(n2)+1)/2 launches).

#include <cuda_runtime.h>
#include <math_constants.h>
#include <limits.h>

#define SORT_INSTANCE(SUFFIX, T, POS_SENT, NEG_SENT)                                         \
  __device__ __forceinline__ bool before_##SUFFIX(T va, long long ia, T vb, long long ib,  \
                                                  int desc) {                              \
    if (va == vb) return ia < ib;                                                           \
    return desc ? (va > vb) : (va < vb);                                                    \
  }                                                                                         \
                                                                                            \
  extern "C" __global__ void sort_init_##SUFFIX(const T *in, T *vals, long long *idx,       \
                                                int rows, int cols, int n2, int desc) {    \
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;                      \
    if (tid >= (long long)rows * n2) return;                                                \
    int r = (int)(tid / n2), j = (int)(tid % n2);                                           \
    if (j < cols) {                                                                         \
      vals[tid] = in[(long long)r * cols + j];                                              \
      idx[tid] = j;                                                                         \
    } else {                                                                                \
      vals[tid] = desc ? (NEG_SENT) : (POS_SENT);                                           \
      idx[tid] = LLONG_MAX;                                                                 \
    }                                                                                       \
  }                                                                                         \
                                                                                            \
  extern "C" __global__ void sort_step_##SUFFIX(T *vals, long long *idx, int rows, int n2,  \
                                                int j, int k, int desc) {                  \
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;                      \
    if (tid >= (long long)rows * n2) return;                                                \
    int i = (int)(tid % n2);                                                                \
    int p = i ^ j;                                                                          \
    if (p <= i) return;                                                                     \
    long long base = tid - i;                                                               \
    T vi = vals[base + i], vp = vals[base + p];                                             \
    long long ii = idx[base + i], ip = idx[base + p];                                       \
    bool up = (i & k) == 0;                                                                 \
    bool swap = up ? before_##SUFFIX(vp, ip, vi, ii, desc) : before_##SUFFIX(vi, ii, vp, ip, desc); \
    if (swap) {                                                                             \
      vals[base + i] = vp; vals[base + p] = vi;                                             \
      idx[base + i] = ip;  idx[base + p] = ii;                                              \
    }                                                                                       \
  }                                                                                         \
                                                                                            \
  /* one block per row, blockDim.x == n2 <= 1024 */                                        \
  extern "C" __global__ void sort_shared_##SUFFIX(const T *in, T *outV, long long *outI,    \
                                                  int cols, int n2, int desc) {            \
    __shared__ T sv[1024];                                                                  \
    __shared__ long long si[1024];                                                          \
    int r = blockIdx.x, i = threadIdx.x;                                                    \
    if (i < cols) { sv[i] = in[(long long)r * cols + i]; si[i] = i; }                       \
    else { sv[i] = desc ? (NEG_SENT) : (POS_SENT); si[i] = LLONG_MAX; }                     \
    __syncthreads();                                                                        \
    for (int k = 2; k <= n2; k <<= 1) {                                                     \
      for (int j = k >> 1; j > 0; j >>= 1) {                                                \
        int p = i ^ j;                                                                      \
        if (p > i) {                                                                        \
          T vi = sv[i], vp = sv[p];                                                         \
          long long ii = si[i], ip = si[p];                                                 \
          bool up = (i & k) == 0;                                                           \
          bool swap = up ? before_##SUFFIX(vp, ip, vi, ii, desc)                            \
                         : before_##SUFFIX(vi, ii, vp, ip, desc);                           \
          if (swap) { sv[i] = vp; sv[p] = vi; si[i] = ip; si[p] = ii; }                     \
        }                                                                                   \
        __syncthreads();                                                                    \
      }                                                                                     \
    }                                                                                       \
    if (i < cols) { outV[(long long)r * cols + i] = sv[i]; outI[(long long)r * cols + i] = si[i]; } \
  }                                                                                         \
                                                                                            \
  extern "C" __global__ void sort_gather_##SUFFIX(const T *vals, const long long *idx,      \
                                                  T *outV, long long *outI, int rows,       \
                                                  int cols, int n2) {                       \
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;                      \
    if (tid >= (long long)rows * cols) return;                                              \
    int r = (int)(tid / cols), j = (int)(tid % cols);                                       \
    outV[tid] = vals[(long long)r * n2 + j];                                                \
    outI[tid] = idx[(long long)r * n2 + j];                                                 \
  }

SORT_INSTANCE(f32, float, CUDART_INF_F, -CUDART_INF_F)
SORT_INSTANCE(f64, double, CUDART_INF, -CUDART_INF)
SORT_INSTANCE(i32, int, INT_MAX, INT_MIN)
SORT_INSTANCE(i64, long long, LLONG_MAX, LLONG_MIN)
