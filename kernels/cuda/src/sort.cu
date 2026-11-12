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

// ------------------- Row-wise argmax and small top-k -------------------
//
// argmax_row_f32: one 256-thread block per row, tree reduction; ties go to
// the lowest index (the CPU kernel's first-max rule). Replaces the one
// thread-per-row scan, which took ~1 ms on a 250k-entry logits row.
//
// topk_row_f32: the k (≤ 64) largest (or smallest) entries of each row with
// their indices, descending (ascending) and ties by index, in one block per
// row: every thread keeps a sorted list of the best k entries among the
// columns it strides over, then k rounds of a block-wide "best head" pick
// merge the 256 lists. For k = 40 on a 250k row this is ~50 µs where the
// bitonic full-row sort was 171 launches. Sampling a token needs only this.

#define ARG_THREADS 256
#define TOPK_MAX 64

__device__ __forceinline__ bool arg_better(float va, long long ia, float vb, long long ib) {
  // "a beats b": larger value, or equal value and lower index.
  return va > vb || (va == vb && ia < ib);
}

extern "C" __global__ void argmax_row_f32(const float *a, long long *out, long long rows,
                                          long long cols) {
  __shared__ float sv[ARG_THREADS];
  __shared__ long long si[ARG_THREADS];
  long long row = blockIdx.x;
  if (row >= rows) return;
  const float *r = a + row * cols;
  float bv = -CUDART_INF_F;
  long long bi = LLONG_MAX;
  for (long long j = threadIdx.x; j < cols; j += ARG_THREADS) {
    float v = r[j];
    if (arg_better(v, j, bv, bi)) { bv = v; bi = j; }
  }
  sv[threadIdx.x] = bv;
  si[threadIdx.x] = bi;
  __syncthreads();
  for (int s = ARG_THREADS / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      float ov = sv[threadIdx.x + s];
      long long oi = si[threadIdx.x + s];
      if (arg_better(ov, oi, sv[threadIdx.x], si[threadIdx.x])) {
        sv[threadIdx.x] = ov;
        si[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) out[row] = (si[0] == LLONG_MAX) ? 0 : si[0];
}

extern "C" __global__ void topk_row_f32(const float *a, float *vals, long long *idx, long long rows,
                                        long long cols, int k, int largest) {
  __shared__ float sv[ARG_THREADS];
  __shared__ long long si[ARG_THREADS];
  __shared__ int st[ARG_THREADS];
  long long row = blockIdx.x;
  if (row >= rows || k <= 0) return;
  const float *r = a + row * cols;
  float lv[TOPK_MAX];
  long long li[TOPK_MAX];
  int n = 0;
  float sign = largest ? 1.f : -1.f;
  for (long long j = threadIdx.x; j < cols; j += ARG_THREADS) {
    float v = sign * r[j];
    if (n == k && !arg_better(v, j, lv[n - 1], li[n - 1])) continue;
    int p = (n < k) ? n : n - 1; // slot to fill (drop the tail when full)
    while (p > 0 && arg_better(v, j, lv[p - 1], li[p - 1])) {
      lv[p] = lv[p - 1];
      li[p] = li[p - 1];
      p--;
    }
    lv[p] = v;
    li[p] = j;
    if (n < k) n++;
  }
  int ptr = 0;
  for (int round = 0; round < k; round++) {
    sv[threadIdx.x] = ptr < n ? lv[ptr] : -CUDART_INF_F;
    si[threadIdx.x] = ptr < n ? li[ptr] : LLONG_MAX;
    st[threadIdx.x] = threadIdx.x;
    __syncthreads();
    for (int s = ARG_THREADS / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        float ov = sv[threadIdx.x + s];
        long long oi = si[threadIdx.x + s];
        if (arg_better(ov, oi, sv[threadIdx.x], si[threadIdx.x])) {
          sv[threadIdx.x] = ov;
          si[threadIdx.x] = oi;
          st[threadIdx.x] = st[threadIdx.x + s];
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      vals[row * k + round] = sign * sv[0];
      idx[row * k + round] = si[0];
    }
    if (threadIdx.x == st[0]) ptr++;
    __syncthreads();
  }
}
