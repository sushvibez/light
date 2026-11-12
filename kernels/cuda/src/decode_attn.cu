// Decode-step attention against a fixed-capacity KV cache whose length lives
// on the device, so one step's launches are the same whatever the position
// and the step can be captured into a CUDA graph: ops.KVAppend,
// ops.DecodeAttention and ops.IncrementI64.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define DA_KEYS 32   // keys per block of decode_attn_partial (decodeAttnKeys in decode_attn.go)
#define DA_MAX_G 16  // query heads per KV head (GQA group)
#define DA_MAX_D 256 // head dim

__device__ __forceinline__ float da_ld(const float *p) { return *p; }
__device__ __forceinline__ float da_ld(const __nv_bfloat16 *p) { return __bfloat162float(*p); }
__device__ __forceinline__ void da_st(float *p, float v) { *p = v; }
__device__ __forceinline__ void da_st(__nv_bfloat16 *p, float v) { *p = __float2bfloat16(v); }

// k, v (B, KH, 1, D) contiguous -> kbuf, vbuf (B, KH, Cap, D) at position pos[0].
template <typename T>
__device__ __forceinline__ void kv_append_t(const T *k, const T *v, T *kbuf, T *vbuf, const long long *pos,
                                            int B, int KH, int D, int Cap) {
  long long n = (long long)B * KH * D;
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int d = (int)(i % D);
  long long bh = i / D;
  long long dst = (bh * Cap + pos[0]) * D + d;
  kbuf[dst] = k[i];
  vbuf[dst] = v[i];
}
extern "C" __global__ void kv_append_f32(const float *k, const float *v, float *kbuf, float *vbuf, const long long *pos,
                                         int B, int KH, int D, int Cap) {
  kv_append_t(k, v, kbuf, vbuf, pos, B, KH, D, Cap);
}
extern "C" __global__ void kv_append_bf16(const __nv_bfloat16 *k, const __nv_bfloat16 *v, __nv_bfloat16 *kbuf,
                                          __nv_bfloat16 *vbuf, const long long *pos, int B, int KH, int D, int Cap) {
  kv_append_t(k, v, kbuf, vbuf, pos, B, KH, D, Cap);
}

__device__ __forceinline__ float2 da_ld2(const float *p) { return *reinterpret_cast<const float2 *>(p); }
__device__ __forceinline__ float2 da_ld2(const __nv_bfloat16 *p) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(p));
}

// Pass 1. Grid (B*KH, ceil(Cap / DA_KEYS)), 128 threads. Block (b, kh, s)
// serves the g = NH / KH query heads of KV head kh over keys
// [s*DA_KEYS, min(n, (s+1)*DA_KEYS)) where n = pos[0] + 1 is the number
// of valid keys; blocks past n return at once, so the grid does not depend
// on the position. Scores: one thread per (key, head) pair, the g threads
// of a key reading its row together; softmax: a warp per head over the
// block's ≤ 32 keys; p·V: a thread per dim for all g heads. Writes
// part[((b*NH + kh*g + h)*splits + s)*(D+2) + {0: max, 1: sum of exp,
// 2..: sum p v}]. Dynamic shared memory: (g*(D+1) + g*DA_KEYS) floats
// (q rows padded by one so threads on different heads hit different
// banks).
template <typename T>
__device__ __forceinline__ void decode_attn_partial_t(const T *Q, const T *K, const T *V, float *part, const long long *pos,
                                                      int NH, int KH, int Cap, int D, float scale, int window) {
  extern __shared__ float sm[];
  const int g = NH / KH;
  const int qp = D + 1;
  float *q = sm, *p = sm + g * qp;
  const int bkh = blockIdx.x, s = blockIdx.y, splits = gridDim.y;
  const int b = bkh / KH, kh = bkh % KH;
  const int n = (int)pos[0] + 1;
  // A sliding window keeps keys [n - window, n); blocks wholly before it
  // return (the combine skips them too), keys inside a block before it
  // score -inf.
  const int start = window > 0 ? max(0, n - window) : 0;
  const int j0 = s * DA_KEYS;
  if (j0 >= n || j0 + DA_KEYS <= start) return;
  const int cnt = min(DA_KEYS, n - j0);
  const T *qsrc = Q + ((long long)b * NH + (long long)kh * g) * D;
  for (int i = threadIdx.x; i < g * D; i += blockDim.x) q[(i / D) * qp + (i % D)] = da_ld(qsrc + i);
  __syncthreads();
  const T *Kb = K + ((long long)bkh * Cap + j0) * D;
  const T *Vb = V + ((long long)bkh * Cap + j0) * D;
  for (int idx = threadIdx.x; idx < g * cnt; idx += blockDim.x) {
    const int j = idx / g, h = idx % g;
    const T *kr = Kb + (long long)j * D;
    const float *qh = q + h * qp;
    float a = 0.f;
#pragma unroll 8
    for (int d = 0; d < D; d += 2) {
      const float2 kv = da_ld2(kr + d);
      a += qh[d] * kv.x + qh[d + 1] * kv.y;
    }
    p[h * DA_KEYS + j] = j0 + j < start ? -INFINITY : a * scale;
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int h = w; h < g; h += nw) {
    float v = lane < cnt ? p[h * DA_KEYS + lane] : -INFINITY;
    float m = v;
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float e = lane < cnt ? expf(v - m) : 0.f;
    float l = e;
    for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
    if (lane < cnt) p[h * DA_KEYS + lane] = e;
    if (lane == 0) {
      float *out = part + (((long long)b * NH + (long long)kh * g + h) * splits + s) * (D + 2);
      out[0] = m;
      out[1] = l;
    }
  }
  __syncthreads();
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc[DA_MAX_G];
    for (int h = 0; h < g; h++) acc[h] = 0.f;
    for (int j = 0; j < cnt; j++) {
      const float vv = da_ld(Vb + (long long)j * D + d);
      for (int h = 0; h < g; h++) acc[h] += p[h * DA_KEYS + j] * vv;
    }
    for (int h = 0; h < g; h++) part[(((long long)b * NH + (long long)kh * g + h) * splits + s) * (D + 2) + 2 + d] = acc[h];
  }
}
extern "C" __global__ void decode_attn_partial_f32(const float *Q, const float *K, const float *V, float *part,
                                                   const long long *pos, int NH, int KH, int Cap, int D, float scale,
                                                   int window) {
  decode_attn_partial_t(Q, K, V, part, pos, NH, KH, Cap, D, scale, window);
}
extern "C" __global__ void decode_attn_partial_bf16(const __nv_bfloat16 *Q, const __nv_bfloat16 *K, const __nv_bfloat16 *V,
                                                    float *part, const long long *pos, int NH, int KH, int Cap, int D,
                                                    float scale, int window) {
  decode_attn_partial_t(Q, K, V, part, pos, NH, KH, Cap, D, scale, window);
}

// Pass 2: one block per (b, nh) merges the ceil(n / DA_KEYS) live splits
// into O (B*NH, D).
template <typename T>
__device__ __forceinline__ void decode_attn_combine_t(const float *part, T *O, const long long *pos, int splits, int D,
                                                      int window) {
  int bh = blockIdx.x;
  int n = (int)pos[0] + 1;
  int start = window > 0 ? max(0, n - window) : 0;
  int s0 = start / DA_KEYS;
  int ns = (n + DA_KEYS - 1) / DA_KEYS;
  const float *pp = part + (long long)bh * splits * (D + 2);
  float M = -INFINITY;
  for (int s = s0; s < ns; s++) M = fmaxf(M, pp[s * (D + 2)]);
  float L = 0.f;
  for (int s = s0; s < ns; s++) L += expf(pp[s * (D + 2)] - M) * pp[s * (D + 2) + 1];
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float a = 0.f;
    for (int s = s0; s < ns; s++) a += expf(pp[s * (D + 2)] - M) * pp[s * (D + 2) + 2 + d];
    da_st(O + (long long)bh * D + d, a / L);
  }
}
extern "C" __global__ void decode_attn_combine_f32(const float *part, float *O, const long long *pos, int splits, int D,
                                                   int window) {
  decode_attn_combine_t(part, O, pos, splits, D, window);
}
extern "C" __global__ void decode_attn_combine_bf16(const float *part, __nv_bfloat16 *O, const long long *pos, int splits,
                                                    int D, int window) {
  decode_attn_combine_t(part, O, pos, splits, D, window);
}

// p[i] += 1 for i < n (the device-side position counter after a step).
extern "C" __global__ void inc_i64(long long *p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += 1;
}
