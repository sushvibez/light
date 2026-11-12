// Decode-step attention against a fixed-capacity KV cache whose length lives
// on the device, so one step's launches are the same whatever the position
// and the step can be captured into a CUDA graph: ops.KVAppend,
// ops.DecodeAttention and ops.IncrementI64.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define DA_KEYS 128  // keys per block of decode_attn_partial (decodeAttnKeys in decode_attn.go)
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

// Block-wide max or sum; red holds 33 floats. Every thread gets the result.
__device__ float da_block_reduce(float x, float *red, bool isMax) {
  int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int o = 16; o > 0; o >>= 1) {
    float y = __shfl_down_sync(0xffffffffu, x, o);
    x = isMax ? fmaxf(x, y) : x + y;
  }
  if (lane == 0) red[w] = x;
  __syncthreads();
  if (threadIdx.x == 0) {
    float a = red[0];
    for (int i = 1; i < nw; i++) a = isMax ? fmaxf(a, red[i]) : a + red[i];
    red[32] = a;
  }
  __syncthreads();
  float a = red[32];
  __syncthreads();
  return a;
}

// Pass 1. Grid (B*KH, ceil(Cap / DA_KEYS)), 128 threads. Block (b, kh, s)
// serves the g = NH / KH query heads of KV head kh over keys
// [s*DA_KEYS, min(n, (s+1)*DA_KEYS)) where n = pos[0] + 1 is the number
// of valid keys; blocks past n return at once, so the grid does not depend
// on the position. Each K row is read once for all g heads (a warp per
// key, lanes across D), each V row once (threads across D). Writes
// part[((b*NH + kh*g + h)*splits + s)*(D+2) + {0: max, 1: sum of exp,
// 2..: sum p v}]. Dynamic shared memory: (g*D + g*DA_KEYS + 33) floats.
template <typename T>
__device__ __forceinline__ void decode_attn_partial_t(const T *Q, const T *K, const T *V, float *part, const long long *pos,
                                                      int NH, int KH, int Cap, int D, float scale) {
  extern __shared__ float sm[];
  int g = NH / KH;
  float *q = sm, *p = sm + g * D, *red = p + g * DA_KEYS;
  int bkh = blockIdx.x, s = blockIdx.y, splits = gridDim.y;
  int b = bkh / KH, kh = bkh % KH;
  int n = (int)pos[0] + 1;
  int j0 = s * DA_KEYS;
  if (j0 >= n) return;
  int cnt = min(DA_KEYS, n - j0);
  const T *qp = Q + ((long long)b * NH + (long long)kh * g) * D;
  for (int i = threadIdx.x; i < g * D; i += blockDim.x) q[i] = da_ld(qp + i);
  __syncthreads();
  const T *Kb = K + ((long long)bkh * Cap + j0) * D;
  const T *Vb = V + ((long long)bkh * Cap + j0) * D;
  int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  int nd = D >> 5;  // D is a multiple of 32
  for (int j = w; j < cnt; j += nw) {  // one warp per key: coalesced rows
    float kr[DA_MAX_D / 32];
    for (int c = 0; c < nd; c++) kr[c] = da_ld(Kb + (long long)j * D + lane + 32 * c);
    for (int h = 0; h < g; h++) {
      float a = 0.f;
      for (int c = 0; c < nd; c++) a += q[h * D + lane + 32 * c] * kr[c];
      for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
      if (lane == 0) p[h * DA_KEYS + j] = a * scale;
    }
  }
  __syncthreads();
  float mh[DA_MAX_G], lh[DA_MAX_G];
  for (int h = 0; h < g; h++) {
    float *ph = p + h * DA_KEYS;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < cnt; j += blockDim.x) m = fmaxf(m, ph[j]);
    m = da_block_reduce(m, red, true);
    float l = 0.f;
    for (int j = threadIdx.x; j < cnt; j += blockDim.x) {
      float e = expf(ph[j] - m);
      ph[j] = e;
      l += e;
    }
    l = da_block_reduce(l, red, false);  // its barriers also publish ph
    mh[h] = m;
    lh[h] = l;
  }
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc[DA_MAX_G];
    for (int h = 0; h < g; h++) acc[h] = 0.f;
    for (int j = 0; j < cnt; j++) {
      float vv = da_ld(Vb + (long long)j * D + d);
      for (int h = 0; h < g; h++) acc[h] += p[h * DA_KEYS + j] * vv;
    }
    for (int h = 0; h < g; h++) part[(((long long)b * NH + (long long)kh * g + h) * splits + s) * (D + 2) + 2 + d] = acc[h];
  }
  if (threadIdx.x == 0) {
    for (int h = 0; h < g; h++) {
      float *out = part + (((long long)b * NH + (long long)kh * g + h) * splits + s) * (D + 2);
      out[0] = mh[h];
      out[1] = lh[h];
    }
  }
}
extern "C" __global__ void decode_attn_partial_f32(const float *Q, const float *K, const float *V, float *part,
                                                   const long long *pos, int NH, int KH, int Cap, int D, float scale) {
  decode_attn_partial_t(Q, K, V, part, pos, NH, KH, Cap, D, scale);
}
extern "C" __global__ void decode_attn_partial_bf16(const __nv_bfloat16 *Q, const __nv_bfloat16 *K, const __nv_bfloat16 *V,
                                                    float *part, const long long *pos, int NH, int KH, int Cap, int D,
                                                    float scale) {
  decode_attn_partial_t(Q, K, V, part, pos, NH, KH, Cap, D, scale);
}

// Pass 2: one block per (b, nh) merges the ceil(n / DA_KEYS) live splits
// into O (B*NH, D).
template <typename T>
__device__ __forceinline__ void decode_attn_combine_t(const float *part, T *O, const long long *pos, int splits, int D) {
  int bh = blockIdx.x;
  int n = (int)pos[0] + 1;
  int ns = (n + DA_KEYS - 1) / DA_KEYS;
  const float *pp = part + (long long)bh * splits * (D + 2);
  float M = -INFINITY;
  for (int s = 0; s < ns; s++) M = fmaxf(M, pp[s * (D + 2)]);
  float L = 0.f;
  for (int s = 0; s < ns; s++) L += expf(pp[s * (D + 2)] - M) * pp[s * (D + 2) + 1];
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float a = 0.f;
    for (int s = 0; s < ns; s++) a += expf(pp[s * (D + 2)] - M) * pp[s * (D + 2) + 2 + d];
    da_st(O + (long long)bh * D + d, a / L);
  }
}
extern "C" __global__ void decode_attn_combine_f32(const float *part, float *O, const long long *pos, int splits, int D) {
  decode_attn_combine_t(part, O, pos, splits, D);
}
extern "C" __global__ void decode_attn_combine_bf16(const float *part, __nv_bfloat16 *O, const long long *pos, int splits,
                                                    int D) {
  decode_attn_combine_t(part, O, pos, splits, D);
}

// p[i] += 1 for i < n (the device-side position counter after a step).
extern "C" __global__ void inc_i64(long long *p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += 1;
}
