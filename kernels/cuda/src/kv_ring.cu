// Ring-buffer key / value cache and single-query (decode) attention against
// it: ops.KVRingWrite and ops.AttnDecode.

#include <cuda_runtime.h>

#define AD_KEYS 128  // keys per block of attn_decode_partial_f32 (AttnDecodeKeys in kv_ring.go)

// src (B, H, T, D) with element strides sb, sh, st (last axis contiguous)
// -> dst (B, H, Cap, D), position t at slot (slot0 + t) % Cap.
extern "C" __global__ void kv_ring_write_f32(const float *src, float *dst, int B, int H, int T, int D, int Cap,
                                             int slot0, long long sb, long long sh, long long st) {
  long long n = (long long)B * H * T * D;
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int d = (int)(i % D);
  long long r = i / D;
  int t = (int)(r % T);
  r /= T;
  int h = (int)(r % H);
  int b = (int)(r / H);
  int slot = (slot0 + t) % Cap;
  dst[(((long long)b * H + h) * Cap + slot) * D + d] = src[b * sb + h * sh + t * st + d];
}

// Block-wide max or sum; red holds 33 floats. Every thread gets the result.
__device__ float ad_block_reduce(float x, float *red, bool isMax) {
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

// Split-K decode attention, pass 1. Grid (B*H, splits), a multiple of 32
// threads. Block (bh, s) takes keys [s*AD_KEYS, min(n, (s+1)*AD_KEYS)) and
// writes part[(bh*splits + s)*(D+2) + {0: max, 1: sum of exp, 2..: sum p v}].
// Dynamic shared memory: (D + AD_KEYS + 33) floats.
extern "C" __global__ void attn_decode_partial_f32(const float *Q, const float *K, const float *V, float *part,
                                                   int H, int Cap, int D, int n, float scale,
                                                   long long qb, long long qh) {
  extern __shared__ float sm[];
  float *q = sm, *p = sm + D, *red = sm + D + AD_KEYS;
  int bh = blockIdx.x, s = blockIdx.y, splits = gridDim.y;
  int b = bh / H, h = bh % H;
  int j0 = s * AD_KEYS;
  int cnt = min(AD_KEYS, n - j0);
  const float *qp = Q + b * qb + h * qh;
  for (int d = threadIdx.x; d < D; d += blockDim.x) q[d] = qp[d];
  __syncthreads();
  const float *Kb = K + ((long long)bh * Cap + j0) * D;
  const float *Vb = V + ((long long)bh * Cap + j0) * D;
  int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int j = w; j < cnt; j += nw) {  // one warp per key: coalesced rows
    float acc = 0.f;
    for (int d = lane; d < D; d += 32) acc += q[d] * Kb[(long long)j * D + d];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) p[j] = acc * scale;
  }
  __syncthreads();
  float m = -INFINITY;
  for (int j = threadIdx.x; j < cnt; j += blockDim.x) m = fmaxf(m, p[j]);
  m = ad_block_reduce(m, red, true);
  float l = 0.f;
  for (int j = threadIdx.x; j < cnt; j += blockDim.x) {
    float e = expf(p[j] - m);
    p[j] = e;
    l += e;
  }
  l = ad_block_reduce(l, red, false);  // its barriers also publish p
  float *out = part + ((long long)bh * splits + s) * (D + 2);
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float a = 0.f;
    for (int j = 0; j < cnt; j++) a += p[j] * Vb[(long long)j * D + d];
    out[2 + d] = a;
  }
  if (threadIdx.x == 0) {
    out[0] = m;
    out[1] = l;
  }
}

// Pass 2: one block per (b, h) merges the splits into O (B*H, D).
extern "C" __global__ void attn_decode_combine_f32(const float *part, float *O, int splits, int D) {
  int bh = blockIdx.x;
  const float *pp = part + (long long)bh * splits * (D + 2);
  float M = -INFINITY;
  for (int s = 0; s < splits; s++) M = fmaxf(M, pp[s * (D + 2)]);
  float L = 0.f;
  for (int s = 0; s < splits; s++) L += expf(pp[s * (D + 2)] - M) * pp[s * (D + 2) + 1];
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float a = 0.f;
    for (int s = 0; s < splits; s++) a += expf(pp[s * (D + 2)] - M) * pp[s * (D + 2) + 2 + d];
    O[(long long)bh * D + d] = a / L;
  }
}
