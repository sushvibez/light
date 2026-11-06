// FlashAttention forward (F32), ported from accelerate's k_flash_attn /
// k_flash_attn_tiled.
//
// Layout: Q, K, V, O are (BH, S, D) row-major; one batch-head is a
// contiguous S*D slab. Each query row keeps an online softmax (running max
// m, running denominator l) and a D-wide accumulator, so the (S, S) score
// matrix is never materialised.
//
// Three variants, chosen by the host (kernels/cuda/flash_attn.go):
//   flash_attn_simple_f32   one thread per query row, K/V read from global.
//                           Fallback when shared memory is too small.
//   flash_attn_tiled_f32    a block owns TILE_Q query rows of one batch-head
//                           and cooperatively stages tileK keys/values into
//                           shared memory (each key read once per block, not
//                           once per query). Runtime D <= 256, accumulator
//                           lives in local memory.
//   flash_attn_tiled_d{32,64,128}_f32
//                           same algorithm with D fixed at compile time so
//                           the q row and accumulator stay in registers and
//                           the inner loops fully unroll.

#include <cuda_runtime.h>

#define FA_MAX_D 256

// Dynamic shared memory shared by every tiled variant (K tile then V tile).
extern __shared__ float fa_smem[];

extern "C" __global__ void flash_attn_simple_f32(
    const float *Q, const float *K, const float *V, float *O,
    int BH, int S, int D, float scale, int causal) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // over BH*S query rows
  if (idx >= BH * S) return;
  int b = idx / S, i = idx % S;
  const float *q = Q + (long)(b * S + i) * D;
  float m = -1e30f, l = 0.f;
  float acc[FA_MAX_D];
  for (int d = 0; d < D; d++) acc[d] = 0.f;
  int jmax = causal ? (i + 1) : S;
  for (int j = 0; j < jmax; j++) {
    const float *k = K + (long)(b * S + j) * D;
    float s = 0.f;
    for (int d = 0; d < D; d++) s += q[d] * k[d];
    s *= scale;
    float mnew = fmaxf(m, s);
    float corr = expf(m - mnew);
    float p = expf(s - mnew);
    l = l * corr + p;
    const float *v = V + (long)(b * S + j) * D;
    for (int d = 0; d < D; d++) acc[d] = acc[d] * corr + p * v[d];
    m = mnew;
  }
  float *o = O + (long)(b * S + i) * D;
  float inv = (l > 0.f) ? 1.f / l : 0.f;
  for (int d = 0; d < D; d++) o[d] = acc[d] * inv;
}

extern "C" __global__ void flash_attn_tiled_f32(
    const float *Q, const float *K, const float *V, float *O,
    int BH, int S, int D, float scale, int causal, int tileK) {
  int b = blockIdx.y;
  int qi = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = qi < S;
  const float *q = active ? (Q + (long)(b * S + qi) * D) : Q;
  float m = -1e30f, l = 0.f;
  float acc[FA_MAX_D];
  for (int d = 0; d < D; d++) acc[d] = 0.f;

  float *sK = fa_smem;
  float *sV = fa_smem + tileK * D;

  for (int kStart = 0; kStart < S; kStart += tileK) {
    int kCount = min(tileK, S - kStart);
    for (int e = threadIdx.x; e < kCount * D; e += blockDim.x) {
      int kk = e / D, dd = e % D;
      sK[e] = K[(long)(b * S + kStart + kk) * D + dd];
      sV[e] = V[(long)(b * S + kStart + kk) * D + dd];
    }
    __syncthreads();
    if (active) {
      for (int j = 0; j < kCount; j++) {
        if (causal && (kStart + j) > qi) break;
        float s = 0.f;
        for (int d = 0; d < D; d++) s += q[d] * sK[j * D + d];
        s *= scale;
        float mnew = fmaxf(m, s);
        float corr = expf(m - mnew), p = expf(s - mnew);
        l = l * corr + p;
        for (int d = 0; d < D; d++) acc[d] = acc[d] * corr + p * sV[j * D + d];
        m = mnew;
      }
    }
    __syncthreads();
  }
  if (active) {
    float inv = (l > 0.f) ? 1.f / l : 0.f;
    float *o = O + (long)(b * S + qi) * D;
    for (int d = 0; d < D; d++) o[d] = acc[d] * inv;
  }
}

// Compile-time-D variant: q and acc are register arrays, loops unroll.
template <int DD>
__device__ __forceinline__ void flash_attn_tiled_fixed(
    const float *Q, const float *K, const float *V, float *O,
    int BH, int S, float scale, int causal, int tileK) {
  int b = blockIdx.y;
  int qi = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = qi < S;
  float qr[DD];
  float acc[DD];
  const float *q = active ? (Q + (long)(b * S + qi) * DD) : Q;
#pragma unroll
  for (int d = 0; d < DD; d++) {
    qr[d] = q[d] * scale;
    acc[d] = 0.f;
  }
  float m = -1e30f, l = 0.f;

  float *sK = fa_smem;
  float *sV = fa_smem + tileK * DD;

  for (int kStart = 0; kStart < S; kStart += tileK) {
    int kCount = min(tileK, S - kStart);
    for (int e = threadIdx.x; e < kCount * DD; e += blockDim.x) {
      int kk = e / DD, dd = e % DD;
      sK[e] = K[(long)(b * S + kStart + kk) * DD + dd];
      sV[e] = V[(long)(b * S + kStart + kk) * DD + dd];
    }
    __syncthreads();
    if (active) {
      int jEnd = kCount;
      if (causal && (qi - kStart + 1) < jEnd) jEnd = qi - kStart + 1;
      for (int j = 0; j < jEnd; j++) {
        const float *kr = sK + j * DD;
        float s = 0.f;
#pragma unroll
        for (int d = 0; d < DD; d++) s += qr[d] * kr[d];
        float mnew = fmaxf(m, s);
        float corr = __expf(m - mnew), p = __expf(s - mnew);
        l = l * corr + p;
        const float *vr = sV + j * DD;
#pragma unroll
        for (int d = 0; d < DD; d++) acc[d] = acc[d] * corr + p * vr[d];
        m = mnew;
      }
    }
    __syncthreads();
  }
  if (active) {
    float inv = (l > 0.f) ? 1.f / l : 0.f;
    float *o = O + (long)(b * S + qi) * DD;
#pragma unroll
    for (int d = 0; d < DD; d++) o[d] = acc[d] * inv;
  }
}

extern "C" __global__ void flash_attn_tiled_d32_f32(
    const float *Q, const float *K, const float *V, float *O,
    int BH, int S, float scale, int causal, int tileK) {
  flash_attn_tiled_fixed<32>(Q, K, V, O, BH, S, scale, causal, tileK);
}

extern "C" __global__ void flash_attn_tiled_d64_f32(
    const float *Q, const float *K, const float *V, float *O,
    int BH, int S, float scale, int causal, int tileK) {
  flash_attn_tiled_fixed<64>(Q, K, V, O, BH, S, scale, causal, tileK);
}

extern "C" __global__ void flash_attn_tiled_d128_f32(
    const float *Q, const float *K, const float *V, float *O,
    int BH, int S, float scale, int causal, int tileK) {
  flash_attn_tiled_fixed<128>(Q, K, V, O, BH, S, scale, causal, tileK);
}
