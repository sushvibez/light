// Mixture-of-experts grouped GEMM (2026-10-28), the counterpart of Candle's
// candle-nn `moe_gemm` (candle-kernels/src/moe/moe_wmma.cu).
//
// Routing produces M = tokens · topk (token, slot) pairs. The caller sorts
// the pairs by expert: `sortedPairs[i]` is the flat pair id p = token·topk +
// slot of the i-th pair in expert order, and expertOffsets[e]..[e+1] is the
// range of i that belongs to expert e (an exclusive prefix sum of the per-
// expert counts, computed here by moe_count_experts + moe_expert_offsets).
//
//   out[p, :] = w(p) · input[row(p), :] @ weights[e, :, :]ᵀ
//
// with row(p) = p / topk when the input has one row per token (first MoE
// GEMM: tokens → gate/up) or row(p) = p when the input already has one row
// per pair (second GEMM: activations → down), and w(p) = topkWeights[p] or
// 1 when topkWeights is null. Every pair row is written by exactly one
// block, so no atomics and the result is deterministic. Combining the topk
// rows of a token is the caller's reduction over the (tokens, topk, N) view.
//
// grid = (numExperts, ceil(N / 32)); block = 128 threads; each block walks
// its expert's rows in 32-row tiles with 32×32 shared-memory tiles over K and
// F32 accumulation (8 outputs per thread). BF16 / F16 convert at the loads
// and stores.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void moe_count_experts(const long long *expertIDs, int *counts, unsigned long m) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) atomicAdd(&counts[expertIDs[i]], 1);
}

// Exclusive scan of counts into offsets[0..E]; one thread, E is small.
extern "C" __global__ void moe_expert_offsets(const int *counts, int *offsets, int numExperts) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int acc = 0;
  for (int e = 0; e < numExperts; e++) {
    offsets[e] = acc;
    acc += counts[e];
  }
  offsets[numExperts] = acc;
}

#define IDENT(v) (v)
#define TILE 32
#define MOE_GEMM(NAME, T, TO, FROM)                                                                 \
  extern "C" __global__ void NAME(                                                                  \
      const T *input, const T *weights, const long long *sortedPairs, const int *expertOffsets,     \
      const float *topkWeights, T *output, int topk, unsigned long sizeM,                           \
      unsigned long N, unsigned long K, int rowPerPair) {                                           \
    __shared__ float As[TILE][TILE + 1];                                                            \
    __shared__ float Bs[TILE][TILE + 1];                                                            \
    const int expert = blockIdx.x;                                                                  \
    const unsigned long nBase = (unsigned long)blockIdx.y * TILE;                                   \
    const int segStart = expertOffsets[expert];                                                     \
    const int segEnd = expertOffsets[expert + 1];                                                   \
    if (segStart >= segEnd || nBase >= N) return;                                                   \
    const T *W = weights + (size_t)expert * N * K;                                                  \
    const int t = threadIdx.x;                                                                      \
    const int ty = t / 16, tx = t % 16; /* rows ty*4..+3, cols tx*2, tx*2+1 */                      \
    for (int mBase = segStart; mBase < segEnd; mBase += TILE) {                                     \
      float acc[4][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};                           \
      /* Resolve the 32 input rows of this tile once. */                                            \
      __shared__ long long rowOf[TILE];                                                             \
      if (t < TILE) {                                                                               \
        int i = mBase + t;                                                                          \
        long long p = (i < segEnd) ? sortedPairs[i] : -1;                                           \
        rowOf[t] = (p < 0) ? -1 : (rowPerPair ? p : p / topk);                                      \
      }                                                                                             \
      __syncthreads();                                                                              \
      for (unsigned long kBase = 0; kBase < K; kBase += TILE) {                                     \
        for (int e = t; e < TILE * TILE; e += 128) {                                                \
          int r = e / TILE, c = e % TILE;                                                           \
          long long row = rowOf[r];                                                                 \
          unsigned long k = kBase + c;                                                              \
          As[r][c] = (row >= 0 && k < K) ? TO(input[(size_t)row * K + k]) : 0.f;                   \
          unsigned long n = nBase + r;                                                              \
          Bs[r][c] = (n < N && k < K) ? TO(W[(size_t)n * K + k]) : 0.f;                             \
        }                                                                                           \
        __syncthreads();                                                                            \
        for (int kk = 0; kk < TILE; kk++) {                                                         \
          float b0 = Bs[tx * 2][kk], b1 = Bs[tx * 2 + 1][kk];                                       \
          for (int r = 0; r < 4; r++) {                                                             \
            float a = As[ty * 4 + r][kk];                                                           \
            acc[r][0] += a * b0;                                                                    \
            acc[r][1] += a * b1;                                                                    \
          }                                                                                         \
        }                                                                                           \
        __syncthreads();                                                                            \
      }                                                                                             \
      for (int r = 0; r < 4; r++) {                                                                 \
        int i = mBase + ty * 4 + r;                                                                 \
        if (i >= segEnd) continue;                                                                  \
        long long p = sortedPairs[i];                                                               \
        if (p < 0 || (unsigned long)p >= sizeM) continue;                                           \
        float w = topkWeights ? topkWeights[p] : 1.f;                                               \
        for (int c = 0; c < 2; c++) {                                                               \
          unsigned long n = nBase + tx * 2 + c;                                                     \
          if (n < N) output[(size_t)p * N + n] = FROM(acc[r][c] * w);                               \
        }                                                                                           \
      }                                                                                             \
      __syncthreads();                                                                              \
    }                                                                                               \
  }
MOE_GEMM(moe_gemm_f32, float, IDENT, IDENT)
MOE_GEMM(moe_gemm_bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16)
MOE_GEMM(moe_gemm_f16, __half, __half2float, __float2half)
