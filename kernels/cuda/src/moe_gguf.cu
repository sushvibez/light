// MoE grouped GEMM over GGUF block-quantized expert weights (2026-10-28):
// the quantized twin of moe.cu, Candle's `moe_gemm_gguf`.
//
//   out[p, n] = w(p) · Σ_k x[row(p), k] · dequant(W[e(p)])[n, k]
//
// x is F32 (rows, K); W is (E, N, K) in one GGUF block format, each expert a
// contiguous (N, K) block matrix; routing (sortedPairs / expertOffsets,
// topkWeights, rowPerPair) exactly as in moe.cu. Output is F32 (M, N),
// one plain store per pair row.
//
// Thread mapping follows matmul_q.cu: one warp per output column n and a
// tile of MT=4 consecutive sorted pairs of one expert; lanes stride over
// the K-blocks of weight row n, decode each block once and FMA it against
// the 4 activation rows, then warp-reduce. Grid: x = N/8 warps, y = pair
// tiles (strided so y ≤ 1024), z = expert. Blocks whose expert has no pairs
// exit immediately. Block decode bodies are those of matmul_q.cu.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define MT 4

__device__ __forceinline__ float half_le(const unsigned char *p) {
  unsigned short u = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
  __half_raw hr;
  hr.x = u;
  return __half2float((__half)hr);
}

#define ACC_ELEM(wv, kk)                                                    \
  {                                                                         \
    _Pragma("unroll") for (int mm = 0; mm < MT; mm++) {                     \
      if (xrow[mm]) acc[mm] += (wv) * xrow[mm][(kk)];                       \
    }                                                                       \
  }

// Per-tile prologue / epilogue shared by every format. The decode loop in
// between reads `row`, `nblocks`, `lane`, `K` and calls ACC_ELEM.
#define MOE_Q_KERNEL(NAME, BE, BB, ...)                                                     \
  extern "C" __global__ void NAME(                                                          \
      const float *x, const unsigned char *w, const long long *sortedPairs,                 \
      const int *expertOffsets, const float *topkWeights, float *out, int topk,             \
      unsigned long sizeM, int N, int K, int rowPerPair) {                                  \
    int lane = threadIdx.x & 31;                                                            \
    int n = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;                                   \
    int e = blockIdx.z;                                                                     \
    int segStart = expertOffsets[e], segEnd = expertOffsets[e + 1];                         \
    if (n >= N || segStart >= segEnd) return;                                               \
    int nblocks = K / (BE);                                                                 \
    const unsigned char *row = w + ((size_t)e * N + n) * (size_t)nblocks * (BB);            \
    for (int m0 = segStart + (int)blockIdx.y * MT; m0 < segEnd; m0 += (int)gridDim.y * MT) { \
      const float *xrow[MT];                                                                \
      long long pair[MT];                                                                   \
      _Pragma("unroll") for (int mm = 0; mm < MT; mm++) {                                   \
        int i = m0 + mm;                                                                    \
        long long p = (i < segEnd) ? sortedPairs[i] : -1;                                   \
        if (p < 0 || (unsigned long)p >= sizeM) p = -1;                                     \
        pair[mm] = p;                                                                       \
        xrow[mm] = (p < 0) ? (const float *)0 : x + (size_t)(rowPerPair ? p : p / topk) * K; \
      }                                                                                     \
      float acc[MT];                                                                        \
      _Pragma("unroll") for (int mm = 0; mm < MT; mm++) acc[mm] = 0.f;                      \
      __VA_ARGS__                                                                           \
      _Pragma("unroll") for (int mm = 0; mm < MT; mm++) {                                   \
        float v = acc[mm];                                                                  \
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);  \
        if (lane == 0 && pair[mm] >= 0) {                                                   \
          float s = topkWeights ? topkWeights[pair[mm]] : 1.f;                              \
          out[(size_t)pair[mm] * N + n] = v * s;                                            \
        }                                                                                   \
      }                                                                                     \
    }                                                                                       \
  }

// ---- Q4_0: 32 elems / 18 B ----
MOE_Q_KERNEL(moe_gemm_q4_0_f32, 32, 18,
  for (int b = lane; b < nblocks; b += 32) {
    const unsigned char *blk = row + b * 18;
    float d = half_le(blk);
    int k0 = b * 32;
    for (int j = 0; j < 16; j++) {
      unsigned char q = blk[2 + j];
      ACC_ELEM(d * (float)((int)(q & 0x0F) - 8), k0 + j)
      ACC_ELEM(d * (float)((int)(q >> 4) - 8), k0 + j + 16)
    }
  })

// ---- Q5_0: 32 elems / 22 B ----
MOE_Q_KERNEL(moe_gemm_q5_0_f32, 32, 22,
  for (int b = lane; b < nblocks; b += 32) {
    const unsigned char *blk = row + b * 22;
    float d = half_le(blk);
    unsigned qh = (unsigned)blk[2] | ((unsigned)blk[3] << 8) | ((unsigned)blk[4] << 16) | ((unsigned)blk[5] << 24);
    int k0 = b * 32;
    for (int j = 0; j < 16; j++) {
      unsigned char q = blk[6 + j];
      int lo = (int)(q & 0x0F) | (int)(((qh >> j) & 1u) << 4);
      int hi = (int)(q >> 4) | (int)(((qh >> (j + 16)) & 1u) << 4);
      ACC_ELEM(d * (float)(lo - 16), k0 + j)
      ACC_ELEM(d * (float)(hi - 16), k0 + j + 16)
    }
  })

// ---- Q8_0: 32 elems / 34 B ----
MOE_Q_KERNEL(moe_gemm_q8_0_f32, 32, 34,
  for (int b = lane; b < nblocks; b += 32) {
    const unsigned char *blk = row + b * 34;
    float d = half_le(blk);
    int k0 = b * 32;
    for (int j = 0; j < 32; j++) {
      ACC_ELEM(d * (float)((signed char)blk[2 + j]), k0 + j)
    }
  })

__device__ __forceinline__ void q4k_scale_min_moe(int j, const unsigned char *scales,
                                                  unsigned &sc, unsigned &m) {
  if (j < 4) {
    sc = (unsigned)(scales[j] & 63);
    m = (unsigned)(scales[j + 4] & 63);
  } else {
    sc = (unsigned)((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4));
    m = (unsigned)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
  }
}

// ---- Q4_K: 256 elems / 144 B ----
MOE_Q_KERNEL(moe_gemm_q4_k_f32, 256, 144,
  for (int b = lane; b < nblocks; b += 32) {
    const unsigned char *blk = row + b * 144;
    float d = half_le(blk);
    float dmin = half_le(blk + 2);
    const unsigned char *scales = blk + 4;
    const unsigned char *qs = blk + 16;
    int k0 = b * 256;
    for (int p = 0; p < 4; p++) {
      unsigned sc0, m0_, sc1, m1_;
      q4k_scale_min_moe(2 * p, scales, sc0, m0_);
      q4k_scale_min_moe(2 * p + 1, scales, sc1, m1_);
      float dl0 = d * (float)sc0, ml0 = dmin * (float)m0_;
      float dl1 = d * (float)sc1, ml1 = dmin * (float)m1_;
      for (int l = 0; l < 32; l++) {
        unsigned char q = qs[p * 32 + l];
        ACC_ELEM(dl0 * (float)(q & 0x0F) - ml0, k0 + p * 64 + l)
        ACC_ELEM(dl1 * (float)(q >> 4) - ml1, k0 + p * 64 + 32 + l)
      }
    }
  })

// ---- Q5_K: 256 elems / 176 B ----
MOE_Q_KERNEL(moe_gemm_q5_k_f32, 256, 176,
  for (int b = lane; b < nblocks; b += 32) {
    const unsigned char *blk = row + b * 176;
    float d = half_le(blk);
    float dmin = half_le(blk + 2);
    const unsigned char *scales = blk + 4;
    const unsigned char *qh = blk + 16;
    const unsigned char *qs = blk + 48;
    int k0 = b * 256;
    for (int p = 0; p < 4; p++) {
      unsigned sc0, m0_, sc1, m1_;
      q4k_scale_min_moe(2 * p, scales, sc0, m0_);
      q4k_scale_min_moe(2 * p + 1, scales, sc1, m1_);
      float dl0 = d * (float)sc0, ml0 = dmin * (float)m0_;
      float dl1 = d * (float)sc1, ml1 = dmin * (float)m1_;
      unsigned bit0 = 1u << (2 * p), bit1 = 1u << (2 * p + 1);
      for (int l = 0; l < 32; l++) {
        unsigned char q = qs[p * 32 + l];
        unsigned h = qh[l];
        unsigned lo = (unsigned)(q & 0x0F) | ((h & bit0) ? 0x10u : 0u);
        unsigned hi = (unsigned)(q >> 4) | ((h & bit1) ? 0x10u : 0u);
        ACC_ELEM(dl0 * (float)lo - ml0, k0 + p * 64 + l)
        ACC_ELEM(dl1 * (float)hi - ml1, k0 + p * 64 + 32 + l)
      }
    }
  })

// ---- Q6_K: 256 elems / 210 B ----
MOE_Q_KERNEL(moe_gemm_q6_k_f32, 256, 210,
  for (int b = lane; b < nblocks; b += 32) {
    const unsigned char *blk = row + b * 210;
    float d = half_le(blk + 208);
    const signed char *sc = (const signed char *)(blk + 192);
    int k0 = b * 256;
    for (int half = 0; half < 2; half++) {
      const unsigned char *ql = blk + half * 64;
      const unsigned char *qh = blk + 128 + half * 32;
      const signed char *s = sc + half * 8;
      int kb = k0 + half * 128;
      for (int l = 0; l < 32; l++) {
        int is = l >> 4;
        unsigned h = qh[l];
        int q0 = (int)(ql[l] & 0x0F) | (int)(((h >> 0) & 3u) << 4);
        int q1 = (int)(ql[l + 32] & 0x0F) | (int)(((h >> 2) & 3u) << 4);
        int q2 = (int)(ql[l] >> 4) | (int)(((h >> 4) & 3u) << 4);
        int q3 = (int)(ql[l + 32] >> 4) | (int)(((h >> 6) & 3u) << 4);
        ACC_ELEM(d * (float)s[0 + is] * (float)(q0 - 32), kb + l)
        ACC_ELEM(d * (float)s[2 + is] * (float)(q1 - 32), kb + 32 + l)
        ACC_ELEM(d * (float)s[4 + is] * (float)(q2 - 32), kb + 64 + l)
        ACC_ELEM(d * (float)s[6 + is] * (float)(q3 - 32), kb + 96 + l)
      }
    }
  })
