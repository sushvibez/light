// Quantized-weight matmul: F32 activations × GGUF block-quantized weights.
//
//   out[m, n] = Σ_k x[m, k] * dequant(w)[n, k]        x: (M, K) F32 row-major
//                                                      w: (N, K) quantized, row-major blocks
//                                                      out: (M, N) F32
//
// One warp computes one output column n for a tile of MT=4 rows of x. Lanes
// stride over the K-blocks of weight row n, decode the block once (scales
// hoisted per block / per 32-element sub-block) and FMA against the 4 x rows,
// then warp-reduce. Weights are read exactly once per M-tile, which is what
// matters for decode (M=1): the kernel is bound by weight bandwidth.
//
// Block layouts match kernels/cuda/src/dequant.cu and quant/*.go; the decode
// code is duplicated here rather than shared so each kernel can hoist the
// per-block header loads out of the element loop.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define MT 4

__device__ __forceinline__ float half_le(const unsigned char *p) {
  unsigned short u = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
  __half_raw hr;
  hr.x = u;
  return __half2float((__half)hr);
}

__device__ __forceinline__ void warp_reduce_store(float acc[MT], float *out,
                                                  int m0, int M, int N, int n, int lane) {
#pragma unroll
  for (int mm = 0; mm < MT; mm++) {
    float v = acc[mm];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0 && m0 + mm < M) out[(size_t)(m0 + mm) * N + n] = v;
  }
}

// Accumulate w * x[m0..m0+MT, k] for one decoded weight element.
#define ACC_ELEM(wv, kk)                                                   \
  {                                                                        \
    _Pragma("unroll") for (int mm = 0; mm < MT; mm++) {                    \
      if (m0 + mm < M) acc[mm] += (wv) * x[(size_t)(m0 + mm) * K + (kk)]; \
    }                                                                      \
  }

#define KERNEL_PROLOGUE(BE, BB)                                            \
  int lane = threadIdx.x & 31;                                             \
  int n = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;                    \
  if (n >= N) return;                                                      \
  int m0 = blockIdx.y * MT;                                                \
  int nblocks = K / (BE);                                                  \
  const unsigned char *row = w + (size_t)n * nblocks * (BB);               \
  float acc[MT];                                                           \
  _Pragma("unroll") for (int mm = 0; mm < MT; mm++) acc[mm] = 0.f;

// ---- Q4_0: 32 elems / 18 B: [f16 d | qs[16]] ----
extern "C" __global__ void matmul_q4_0_f32(const float *x, const unsigned char *w, float *out,
                                           int M, int N, int K) {
  KERNEL_PROLOGUE(32, 18)
  for (int b = lane; b < nblocks; b += 32) {
    const unsigned char *blk = row + b * 18;
    float d = half_le(blk);
    int k0 = b * 32;
    for (int j = 0; j < 16; j++) {
      unsigned char q = blk[2 + j];
      ACC_ELEM(d * (float)((int)(q & 0x0F) - 8), k0 + j)
      ACC_ELEM(d * (float)((int)(q >> 4) - 8), k0 + j + 16)
    }
  }
  warp_reduce_store(acc, out, m0, M, N, n, lane);
}

// ---- Q5_0: 32 elems / 22 B: [f16 d | u32 qh | qs[16]] ----
extern "C" __global__ void matmul_q5_0_f32(const float *x, const unsigned char *w, float *out,
                                           int M, int N, int K) {
  KERNEL_PROLOGUE(32, 22)
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
  }
  warp_reduce_store(acc, out, m0, M, N, n, lane);
}

// ---- Q8_0: 32 elems / 34 B: [f16 d | int8 qs[32]] ----
extern "C" __global__ void matmul_q8_0_f32(const float *x, const unsigned char *w, float *out,
                                           int M, int N, int K) {
  KERNEL_PROLOGUE(32, 34)
  for (int b = lane; b < nblocks; b += 32) {
    const unsigned char *blk = row + b * 34;
    float d = half_le(blk);
    int k0 = b * 32;
    for (int j = 0; j < 32; j++) {
      ACC_ELEM(d * (float)((signed char)blk[2 + j]), k0 + j)
    }
  }
  warp_reduce_store(acc, out, m0, M, N, n, lane);
}

// ---- Q8_0 decode path (M == 1): out[n] = Σ_k x[k]·dequant(w[n, k]) ----
// One warp per output row; each lane owns every 32nd block. A 34-byte
// block is fetched as nine aligned 32-bit words and realigned with funnel
// shifts (rows start 4-byte aligned when K % 64 == 0, which the launcher
// checks), so every weight byte crosses the memory system once; x is read
// as float4. The generic kernel above issues one byte load per weight and
// is L2-bound on a 7B decode step.
extern "C" __global__ void matmul_q8_0_f32_mv(const float *__restrict__ x,
                                              const unsigned char *__restrict__ w,
                                              float *__restrict__ out, int N, int K) {
  int lane = threadIdx.x & 31;
  int n = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (n >= N) return;
  int nblocks = K >> 5;
  const unsigned int *row = (const unsigned int *)(w + (size_t)n * nblocks * 34);
  float acc = 0.f;
  for (int b = lane; b < nblocks; b += 32) {
    unsigned int byteOff = (unsigned int)b * 34u;
    const unsigned int *p = row + (byteOff >> 2);
    bool odd = (byteOff & 2u) != 0;  // block starts at byte 2 of its first word
    unsigned int wd[9];
#pragma unroll
    for (int i = 0; i < 9; i++) wd[i] = __ldg(p + i);
    unsigned short dbits = odd ? (unsigned short)(wd[0] >> 16) : (unsigned short)(wd[0] & 0xffffu);
    __half_raw hr;
    hr.x = dbits;
    float d = __half2float((__half)hr);
    const float4 *xv = (const float4 *)(x + b * 32);
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      unsigned int q = odd ? wd[j + 1] : __funnelshift_r(wd[j], wd[j + 1], 16);
      float4 xx = xv[j];
      float c0 = (float)((int)(q << 24) >> 24);
      float c1 = (float)((int)(q << 16) >> 24);
      float c2 = (float)((int)(q << 8) >> 24);
      float c3 = (float)((int)q >> 24);
      sum += c0 * xx.x + c1 * xx.y + c2 * xx.z + c3 * xx.w;
    }
    acc += d * sum;
  }
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) out[n] = acc;
}

__device__ __forceinline__ void q4k_scale_min_mm(int j, const unsigned char *scales,
                                                 unsigned &sc, unsigned &m) {
  if (j < 4) {
    sc = (unsigned)(scales[j] & 63);
    m = (unsigned)(scales[j + 4] & 63);
  } else {
    sc = (unsigned)((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4));
    m = (unsigned)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
  }
}

// ---- Q4_K: 256 elems / 144 B: [f16 d | f16 dmin | scales[12] | qs[128]] ----
// Element idx: p = idx/64, r = idx%64, sub j = 2p + (r>>5), l = r&31;
// nibble low of qs[p*32+l] when r<32 else high. y = d*sc_j*q - dmin*m_j.
extern "C" __global__ void matmul_q4_k_f32(const float *x, const unsigned char *w, float *out,
                                           int M, int N, int K) {
  KERNEL_PROLOGUE(256, 144)
  for (int b = lane; b < nblocks; b += 32) {
    const unsigned char *blk = row + b * 144;
    float d = half_le(blk);
    float dmin = half_le(blk + 2);
    const unsigned char *scales = blk + 4;
    const unsigned char *qs = blk + 16;
    int k0 = b * 256;
    for (int p = 0; p < 4; p++) {
      unsigned sc0, m0_, sc1, m1_;
      q4k_scale_min_mm(2 * p, scales, sc0, m0_);
      q4k_scale_min_mm(2 * p + 1, scales, sc1, m1_);
      float dl0 = d * (float)sc0, ml0 = dmin * (float)m0_;
      float dl1 = d * (float)sc1, ml1 = dmin * (float)m1_;
      for (int l = 0; l < 32; l++) {
        unsigned char q = qs[p * 32 + l];
        ACC_ELEM(dl0 * (float)(q & 0x0F) - ml0, k0 + p * 64 + l)
        ACC_ELEM(dl1 * (float)(q >> 4) - ml1, k0 + p * 64 + 32 + l)
      }
    }
  }
  warp_reduce_store(acc, out, m0, M, N, n, lane);
}

// ---- Q5_K: 256 elems / 176 B: [f16 d | f16 dmin | scales[12] | qh[32] | qs[128]] ----
extern "C" __global__ void matmul_q5_k_f32(const float *x, const unsigned char *w, float *out,
                                           int M, int N, int K) {
  KERNEL_PROLOGUE(256, 176)
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
      q4k_scale_min_mm(2 * p, scales, sc0, m0_);
      q4k_scale_min_mm(2 * p + 1, scales, sc1, m1_);
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
  }
  warp_reduce_store(acc, out, m0, M, N, n, lane);
}

// ---- Q6_K: 256 elems / 210 B: [ql[128] | qh[64] | int8 scales[16] | f16 d] ----
// Two halves of 128; within a half, groups of 32 elements:
//   g0: ql[l]    &0xF | (qh[l]>>0 &3)<<4, scale[half*8 + 0 + l/16]
//   g1: ql[l+32] &0xF | (qh[l]>>2 &3)<<4, scale[half*8 + 2 + l/16]
//   g2: ql[l]    >>4  | (qh[l]>>4 &3)<<4, scale[half*8 + 4 + l/16]
//   g3: ql[l+32] >>4  | (qh[l]>>6 &3)<<4, scale[half*8 + 6 + l/16]
// y = d * scale * (q - 32)
extern "C" __global__ void matmul_q6_k_f32(const float *x, const unsigned char *w, float *out,
                                           int M, int N, int K) {
  KERNEL_PROLOGUE(256, 210)
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
  }
  warp_reduce_store(acc, out, m0, M, N, n, lane);
}
