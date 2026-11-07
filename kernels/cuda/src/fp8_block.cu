// Block-scaled ("fine-grained") FP8 weights: Qwen3.5 / Qwen3 / DeepSeek-V3
// FP8 checkpoints store each projection as F8E4M3 codes w (N, K) plus an F32
// scale per (bR × bC) block, W[n, k] = w[n, k] · s[n / bR, k / bC].
//
// E4M3 → float without FP8 hardware (sm_80): the code's low 7 bits shifted
// left by 7 are an F16 bit pattern (4-bit exponent into the low exponent
// bits, 3-bit mantissa into the top mantissa bits) whose value is exactly
// the E4M3 value · 2⁻⁸, subnormals included (E4M3's single NaN, 0x7F, is
// not decoded, and no weight uses it). The kernels decode with that and
// fold the 2⁸ into the block scale. Two codes become one half2 per shift.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Four codes in one little-endian word → (c0, c2) and (c1, c3) as half2,
// each value · 2⁻⁸.
__device__ __forceinline__ void fp8x4_to_half2x2(unsigned int v, __half2 &h02, __half2 &h13) {
  unsigned int lo = ((v & 0x007F007Fu) << 7) | ((v & 0x00800080u) << 8);
  unsigned int hi = ((v & 0x7F007F00u) >> 1) | (v & 0x80008000u);
  h02 = *reinterpret_cast<__half2 *>(&lo);
  h13 = *reinterpret_cast<__half2 *>(&hi);
}

__device__ __forceinline__ float load_act(const float *x, long long i) { return x[i]; }
__device__ __forceinline__ float load_act(const __nv_bfloat16 *x, long long i) { return __bfloat162float(x[i]); }
__device__ __forceinline__ void store_act(float *y, long long i, float v) { y[i] = v; }
__device__ __forceinline__ void store_act(__nv_bfloat16 *y, long long i, float v) { y[i] = __float2bfloat16_rn(v); }

// ---------------------------------------------------------------------------
// Dequantize to the activation dtype (prefill path, followed by a GEMM).
// Thread i expands codes 4i .. 4i+3 (one row, one block: C % 4 == 0 and
// bC % 4 == 0). out = (w · s) in F32, rounded once, as transformers'
// Fp8Dequantize does.
template <typename T>
__device__ void fp8_block_dequant(const unsigned int *w, const float *s, T *out,
                                  int R, int C, int bR, int bC) {
  long long n4 = (long long)R * C / 4;
  int gc = C / bC;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
       i += (long long)gridDim.x * blockDim.x) {
    long long e = i * 4;
    int row = (int)(e / C), col = (int)(e % C);
    float sc = s[(row / bR) * gc + col / bC] * 256.0f;
    __half2 h02, h13;
    fp8x4_to_half2x2(w[i], h02, h13);
    float2 a = __half22float2(h02), b = __half22float2(h13);
    store_act(out, e + 0, a.x * sc);
    store_act(out, e + 1, b.x * sc);
    store_act(out, e + 2, a.y * sc);
    store_act(out, e + 3, b.y * sc);
  }
}

extern "C" __global__ void fp8_block_dequant_f32(const unsigned int *w, const float *s, float *out,
                                                 int R, int C, int bR, int bC) {
  fp8_block_dequant<float>(w, s, out, R, C, bR, bC);
}

extern "C" __global__ void fp8_block_dequant_bf16(const unsigned int *w, const float *s, __nv_bfloat16 *out,
                                                  int R, int C, int bR, int bC) {
  fp8_block_dequant<__nv_bfloat16>(w, s, out, R, C, bR, bC);
}

// ---------------------------------------------------------------------------
// y (M, N) = x (M, K) · Wᵀ for small M (decode): one warp per output row n,
// FP8_GEMV_WARPS rows per block. Each lane reads 16 codes (one uint4) per
// step, so K % 16 == 0 and bC % 16 == 0 (one scale per 16-code chunk); the
// M ≤ FP8_GEMV_MAXM activation rows are read from global memory (L1 / L2
// resident: a few K-element rows). F32 accumulation, warp-shuffle reduce.
#define FP8_GEMV_WARPS 8
#define FP8_GEMV_MAXM 8

template <typename T>
__device__ void fp8_block_gemv(const T *x, const uint4 *w, const float *s, T *y,
                               int M, int N, int K, int bR, int bC) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int n = blockIdx.x * FP8_GEMV_WARPS + warp;
  if (n >= N) return;
  int chunks = K / 16, gc = K / bC;
  const uint4 *wr = w + (long long)n * chunks;
  const float *sr = s + (long long)(n / bR) * gc;
  float acc[FP8_GEMV_MAXM];
#pragma unroll
  for (int m = 0; m < FP8_GEMV_MAXM; m++) acc[m] = 0.f;
  for (int c = lane; c < chunks; c += 32) {
    uint4 q = wr[c];
    int k0 = c * 16;
    float sc = sr[k0 / bC] * 256.0f;
    unsigned int words[4] = {q.x, q.y, q.z, q.w};
    float wv[16];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      __half2 h02, h13;
      fp8x4_to_half2x2(words[j], h02, h13);
      float2 a = __half22float2(h02), b = __half22float2(h13);
      wv[4 * j + 0] = a.x * sc;
      wv[4 * j + 1] = b.x * sc;
      wv[4 * j + 2] = a.y * sc;
      wv[4 * j + 3] = b.y * sc;
    }
#pragma unroll
    for (int m = 0; m < FP8_GEMV_MAXM; m++) {
      if (m < M) {
        const T *xr = x + (long long)m * K + k0;
        float a = 0.f;
#pragma unroll
        for (int j = 0; j < 16; j++) a += wv[j] * load_act(xr, j);
        acc[m] += a;
      }
    }
  }
#pragma unroll
  for (int m = 0; m < FP8_GEMV_MAXM; m++) {
    float v = acc[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0 && m < M) store_act(y, (long long)m * N + n, v);
  }
}

extern "C" __global__ void __launch_bounds__(FP8_GEMV_WARPS * 32)
fp8_block_gemv_f32(const float *x, const uint4 *w, const float *s, float *y, int M, int N, int K, int bR, int bC) {
  fp8_block_gemv<float>(x, w, s, y, M, N, K, bR, bC);
}

extern "C" __global__ void __launch_bounds__(FP8_GEMV_WARPS * 32)
fp8_block_gemv_bf16(const __nv_bfloat16 *x, const uint4 *w, const float *s, __nv_bfloat16 *y,
                    int M, int N, int K, int bR, int bC) {
  fp8_block_gemv<__nv_bfloat16>(x, w, s, y, M, N, K, bR, bC);
}
