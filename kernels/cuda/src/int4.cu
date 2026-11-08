// Group-quantized 4-bit weights (GPTQ, AWQ) in Light's canonical layout
// (quant.Int4): codes U32 (N, K/8) with nibble j of codes[n, c] the code of
// W[n, 8c + j]; scales / zeros F32 (N, K/G); W[n, k] = s[n, k/G] · (code − z[n, k/G]).
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float load_act(const float *x, long long i) { return x[i]; }
__device__ __forceinline__ float load_act(const __nv_bfloat16 *x, long long i) { return __bfloat162float(x[i]); }
__device__ __forceinline__ void store_act(float *y, long long i, float v) { y[i] = v; }
__device__ __forceinline__ void store_act(__nv_bfloat16 *y, long long i, float v) { y[i] = __float2bfloat16_rn(v); }

// ---------------------------------------------------------------------------
// Dequantize (prefill path, followed by a GEMM): thread i expands word i,
// eight weights of one row in one group (G % 8 == 0). Product in F32,
// rounded once to the output dtype.
template <typename T>
__device__ void int4_dequant(const unsigned int *codes, const float *s, const float *z, T *out,
                             int N, int K, int G) {
  long long words = (long long)N * (K / 8);
  int kw = K / 8, ng = K / G;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < words;
       i += (long long)gridDim.x * blockDim.x) {
    int n = (int)(i / kw), c = (int)(i % kw);
    int g = n * ng + (8 * c) / G;
    float sc = s[g], zr = z[g];
    unsigned int w = codes[i];
    long long base = (long long)n * K + 8LL * c;
#pragma unroll
    for (int j = 0; j < 8; j++) store_act(out, base + j, sc * ((float)((w >> (4 * j)) & 0xFu) - zr));
  }
}

extern "C" __global__ void int4_dequant_f32(const unsigned int *codes, const float *s, const float *z, float *out,
                                            int N, int K, int G) {
  int4_dequant<float>(codes, s, z, out, N, K, G);
}

extern "C" __global__ void int4_dequant_bf16(const unsigned int *codes, const float *s, const float *z,
                                             __nv_bfloat16 *out, int N, int K, int G) {
  int4_dequant<__nv_bfloat16>(codes, s, z, out, N, K, G);
}

// ---------------------------------------------------------------------------
// y (M, N) = x (M, K) · Wᵀ for small M (decode): one warp per output row,
// INT4_GEMV_WARPS rows per block. Each lane reads 32 codes (one uint4) per
// step, so K % 32 == 0 and G % 32 == 0 (one scale / zero per step). The
// group term is folded: Σ s·(q − z)·x = s·(Σ q·x − z·Σ x). F32 accumulate,
// warp-shuffle reduce.
#define INT4_GEMV_WARPS 8
#define INT4_GEMV_MAXM 8

template <typename T>
__device__ void int4_gemv(const T *x, const uint4 *codes, const float *s, const float *z, T *y,
                          int M, int N, int K, int G) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int n = blockIdx.x * INT4_GEMV_WARPS + warp;
  if (n >= N) return;
  int steps = K / 32, ng = K / G;
  const uint4 *row = codes + (long long)n * steps;
  const float *sr = s + (long long)n * ng;
  const float *zr = z + (long long)n * ng;
  float acc[INT4_GEMV_MAXM];
#pragma unroll
  for (int m = 0; m < INT4_GEMV_MAXM; m++) acc[m] = 0.f;
  for (int c = lane; c < steps; c += 32) {
    uint4 q = row[c];
    int k0 = c * 32;
    float sc = sr[k0 / G], zp = zr[k0 / G];
    unsigned int words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int m = 0; m < INT4_GEMV_MAXM; m++) {
      if (m < M) {
        const T *xr = x + (long long)m * K + k0;
        float dot = 0.f, sum = 0.f;
#pragma unroll
        for (int w = 0; w < 4; w++) {
#pragma unroll
          for (int j = 0; j < 8; j++) {
            float xv = load_act(xr, 8 * w + j);
            dot += (float)((words[w] >> (4 * j)) & 0xFu) * xv;
            sum += xv;
          }
        }
        acc[m] += sc * (dot - zp * sum);
      }
    }
  }
#pragma unroll
  for (int m = 0; m < INT4_GEMV_MAXM; m++) {
    float v = acc[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0 && m < M) store_act(y, (long long)m * N + n, v);
  }
}

extern "C" __global__ void __launch_bounds__(INT4_GEMV_WARPS * 32)
int4_gemv_f32(const float *x, const uint4 *codes, const float *s, const float *z, float *y, int M, int N, int K, int G) {
  int4_gemv<float>(x, codes, s, z, y, M, N, K, G);
}

extern "C" __global__ void __launch_bounds__(INT4_GEMV_WARPS * 32)
int4_gemv_bf16(const __nv_bfloat16 *x, const uint4 *codes, const float *s, const float *z, __nv_bfloat16 *y,
               int M, int N, int K, int G) {
  int4_gemv<__nv_bfloat16>(x, codes, s, z, y, M, N, K, G);
}
