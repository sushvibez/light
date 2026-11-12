// Decode GEMVs: one row of BF16 activations against several nn.Linear
// weights ((N, K) row-major, BF16) in one launch (ops.LinearMulti), and
// the gated pair silu / gelu(x·Wgᵀ) · (x·Wuᵀ) (ops.GatedLinear). A warp
// computes one output row from 16-byte loads; x sits in shared memory as
// F32; sums are F32 and rounded once on store, as cuBLAS GemmEx does.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define GEMV_MAX_SEG 4
#define GEMV_WARPS 8

struct GemvSeg {
  const __nv_bfloat16 *w;
  void *y;
  int n;
  int pad;
};
struct GemvSegs {
  GemvSeg s[GEMV_MAX_SEG];
};

__device__ __forceinline__ void gemv_load_x(const __nv_bfloat16 *x, float *xs, int K) {
  for (int i = threadIdx.x * 2; i < K; i += blockDim.x * 2) {
    const float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(x + i));
    xs[i] = v.x;
    xs[i + 1] = v.y;
  }
}

// 8 BF16 weights (one 16-byte load) against 8 F32 activations from
// shared memory (two float4 loads): BF16 → F32 is a shift into the high
// half of the word.
__device__ __forceinline__ float gemv_dot8(const uint4 u, const float *xs8) {
  const float4 x0 = *reinterpret_cast<const float4 *>(xs8);
  const float4 x1 = *reinterpret_cast<const float4 *>(xs8 + 4);
  float acc = __uint_as_float(u.x << 16) * x0.x;
  acc += __uint_as_float(u.x & 0xffff0000u) * x0.y;
  acc += __uint_as_float(u.y << 16) * x0.z;
  acc += __uint_as_float(u.y & 0xffff0000u) * x0.w;
  acc += __uint_as_float(u.z << 16) * x1.x;
  acc += __uint_as_float(u.z & 0xffff0000u) * x1.y;
  acc += __uint_as_float(u.w << 16) * x1.z;
  acc += __uint_as_float(u.w & 0xffff0000u) * x1.w;
  return acc;
}

// Lane-partial dot of one weight row with xs; the warp reduce follows.
__device__ __forceinline__ float gemv_row(const __nv_bfloat16 *wr, const float *xs, int K, int lane) {
  float acc = 0.f;
#pragma unroll 4
  for (int c = lane * 8; c < K; c += 256) {
    acc += gemv_dot8(__ldg(reinterpret_cast<const uint4 *>(wr + c)), xs + c);
  }
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  return acc;
}

// Grid ceil(ΣNᵢ / 8), 256 threads, shared K floats. Row r of the
// concatenated outputs lands in segment i at r − Σ_{j<i} Nⱼ.
extern "C" __global__ void gemv_bf16_multi(const __nv_bfloat16 *x, GemvSegs segs, int nseg, int K, int outF32) {
  extern __shared__ float xs[];
  gemv_load_x(x, xs, K);
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int row = blockIdx.x * GEMV_WARPS + warp;
  int seg = 0;
  while (seg < nseg && row >= segs.s[seg].n) {
    row -= segs.s[seg].n;
    seg++;
  }
  if (seg >= nseg) return;
  const float acc = gemv_row(segs.s[seg].w + (long long)row * K, xs, K, lane);
  if (lane == 0) {
    if (outF32)
      reinterpret_cast<float *>(segs.s[seg].y)[row] = acc;
    else
      reinterpret_cast<__nv_bfloat16 *>(segs.s[seg].y)[row] = __float2bfloat16(acc);
  }
}

// y[r] = act(x·Wg[r]) · (x·Wu[r]), act = silu (gelu = 0) or erf-GELU, with
// the intermediate products rounded to BF16 first, as the unfused path
// (two BF16 GEMVs, then the activation and the product) leaves them.
extern "C" __global__ void gemv_bf16_gated(const __nv_bfloat16 *x, const __nv_bfloat16 *wg, const __nv_bfloat16 *wu,
                                           __nv_bfloat16 *y, int N, int K, int gelu) {
  extern __shared__ float xs[];
  gemv_load_x(x, xs, K);
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = blockIdx.x * GEMV_WARPS + warp;
  if (row >= N) return;
  // Both rows in one pass so twice the loads are in flight per lane.
  const __nv_bfloat16 *gr = wg + (long long)row * K, *ur = wu + (long long)row * K;
  float g = 0.f, u = 0.f;
#pragma unroll 4
  for (int c = lane * 8; c < K; c += 256) {
    const uint4 a = __ldg(reinterpret_cast<const uint4 *>(gr + c));
    const uint4 b = __ldg(reinterpret_cast<const uint4 *>(ur + c));
    g += gemv_dot8(a, xs + c);
    u += gemv_dot8(b, xs + c);
  }
  for (int o = 16; o > 0; o >>= 1) {
    g += __shfl_xor_sync(0xffffffffu, g, o);
    u += __shfl_xor_sync(0xffffffffu, u, o);
  }
  if (lane == 0) {
    const float gb = __bfloat162float(__float2bfloat16(g));
    const float ub = __bfloat162float(__float2bfloat16(u));
    float a;
    if (gelu)
      a = 0.5f * gb * (1.f + erff(gb * 0.70710678118654752f));
    else
      a = gb / (1.f + expf(-gb));
    a = __bfloat162float(__float2bfloat16(a));
    y[row] = __float2bfloat16(a * ub);
  }
}
