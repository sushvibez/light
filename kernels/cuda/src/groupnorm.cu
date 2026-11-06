// GroupNorm forward kernels (F32 and BF16). One CUDA block per (n, group)
// pair; threads in the block cooperatively reduce over the (C/G * H*W)
// elements that fall in that group, then write the normalised + affined
// output. Stable Diffusion's UNet/VAE use GN(num_groups=32) everywhere.
//
// Tensors are (N, C, H, W) row-major. `gamma` and `beta` are length-C
// per-channel learnable affine parameters (kept in F32 even on the BF16
// path — they're tiny and matter for stability).

#include <cuda_runtime.h>
#include <cuda_bf16.h>

static __device__ __forceinline__ float bf2f(__nv_bfloat16 x) { return __bfloat162float(x); }
static __device__ __forceinline__ __nv_bfloat16 f2bf(float x) { return __float2bfloat16(x); }

// groupnorm_f32: one block per (n, g). Shared memory = 2 * blockDim.x floats
// for parallel reductions of sum and sum-of-squares.
extern "C" __global__ void groupnorm_f32(
    const float *input, const float *gamma, const float *beta, float *out,
    unsigned long N, unsigned long C, unsigned long G,
    unsigned long HW, float eps) {
  unsigned long ng = blockIdx.x;
  unsigned long n = ng / G;
  unsigned long g = ng % G;
  unsigned long Cg = C / G;
  unsigned long M = Cg * HW;
  int tid = threadIdx.x;
  int bs = blockDim.x;

  extern __shared__ float sdata_gn[];
  float *sSum = sdata_gn;
  float *sSumSq = sdata_gn + bs;

  float sum = 0.0f, sumSq = 0.0f;
  for (unsigned long i = tid; i < M; i += bs) {
    unsigned long c_in_g = i / HW;
    unsigned long k = i % HW;
    unsigned long c = g * Cg + c_in_g;
    float v = input[(n * C + c) * HW + k];
    sum += v;
    sumSq += v * v;
  }
  sSum[tid] = sum;
  sSumSq[tid] = sumSq;
  __syncthreads();
  for (int s = bs >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      sSum[tid]   += sSum[tid + s];
      sSumSq[tid] += sSumSq[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    float invM = 1.0f / (float)M;
    float m = sSum[0] * invM;
    float var = sSumSq[0] * invM - m * m;
    if (var < 0.0f) var = 0.0f;
    sSum[0] = m;
    sSumSq[0] = rsqrtf(var + eps);
  }
  __syncthreads();
  float m = sSum[0];
  float invSigma = sSumSq[0];

  for (unsigned long i = tid; i < M; i += bs) {
    unsigned long c_in_g = i / HW;
    unsigned long k = i % HW;
    unsigned long c = g * Cg + c_in_g;
    unsigned long idx = (n * C + c) * HW + k;
    float xh = (input[idx] - m) * invSigma;
    out[idx] = xh * gamma[c] + beta[c];
  }
}

// groupnorm_bf16: BF16 input/weight/beta/output (same dtype throughout).
// Reductions still happen in F32 inside the kernel for variance stability —
// the BF16 storage of gamma/beta only loses ~1 mantissa bit on a small
// per-channel constant, which is well below the noise floor for diffusion
// inference. Going uniform here removes the F32-weight runtime cast and
// the dtype dance in nn.Module land.
extern "C" __global__ void groupnorm_bf16(
    const __nv_bfloat16 *input, const __nv_bfloat16 *gamma, const __nv_bfloat16 *beta,
    __nv_bfloat16 *out,
    unsigned long N, unsigned long C, unsigned long G,
    unsigned long HW, float eps) {
  unsigned long ng = blockIdx.x;
  unsigned long n = ng / G;
  unsigned long g = ng % G;
  unsigned long Cg = C / G;
  unsigned long M = Cg * HW;
  int tid = threadIdx.x;
  int bs = blockDim.x;

  extern __shared__ float sdata_gnb[];
  float *sSum = sdata_gnb;
  float *sSumSq = sdata_gnb + bs;

  float sum = 0.0f, sumSq = 0.0f;
  for (unsigned long i = tid; i < M; i += bs) {
    unsigned long c_in_g = i / HW;
    unsigned long k = i % HW;
    unsigned long c = g * Cg + c_in_g;
    float v = bf2f(input[(n * C + c) * HW + k]);
    sum += v;
    sumSq += v * v;
  }
  sSum[tid] = sum;
  sSumSq[tid] = sumSq;
  __syncthreads();
  for (int s = bs >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      sSum[tid]   += sSum[tid + s];
      sSumSq[tid] += sSumSq[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    float invM = 1.0f / (float)M;
    float m = sSum[0] * invM;
    float var = sSumSq[0] * invM - m * m;
    if (var < 0.0f) var = 0.0f;
    sSum[0] = m;
    sSumSq[0] = rsqrtf(var + eps);
  }
  __syncthreads();
  float m = sSum[0];
  float invSigma = sSumSq[0];

  for (unsigned long i = tid; i < M; i += bs) {
    unsigned long c_in_g = i / HW;
    unsigned long k = i % HW;
    unsigned long c = g * Cg + c_in_g;
    unsigned long idx = (n * C + c) * HW + k;
    float xh = (bf2f(input[idx]) - m) * invSigma;
    out[idx] = f2bf(xh * bf2f(gamma[c]) + bf2f(beta[c]));
  }
}
