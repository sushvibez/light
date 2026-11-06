// BatchNorm2d forward + backward kernels.
//
// Tensors are (N, C, H, W) row-major. Reductions are per-channel: one CUDA
// block per channel, threads cooperatively reduce over the N*H*W positions
// that share the channel. Shared-memory tree reduction matches the usual
// block-reduction template (power-of-two blockDim.x).

#include <cuda_runtime.h>

// Pass 1: compute per-channel mean, variance, and sigma = sqrt(var + eps).
// blockDim.x = 256; gridDim.x = C; shared memory = 2 * blockDim.x * sizeof(float).
extern "C" __global__ void batchnorm2d_stats_f32(
    const float *input, float *mean_out, float *var_out, float *sigma_out,
    float eps,
    unsigned long N, unsigned long C, unsigned long HW) {
  unsigned c = blockIdx.x;
  if (c >= C) return;
  unsigned tid = threadIdx.x;
  unsigned long M = N * HW;
  extern __shared__ float sdata[];
  float *sSum = sdata;
  float *sSumSq = sdata + blockDim.x;
  float sum = 0.0f, sumSq = 0.0f;
  for (unsigned long i = tid; i < M; i += blockDim.x) {
    unsigned long n = i / HW;
    unsigned long k = i - n * HW;
    float v = input[(n * C + c) * HW + k];
    sum += v;
    sumSq += v * v;
  }
  sSum[tid] = sum;
  sSumSq[tid] = sumSq;
  __syncthreads();
  for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      sSum[tid]   += sSum[tid + s];
      sSumSq[tid] += sSumSq[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    float invM = 1.0f / (float)M;
    float m = sSum[0] * invM;
    float v = sSumSq[0] * invM - m * m;
    if (v < 0) v = 0; // numerical floor
    mean_out[c]  = m;
    var_out[c]   = v;
    sigma_out[c] = sqrtf(v + eps);
  }
}

// Pass 2: apply (x - mean) / sigma * gamma + beta, also writing x_hat.
// One thread per output element. sigma is computed once per channel in pass 1.
extern "C" __global__ void batchnorm2d_apply_f32(
    const float *input, const float *mean, const float *sigma,
    const float *gamma, const float *beta,
    float *out, float *xHat,
    unsigned long N, unsigned long C, unsigned long HW) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = N * C * HW;
  if (i >= total) return;
  unsigned long k = i % HW;
  unsigned long c = (i / HW) % C;
  unsigned long n = i / (C * HW);
  float m = mean[c];
  float s = sigma[c];
  float x = input[(n * C + c) * HW + k];
  float xh = (x - m) / s;
  xHat[i] = xh;
  out[i] = xh * gamma[c] + beta[c];
}

// Pass 1 backward: per-channel sum(g) and sum(g * x_hat).
extern "C" __global__ void batchnorm2d_backward_stats_f32(
    const float *gradOut, const float *xHat,
    float *sumG, float *sumGxHat,
    unsigned long N, unsigned long C, unsigned long HW) {
  unsigned c = blockIdx.x;
  if (c >= C) return;
  unsigned tid = threadIdx.x;
  unsigned long M = N * HW;
  extern __shared__ float sdata[];
  float *sG = sdata;
  float *sGxH = sdata + blockDim.x;
  float g = 0.0f, gxh = 0.0f;
  for (unsigned long i = tid; i < M; i += blockDim.x) {
    unsigned long n = i / HW;
    unsigned long k = i - n * HW;
    unsigned long idx = (n * C + c) * HW + k;
    float gv = gradOut[idx];
    g += gv;
    gxh += gv * xHat[idx];
  }
  sG[tid] = g;
  sGxH[tid] = gxh;
  __syncthreads();
  for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      sG[tid]   += sG[tid + s];
      sGxH[tid] += sGxH[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    sumG[c]    = sG[0];
    sumGxHat[c] = sGxH[0];
  }
}

// Pass 2 backward: apply the standard BN formula per element.
//   grad_input = (g' - mean(g') - x_hat * mean(g' * x_hat)) / sigma
// where g' = g * gamma and the means are over (N, H, W) per channel.
extern "C" __global__ void batchnorm2d_backward_apply_f32(
    const float *gradOut, const float *xHat,
    const float *sigma, const float *gamma,
    const float *sumG, const float *sumGxHat,
    float *gradIn,
    float M,
    unsigned long N, unsigned long C, unsigned long HW) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = N * C * HW;
  if (i >= total) return;
  unsigned long c = (i / HW) % C;
  float gammaC = gamma[c];
  float sigC   = sigma[c];
  float meanGP     = gammaC * sumG[c] / M;
  float meanGPxHat = gammaC * sumGxHat[c] / M;
  float gp = gradOut[i] * gammaC;
  gradIn[i] = (gp - meanGP - xHat[i] * meanGPxHat) / sigC;
}
