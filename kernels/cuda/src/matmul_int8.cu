// Weight-only INT8 matmul with symmetric per-output-channel dequant,
// ported from accelerate's k_matmul_int8_perchan:
//
//   out[m,n] = scales[n] * sum_k a[m,k] * (float)wq[k,n]
//
// a: F32 (M,K); wq: I8 (K,N); scales: F32 (N,); out: F32 (M,N).
// 16x16 thread tiles over (N, M); the int8 weight is widened in-register
// and accumulated in F32.

#include <cuda_runtime.h>

extern "C" __global__ void matmul_int8_perchan_f32(
    const float *a, const signed char *wq, const float *scales, float *out,
    int M, int K, int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;
  float acc = 0.f;
  const float *ar = a + (long)m * K;
  for (int k = 0; k < K; k++) {
    acc += ar[k] * (float)wq[(long)k * N + n];
  }
  out[(long)m * N + n] = acc * scales[n];
}
