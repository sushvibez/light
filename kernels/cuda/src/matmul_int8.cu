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

// ---- W8A8: int8 activations × int8 weights through cuBLAS IMMA ----
//
// quantize_rows_i8_f32: symmetric per-row absmax quantisation of x (M, K)
// F32 → xq (M, K) int8, xs (M,) F32 with x ≈ xq · xs. One 256-thread block
// per row: block max-reduction, then every lane rounds its strided columns.
//
// dequant_i32_rowcol_f32: out[m, n] = acc[m, n] · xs[m] · ws[n] — the
// epilogue after cublasGemmEx(int8, int8 → int32).

extern "C" __global__ void quantize_rows_i8_f32(const float *__restrict__ x, signed char *__restrict__ xq,
                                                float *__restrict__ xs, int M, int K) {
  __shared__ float red[256];
  int m = blockIdx.x;
  if (m >= M) return;
  const float *row = x + (long long)m * K;
  float amax = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) amax = fmaxf(amax, fabsf(row[k]));
  red[threadIdx.x] = amax;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  float scale = red[0] / 127.f;
  float inv = scale > 0.f ? 1.f / scale : 0.f;
  if (threadIdx.x == 0) xs[m] = scale;
  signed char *qrow = xq + (long long)m * K;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float v = rintf(row[k] * inv);
    v = fminf(127.f, fmaxf(-127.f, v));
    qrow[k] = (signed char)(int)v;
  }
}

extern "C" __global__ void dequant_i32_rowcol_f32(const int *__restrict__ acc, const float *__restrict__ xs,
                                                  const float *__restrict__ ws, float *__restrict__ out,
                                                  long long M, long long N) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * N) return;
  long long m = i / N, n = i % N;
  out[i] = (float)acc[i] * xs[m] * ws[n];
}
