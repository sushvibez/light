// Basic elementwise CUDA kernels for Light.
//
// Each kernel is an unmangled `extern "C" __global__` so that PTX names map
// directly to the strings our Go loader uses with cuModuleGetFunction.
//
// Phase 2 bootstrap: F32 and F64 only. Additional dtypes land as the op
// coverage matches CPU.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BIN_OP(NAME, T, EXPR)                                                        \
  extern "C" __global__ void NAME(const T *a, const T *b, T *out, unsigned long n) { \
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;          \
    if (i < n) {                                                                     \
      T av = a[i];                                                                   \
      T bv = b[i];                                                                   \
      out[i] = (EXPR);                                                               \
    }                                                                                \
  }

#define UN_OP(NAME, T, EXPR)                                                \
  extern "C" __global__ void NAME(const T *a, T *out, unsigned long n) {    \
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < n) {                                                            \
      T av = a[i];                                                          \
      out[i] = (EXPR);                                                      \
    }                                                                       \
  }

// ------------------- binary -------------------
BIN_OP(add_f32, float, av + bv)
BIN_OP(add_f64, double, av + bv)
BIN_OP(add_i32, int, av + bv)
BIN_OP(add_i64, long long, av + bv)

BIN_OP(sub_f32, float, av - bv)
BIN_OP(sub_f64, double, av - bv)

BIN_OP(mul_f32, float, av *bv)
BIN_OP(mul_f64, double, av *bv)

BIN_OP(div_f32, float, av / bv)
BIN_OP(div_f64, double, av / bv)

// ------------------- unary --------------------
UN_OP(neg_f32, float, -av)
UN_OP(neg_f64, double, -av)

UN_OP(exp_f32, float, expf(av))
UN_OP(exp_f64, double, exp(av))

UN_OP(log_f32, float, logf(av))
UN_OP(log_f64, double, log(av))

UN_OP(sqrt_f32, float, sqrtf(av))
UN_OP(sqrt_f64, double, sqrt(av))

UN_OP(relu_f32, float, av > 0.0f ? av : 0.0f)
UN_OP(relu_f64, double, av > 0.0 ? av : 0.0)

UN_OP(sigmoid_f32, float, 1.0f / (1.0f + expf(-av)))
UN_OP(sigmoid_f64, double, 1.0 / (1.0 + exp(-av)))

UN_OP(tanh_f32, float, tanhf(av))
UN_OP(tanh_f64, double, tanh(av))

// SiLU = x * sigmoid(x)
UN_OP(silu_f32, float, av / (1.0f + expf(-av)))
UN_OP(silu_f64, double, av / (1.0 + exp(-av)))

// ------------------- F16 / BF16 (via float conversion) -------------------
#define BIN_HALF(NAME, T, TO, FROM, OP)                                              \
  extern "C" __global__ void NAME(const T *a, const T *b, T *out, unsigned long n) { \
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;          \
    if (i < n) {                                                                     \
      float av = TO(a[i]);                                                           \
      float bv = TO(b[i]);                                                           \
      out[i] = FROM(OP);                                                             \
    }                                                                                \
  }

#define UN_HALF(NAME, T, TO, FROM, EXPR)                                    \
  extern "C" __global__ void NAME(const T *a, T *out, unsigned long n) {    \
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < n) {                                                            \
      float av = TO(a[i]);                                                  \
      out[i] = FROM(EXPR);                                                  \
    }                                                                       \
  }

BIN_HALF(add_f16, __half, __half2float, __float2half, av + bv)
BIN_HALF(sub_f16, __half, __half2float, __float2half, av - bv)
BIN_HALF(mul_f16, __half, __half2float, __float2half, av *bv)
BIN_HALF(div_f16, __half, __half2float, __float2half, av / bv)
UN_HALF(neg_f16, __half, __half2float, __float2half, -av)
UN_HALF(exp_f16, __half, __half2float, __float2half, expf(av))
UN_HALF(log_f16, __half, __half2float, __float2half, logf(av))
UN_HALF(sqrt_f16, __half, __half2float, __float2half, sqrtf(av))
UN_HALF(relu_f16, __half, __half2float, __float2half, av > 0.0f ? av : 0.0f)
UN_HALF(sigmoid_f16, __half, __half2float, __float2half, 1.0f / (1.0f + expf(-av)))
UN_HALF(tanh_f16, __half, __half2float, __float2half, tanhf(av))
UN_HALF(silu_f16, __half, __half2float, __float2half, av / (1.0f + expf(-av)))

BIN_HALF(add_bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16, av + bv)
BIN_HALF(sub_bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16, av - bv)
BIN_HALF(mul_bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16, av *bv)
BIN_HALF(div_bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16, av / bv)
UN_HALF(neg_bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16, -av)
UN_HALF(exp_bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16, expf(av))
UN_HALF(log_bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16, logf(av))
UN_HALF(sqrt_bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16, sqrtf(av))
UN_HALF(relu_bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16, av > 0.0f ? av : 0.0f)
UN_HALF(sigmoid_bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16, 1.0f / (1.0f + expf(-av)))
UN_HALF(tanh_bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16, tanhf(av))
UN_HALF(silu_bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16, av / (1.0f + expf(-av)))
// Exact (erf) GELU in half precision, computed in F32 (Falcon/BERT MLPs at BF16).
UN_HALF(gelu_erf_bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16, 0.5f * av * (1.0f + erff(av * 0.7071067811865476f)))
UN_HALF(gelu_erf_f16, __half, __half2float, __float2half, 0.5f * av * (1.0f + erff(av * 0.7071067811865476f)))
UN_HALF(gelu_f16, __half, __half2float, __float2half, 0.5f * av * (1.0f + tanhf(0.7978845608028654f * (av + 0.044715f * av * av * av))))

// ------------------- Softmax (last-axis) -------------------
// softmax_f32: one block per row, 256 threads, numerically stable.
// rows * cols = N elements. Input is row-major contiguous.
extern "C" __global__ void softmax_f32(const float *a, float *out, unsigned long rows, unsigned long cols) {
  extern __shared__ float sdata[];
  unsigned long row = blockIdx.x;
  if (row >= rows) return;

  const float *rowA = a + row * cols;
  float *rowOut = out + row * cols;
  int tid = threadIdx.x;
  int bs = blockDim.x;

  // Step 1: row max via block reduction.
  float thread_max = -INFINITY;
  for (unsigned long j = tid; j < cols; j += bs) {
    float v = rowA[j];
    if (v > thread_max) thread_max = v;
  }
  sdata[tid] = thread_max;
  __syncthreads();
  for (int s = bs / 2; s > 0; s >>= 1) {
    if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
    __syncthreads();
  }
  float maxVal = sdata[0];
  __syncthreads();

  // Step 2: exp(x - max) → tmp; thread-local sum.
  float thread_sum = 0.0f;
  for (unsigned long j = tid; j < cols; j += bs) {
    float e = expf(rowA[j] - maxVal);
    rowOut[j] = e;
    thread_sum += e;
  }
  sdata[tid] = thread_sum;
  __syncthreads();
  for (int s = bs / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  float sum = sdata[0];

  // Step 3: normalize.
  for (unsigned long j = tid; j < cols; j += bs) {
    rowOut[j] /= sum;
  }
}

// ------------------- Strided copy (for Contiguous) -------------------
// Copy a strided source tensor (arbitrary strides) into a row-major-contig dest.
// meta is [dims[0..rank), srcStrides[0..rank)] as int32.
#define COPY_STRIDED(NAME, T)                                                        \
  extern "C" __global__ void NAME(                                                   \
      const T *src, T *dst, unsigned long n, int rank, const int *meta) {            \
    unsigned long outIdx = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;     \
    if (outIdx >= n) return;                                                         \
    long srcOff = 0;                                                                 \
    unsigned long rem = outIdx;                                                      \
    for (int d = rank - 1; d >= 0; d--) {                                            \
      int dim = meta[d];                                                             \
      int ix = (int)(rem % (unsigned long)dim);                                      \
      rem /= (unsigned long)dim;                                                     \
      srcOff += (long)ix * (long)meta[rank + d];                                     \
    }                                                                                \
    dst[outIdx] = src[srcOff];                                                       \
  }

COPY_STRIDED(copy_strided_f32, float)
COPY_STRIDED(copy_strided_f64, double)
COPY_STRIDED(copy_strided_i32, int)
COPY_STRIDED(copy_strided_i64, long long)
COPY_STRIDED(copy_strided_f16, __half)
COPY_STRIDED(copy_strided_bf16, __nv_bfloat16)
COPY_STRIDED(copy_strided_u8, unsigned char)
COPY_STRIDED(copy_strided_i8, signed char)

extern "C" __global__ void softmax_f64(const double *a, double *out, unsigned long rows, unsigned long cols) {
  extern __shared__ double sdata_d[];
  unsigned long row = blockIdx.x;
  if (row >= rows) return;

  const double *rowA = a + row * cols;
  double *rowOut = out + row * cols;
  int tid = threadIdx.x;
  int bs = blockDim.x;

  double thread_max = -INFINITY;
  for (unsigned long j = tid; j < cols; j += bs) {
    double v = rowA[j];
    if (v > thread_max) thread_max = v;
  }
  sdata_d[tid] = thread_max;
  __syncthreads();
  for (int s = bs / 2; s > 0; s >>= 1) {
    if (tid < s && sdata_d[tid + s] > sdata_d[tid]) sdata_d[tid] = sdata_d[tid + s];
    __syncthreads();
  }
  double maxVal = sdata_d[0];
  __syncthreads();

  double thread_sum = 0.0;
  for (unsigned long j = tid; j < cols; j += bs) {
    double e = exp(rowA[j] - maxVal);
    rowOut[j] = e;
    thread_sum += e;
  }
  sdata_d[tid] = thread_sum;
  __syncthreads();
  for (int s = bs / 2; s > 0; s >>= 1) {
    if (tid < s) sdata_d[tid] += sdata_d[tid + s];
    __syncthreads();
  }
  double sum = sdata_d[0];

  for (unsigned long j = tid; j < cols; j += bs) {
    rowOut[j] /= sum;
  }
}

// ------------------- Strided binary (broadcast) -------------------
// meta layout: [outDims[0..rank), aStrides[0..rank), bStrides[0..rank)] int32.
// aStrides / bStrides are pre-broadcast-adjusted (0 for broadcast axes).
#define BIN_STRIDED(NAME, T, EXPR)                                                    \
  extern "C" __global__ void NAME(                                                    \
      const T *a, const T *b, T *out, unsigned long n, int rank, const int *meta) {  \
    unsigned long outIdx = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;      \
    if (outIdx >= n) return;                                                          \
    long aOff = 0, bOff = 0;                                                          \
    unsigned long rem = outIdx;                                                       \
    for (int d = rank - 1; d >= 0; d--) {                                             \
      int dim = meta[d];                                                              \
      int ix = (int)(rem % (unsigned long)dim);                                       \
      rem /= (unsigned long)dim;                                                      \
      aOff += (long)ix * (long)meta[rank + d];                                        \
      bOff += (long)ix * (long)meta[2 * rank + d];                                    \
    }                                                                                 \
    T av = a[aOff];                                                                   \
    T bv = b[bOff];                                                                   \
    out[outIdx] = (EXPR);                                                             \
  }

BIN_STRIDED(add_strided_f32, float, av + bv)
BIN_STRIDED(sub_strided_f32, float, av - bv)
BIN_STRIDED(mul_strided_f32, float, av *bv)
BIN_STRIDED(div_strided_f32, float, av / bv)
BIN_STRIDED(add_strided_f64, double, av + bv)
BIN_STRIDED(sub_strided_f64, double, av - bv)
BIN_STRIDED(mul_strided_f64, double, av *bv)
BIN_STRIDED(div_strided_f64, double, av / bv)

// ------------------- RMSNorm (last-axis) -------------------
// y = x / sqrt(mean(x²) + eps) * weight
// One block per row, 256 threads, shared-memory block reduction.
extern "C" __global__ void rmsnorm_f32(
    const float *x, const float *weight, float *out,
    unsigned long rows, unsigned long cols, float eps) {
  extern __shared__ float sdata[];
  unsigned long row = blockIdx.x;
  if (row >= rows) return;

  const float *rowX = x + row * cols;
  float *rowOut = out + row * cols;
  int tid = threadIdx.x;
  int bs = blockDim.x;

  // Sum of squares: warp shuffles, then the warp partials once through
  // shared memory (two barriers instead of log2(blockDim)).
  float thread_sum = 0.0f;
  for (unsigned long j = tid; j < cols; j += bs) {
    float v = rowX[j];
    thread_sum += v * v;
  }
  for (int o = 16; o > 0; o >>= 1) thread_sum += __shfl_xor_sync(0xffffffffu, thread_sum, o);
  if ((tid & 31) == 0) sdata[tid >> 5] = thread_sum;
  __syncthreads();
  if (tid < 32) {
    float v = tid < (bs >> 5) ? sdata[tid] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (tid == 0) sdata[0] = v;
  }
  __syncthreads();
  float rms = rsqrtf(sdata[0] / (float)cols + eps);

  // Normalize with weight.
  for (unsigned long j = tid; j < cols; j += bs) {
    rowOut[j] = rowX[j] * rms * weight[j];
  }
}

// ------------------- LayerNorm (last-axis) -------------------
// y = gamma * (x - mean) / sqrt(var + eps) + beta
extern "C" __global__ void layernorm_f32(
    const float *x, const float *gamma, const float *beta, float *out,
    unsigned long rows, unsigned long cols, float eps) {
  extern __shared__ float sdata2[];
  unsigned long row = blockIdx.x;
  if (row >= rows) return;

  const float *rowX = x + row * cols;
  float *rowOut = out + row * cols;
  int tid = threadIdx.x;
  int bs = blockDim.x;

  // Pass 1: sum for mean.
  float thread_sum = 0.0f;
  for (unsigned long j = tid; j < cols; j += bs) thread_sum += rowX[j];
  sdata2[tid] = thread_sum;
  __syncthreads();
  for (int s = bs / 2; s > 0; s >>= 1) {
    if (tid < s) sdata2[tid] += sdata2[tid + s];
    __syncthreads();
  }
  float mean = sdata2[0] / (float)cols;
  __syncthreads();

  // Pass 2: sum of squared deviations for variance.
  float thread_var = 0.0f;
  for (unsigned long j = tid; j < cols; j += bs) {
    float d = rowX[j] - mean;
    thread_var += d * d;
  }
  sdata2[tid] = thread_var;
  __syncthreads();
  for (int s = bs / 2; s > 0; s >>= 1) {
    if (tid < s) sdata2[tid] += sdata2[tid + s];
    __syncthreads();
  }
  float invStd = rsqrtf(sdata2[0] / (float)cols + eps);

  // Pass 3: normalize and scale.
  for (unsigned long j = tid; j < cols; j += bs) {
    rowOut[j] = (rowX[j] - mean) * invStd * gamma[j] + beta[j];
  }
}

// ------------------- IndexSelect (dim=0) -------------------
// out[i, ...] = src[indices[i], ...]
// indices is a 1-D I64 tensor; src is (numRows, rowSize).
extern "C" __global__ void index_select_dim0_f32(
    const float *src, const long long *indices, float *out,
    unsigned long outRows, unsigned long rowSize) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= outRows * rowSize) return;
  unsigned long outRow = i / rowSize;
  unsigned long col = i % rowSize;
  long long srcRow = indices[outRow];
  out[i] = src[srcRow * rowSize + col];
}

// ------------------- Missing elementwise (float) -------------------
UN_OP(abs_f32, float, av < 0 ? -av : av)
UN_OP(abs_f64, double, av < 0 ? -av : av)
UN_OP(sin_f32, float, sinf(av))
UN_OP(sin_f64, double, sin(av))
UN_OP(cos_f32, float, cosf(av))
UN_OP(cos_f64, double, cos(av))
UN_OP(tan_f32, float, tanf(av))
UN_OP(tan_f64, double, tan(av))
UN_OP(floor_f32, float, floorf(av))
UN_OP(floor_f64, double, floor(av))
UN_OP(ceil_f32, float, ceilf(av))
UN_OP(ceil_f64, double, ceil(av))
UN_OP(round_f32, float, roundf(av))
UN_OP(round_f64, double, round(av))
// GELU tanh approximation.
UN_OP(gelu_f32, float, 0.5f *av *(1.0f + tanhf(0.7978845608028654f * (av + 0.044715f * av *av *av))))
UN_OP(gelu_f64, double, 0.5 *av *(1.0 + tanh(0.7978845608028654 * (av + 0.044715 * av *av *av))))
// GELU exact (erf) — PyTorch default / HF BERT "gelu".
UN_OP(gelu_erf_f32, float, 0.5f *av *(1.0f + erff(av * 0.7071067811865476f)))
UN_OP(gelu_erf_f64, double, 0.5 *av *(1.0 + erf(av * 0.7071067811865476)))
BIN_OP(pow_f32, float, powf(av, bv))
BIN_OP(pow_f64, double, pow(av, bv))

// ------------------- Last-axis reductions -------------------
#define REDUCE_LAST(NAME, T, INIT, REDUCE, FINAL)                                \
  extern "C" __global__ void NAME(                                               \
      const T *a, T *out, unsigned long rows, unsigned long cols) {              \
    extern __shared__ char sdata_buf[];                                          \
    T *sdata = (T *)sdata_buf;                                                   \
    unsigned long row = blockIdx.x;                                              \
    if (row >= rows) return;                                                     \
    const T *rowA = a + row * cols;                                              \
    int tid = threadIdx.x;                                                       \
    int bs = blockDim.x;                                                         \
    T acc = INIT;                                                                \
    for (unsigned long j = tid; j < cols; j += bs) {                             \
      T v = rowA[j];                                                             \
      acc = REDUCE(acc, v);                                                      \
    }                                                                            \
    sdata[tid] = acc;                                                            \
    __syncthreads();                                                             \
    for (int s = bs / 2; s > 0; s >>= 1) {                                       \
      if (tid < s) sdata[tid] = REDUCE(sdata[tid], sdata[tid + s]);              \
      __syncthreads();                                                           \
    }                                                                            \
    if (tid == 0) out[row] = FINAL(sdata[0], (T)cols);                           \
  }

#define R_SUM(a, b) ((a) + (b))
#define R_MAX(a, b) ((a) > (b) ? (a) : (b))
#define R_MIN(a, b) ((a) < (b) ? (a) : (b))
#define F_IDENTITY(v, n) (v)
#define F_MEAN(v, n) ((v) / (n))

REDUCE_LAST(reduce_sum_f32, float, 0.0f, R_SUM, F_IDENTITY)
REDUCE_LAST(reduce_sum_f64, double, 0.0, R_SUM, F_IDENTITY)
REDUCE_LAST(reduce_mean_f32, float, 0.0f, R_SUM, F_MEAN)
REDUCE_LAST(reduce_mean_f64, double, 0.0, R_SUM, F_MEAN)
REDUCE_LAST(reduce_max_f32, float, -INFINITY, R_MAX, F_IDENTITY)
REDUCE_LAST(reduce_max_f64, double, -INFINITY, R_MAX, F_IDENTITY)
REDUCE_LAST(reduce_min_f32, float, INFINITY, R_MIN, F_IDENTITY)
REDUCE_LAST(reduce_min_f64, double, INFINITY, R_MIN, F_IDENTITY)

// ------------------- Argmax / Argmin (last axis) -------------------
// Simple single-block per row. Tracks (value, index) pair.
extern "C" __global__ void argmax_last_f32(
    const float *a, long long *out, unsigned long rows, unsigned long cols) {
  unsigned long row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) return;
  const float *rowA = a + row * cols;
  float bestV = rowA[0];
  long long bestI = 0;
  for (unsigned long j = 1; j < cols; j++) {
    if (rowA[j] > bestV) {
      bestV = rowA[j];
      bestI = (long long)j;
    }
  }
  out[row] = bestI;
}

extern "C" __global__ void argmin_last_f32(
    const float *a, long long *out, unsigned long rows, unsigned long cols) {
  unsigned long row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) return;
  const float *rowA = a + row * cols;
  float bestV = rowA[0];
  long long bestI = 0;
  for (unsigned long j = 1; j < cols; j++) {
    if (rowA[j] < bestV) {
      bestV = rowA[j];
      bestI = (long long)j;
    }
  }
  out[row] = bestI;
}

// ------------------- LogSoftmax (last axis) -------------------
extern "C" __global__ void logsoftmax_f32(
    const float *a, float *out, unsigned long rows, unsigned long cols) {
  extern __shared__ float sd_ls[];
  unsigned long row = blockIdx.x;
  if (row >= rows) return;
  const float *rowA = a + row * cols;
  float *rowO = out + row * cols;
  int tid = threadIdx.x;
  int bs = blockDim.x;

  // Row max.
  float tmax = -INFINITY;
  for (unsigned long j = tid; j < cols; j += bs)
    if (rowA[j] > tmax) tmax = rowA[j];
  sd_ls[tid] = tmax;
  __syncthreads();
  for (int s = bs / 2; s > 0; s >>= 1) {
    if (tid < s && sd_ls[tid + s] > sd_ls[tid]) sd_ls[tid] = sd_ls[tid + s];
    __syncthreads();
  }
  float mx = sd_ls[0];
  __syncthreads();

  // sum(exp(x - max))
  float tsum = 0.0f;
  for (unsigned long j = tid; j < cols; j += bs) tsum += expf(rowA[j] - mx);
  sd_ls[tid] = tsum;
  __syncthreads();
  for (int s = bs / 2; s > 0; s >>= 1) {
    if (tid < s) sd_ls[tid] += sd_ls[tid + s];
    __syncthreads();
  }
  float lse = mx + logf(sd_ls[0]);

  // (x - mx) - log(sum(exp(x-mx))) = x - lse
  for (unsigned long j = tid; j < cols; j += bs) rowO[j] = rowA[j] - lse;
}

// ------------------- Cast kernels (float family) -------------------
extern "C" __global__ void cast_f32_to_f16(const float *src, __half *dst, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(src[i]);
}
extern "C" __global__ void cast_f16_to_f32(const __half *src, float *dst, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __half2float(src[i]);
}
extern "C" __global__ void cast_f32_to_bf16(const float *src, __nv_bfloat16 *dst, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2bfloat16(src[i]);
}
extern "C" __global__ void cast_bf16_to_f32(const __nv_bfloat16 *src, float *dst, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __bfloat162float(src[i]);
}
extern "C" __global__ void cast_f32_to_f64(const float *src, double *dst, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = (double)src[i];
}
extern "C" __global__ void cast_f64_to_f32(const double *src, float *dst, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = (float)src[i];
}
extern "C" __global__ void cast_f32_to_i32(const float *src, int *dst, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = (int)src[i];
}
extern "C" __global__ void cast_i32_to_f32(const int *src, float *dst, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = (float)src[i];
}
extern "C" __global__ void cast_f32_to_i64(const float *src, long long *dst, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = (long long)src[i];
}
extern "C" __global__ void cast_i64_to_f32(const long long *src, float *dst, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = (float)src[i];
}

// Integer family ↔ F32 (narrow types go through F32 for every other pair,
// see castLauncher). Float→int truncates toward zero like the CPU kernels.
#define CAST_INT_F32(NAME, T) \
extern "C" __global__ void cast_##NAME##_to_f32(const T *src, float *dst, unsigned long n) { \
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x; \
  if (i < n) dst[i] = (float)src[i]; \
} \
extern "C" __global__ void cast_f32_to_##NAME(const float *src, T *dst, unsigned long n) { \
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x; \
  if (i < n) dst[i] = (T)(long long)src[i]; \
}
CAST_INT_F32(i8, signed char)
CAST_INT_F32(u8, unsigned char)
CAST_INT_F32(i16, short)
CAST_INT_F32(u16, unsigned short)
CAST_INT_F32(u32, unsigned int)
CAST_INT_F32(u64, unsigned long long)
#undef CAST_INT_F32

// ------------------- Compare (produce U8 output) -------------------
#define CMP_OP(NAME, T, OP)                                                         \
  extern "C" __global__ void NAME(const T *a, const T *b, unsigned char *out,       \
                                  unsigned long n) {                                \
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;         \
    if (i < n) {                                                                    \
      T av = a[i];                                                                  \
      T bv = b[i];                                                                  \
      out[i] = (OP) ? 1 : 0;                                                        \
    }                                                                               \
  }

CMP_OP(lt_f32, float, av < bv)
CMP_OP(le_f32, float, av <= bv)
CMP_OP(gt_f32, float, av > bv)
CMP_OP(ge_f32, float, av >= bv)
CMP_OP(eq_f32, float, av == bv)
CMP_OP(ne_f32, float, av != bv)

// Where kernel (ternary select). cond is U8; a, b, out are T.
#define WHERE_OP(NAME, T)                                                            \
  extern "C" __global__ void NAME(const unsigned char *cond, const T *a, const T *b, \
                                  T *out, unsigned long n) {                         \
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;          \
    if (i < n) out[i] = cond[i] ? a[i] : b[i];                                       \
  }

WHERE_OP(where_f32, float)
WHERE_OP(where_f64, double)

// ------------------- Extra last-axis reductions -------------------
// Prod
#define R_MUL(a, b) ((a) * (b))
REDUCE_LAST(reduce_prod_f32, float, 1.0f, R_MUL, F_IDENTITY)
REDUCE_LAST(reduce_prod_f64, double, 1.0, R_MUL, F_IDENTITY)

// L2Norm: reduce with sumOfSquares, then sqrt in the final stage.
#define F_SQRT_F32(v, n) (sqrtf(v))
#define F_SQRT_F64(v, n) (sqrt(v))
// For l2, we need sum(x²). The REDUCE macro feeds elements in directly, so we
// square them via a custom walk. Implement from scratch rather than hack the
// macro.
extern "C" __global__ void reduce_l2_f32(
    const float *a, float *out, unsigned long rows, unsigned long cols) {
  extern __shared__ float sd_l2[];
  unsigned long row = blockIdx.x;
  if (row >= rows) return;
  const float *rowA = a + row * cols;
  int tid = threadIdx.x;
  int bs = blockDim.x;
  float acc = 0.0f;
  for (unsigned long j = tid; j < cols; j += bs) {
    float v = rowA[j];
    acc += v * v;
  }
  sd_l2[tid] = acc;
  __syncthreads();
  for (int s = bs / 2; s > 0; s >>= 1) {
    if (tid < s) sd_l2[tid] += sd_l2[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[row] = sqrtf(sd_l2[0]);
}
extern "C" __global__ void reduce_l2_f64(
    const double *a, double *out, unsigned long rows, unsigned long cols) {
  extern __shared__ double sd_l2d[];
  unsigned long row = blockIdx.x;
  if (row >= rows) return;
  const double *rowA = a + row * cols;
  int tid = threadIdx.x;
  int bs = blockDim.x;
  double acc = 0.0;
  for (unsigned long j = tid; j < cols; j += bs) {
    double v = rowA[j];
    acc += v * v;
  }
  sd_l2d[tid] = acc;
  __syncthreads();
  for (int s = bs / 2; s > 0; s >>= 1) {
    if (tid < s) sd_l2d[tid] += sd_l2d[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[row] = sqrt(sd_l2d[0]);
}

// U8 reductions: All = min-like (0 short-circuits); Any = max-like.
extern "C" __global__ void reduce_all_u8(
    const unsigned char *a, unsigned char *out, unsigned long rows, unsigned long cols) {
  extern __shared__ unsigned char sd_all[];
  unsigned long row = blockIdx.x;
  if (row >= rows) return;
  const unsigned char *rowA = a + row * cols;
  int tid = threadIdx.x;
  int bs = blockDim.x;
  unsigned char acc = 1;
  for (unsigned long j = tid; j < cols; j += bs) {
    if (rowA[j] == 0) acc = 0;
  }
  sd_all[tid] = acc;
  __syncthreads();
  for (int s = bs / 2; s > 0; s >>= 1) {
    if (tid < s && sd_all[tid + s] == 0) sd_all[tid] = 0;
    __syncthreads();
  }
  if (tid == 0) out[row] = sd_all[0];
}
extern "C" __global__ void reduce_any_u8(
    const unsigned char *a, unsigned char *out, unsigned long rows, unsigned long cols) {
  extern __shared__ unsigned char sd_any[];
  unsigned long row = blockIdx.x;
  if (row >= rows) return;
  const unsigned char *rowA = a + row * cols;
  int tid = threadIdx.x;
  int bs = blockDim.x;
  unsigned char acc = 0;
  for (unsigned long j = tid; j < cols; j += bs) {
    if (rowA[j] != 0) acc = 1;
  }
  sd_any[tid] = acc;
  __syncthreads();
  for (int s = bs / 2; s > 0; s >>= 1) {
    if (tid < s && sd_any[tid + s] != 0) sd_any[tid] = 1;
    __syncthreads();
  }
  if (tid == 0) out[row] = sd_any[0];
}

// ------------------- Logical ops (U8 in, U8 out) -------------------
extern "C" __global__ void logical_and_u8(
    const unsigned char *a, const unsigned char *b, unsigned char *out, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = (a[i] != 0 && b[i] != 0) ? 1 : 0;
}
extern "C" __global__ void logical_or_u8(
    const unsigned char *a, const unsigned char *b, unsigned char *out, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = (a[i] != 0 || b[i] != 0) ? 1 : 0;
}
extern "C" __global__ void logical_not_u8(
    const unsigned char *a, unsigned char *out, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = (a[i] == 0) ? 1 : 0;
}

// ------------------- Gather (arbitrary dim) -------------------
// out[i0, ..., ik, ..., iN-1] = src[i0, ..., indices[i0, ..., ik, ..., iN-1], ..., iN-1]
// outer = prod(out_dims[0..dim]), axis_size = src_dims[dim], inner = prod(out_dims[dim+1..]).
extern "C" __global__ void gather_f32(
    const float *src, const long long *indices, float *out,
    unsigned long outer, unsigned long outAxis, unsigned long srcAxis, unsigned long inner) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = outer * outAxis * inner;
  if (i >= total) return;
  unsigned long innerIdx = i % inner;
  unsigned long outerIdx = i / (outAxis * inner);
  long long srcIdx = indices[i];
  unsigned long srcOff = outerIdx * srcAxis * inner + (unsigned long)srcIdx * inner + innerIdx;
  out[i] = src[srcOff];
}

// ------------------- ScatterAdd -------------------
// out[outer, indices[outer, j, inner], inner] += src[outer, j, inner]
// atomicAdd for race safety.
extern "C" __global__ void scatter_add_f32(
    const float *src, const long long *indices, float *out,
    unsigned long outer, unsigned long srcAxis, unsigned long outAxis, unsigned long inner) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = outer * srcAxis * inner;
  if (i >= total) return;
  unsigned long innerIdx = i % inner;
  unsigned long outerIdx = i / (srcAxis * inner);
  long long idx = indices[i];
  unsigned long outOff = outerIdx * outAxis * inner + (unsigned long)idx * inner + innerIdx;
  atomicAdd(&out[outOff], src[i]);
}

// ------------------- RoPE -------------------
// x: (batch*seqLen, headDim) — flattened outer dims; headDim even.
// positions: (seqLen,) I64.
// seqLen, headDim, base as scalars. batch recovered as rows / seqLen.
// Each thread handles one pair (2i, 2i+1) of one row.
extern "C" __global__ void rope_f32(
    const float *x, const long long *positions, float *out,
    unsigned long rows, unsigned long seqLen, unsigned long headDim, float base) {
  unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long halfDim = headDim / 2;
  unsigned long total = rows * halfDim;
  if (tid >= total) return;
  unsigned long row = tid / halfDim;
  unsigned long pair = tid % halfDim;
  unsigned long s = row % seqLen;
  float pos = (float)positions[s];
  float i = (float)(pair * 2);
  float theta = pos * powf(base, -i / (float)headDim);
  float c = cosf(theta);
  float sn = sinf(theta);
  // Rotate-half pairing (HF Llama / GPT-NeoX): dim `pair` with `pair + halfDim`.
  unsigned long base_off = row * headDim + pair;
  float x0 = x[base_off];
  float x1 = x[base_off + halfDim];
  out[base_off] = x0 * c - x1 * sn;
  out[base_off + halfDim] = x0 * sn + x1 * c;
}

// ------------------- Dropout (hash-based RNG) -------------------
// Uses a simple splitmix64-like hash of (seed, i) to derive uniform values in
// [0, 1). Deterministic for a given seed — good enough for dropout masking
// during inference-time experimentation. Training use should seed from a
// dedicated stream.
__device__ __forceinline__ float hashUniform(unsigned long long seed, unsigned long i) {
  unsigned long long z = seed ^ (i * 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z = z ^ (z >> 31);
  unsigned int u = (unsigned int)(z >> 32);
  return (float)u * (1.0f / 4294967296.0f);
}

extern "C" __global__ void dropout_f32(
    const float *x, float *out, unsigned long n, float p, float scale,
    unsigned long long seed) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float u = hashUniform(seed, i);
  out[i] = (u < p) ? 0.0f : x[i] * scale;
}

// ------------------- Tril / Triu -------------------
// Zero the elements above (tril) or below (triu) the k-th diagonal along the
// trailing two axes. Works in-place conceptually — caller passes a copy as `out`.
extern "C" __global__ void tril_f32(
    float *data, unsigned long batch, unsigned long rows, unsigned long cols, int k) {
  unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = batch * rows * cols;
  if (tid >= total) return;
  unsigned long j = tid % cols;
  unsigned long i = (tid / cols) % rows;
  // Tril keeps j <= i + k.
  if ((long long)j > (long long)i + (long long)k) data[tid] = 0;
}

extern "C" __global__ void triu_f32(
    float *data, unsigned long batch, unsigned long rows, unsigned long cols, int k) {
  unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = batch * rows * cols;
  if (tid >= total) return;
  unsigned long j = tid % cols;
  unsigned long i = (tid / cols) % rows;
  // Triu keeps j >= i + k.
  if ((long long)j < (long long)i + (long long)k) data[tid] = 0;
}

// ------------------- Cumsum / Cumprod (serial along axis) -------------------
// Each thread handles one "outer × inner" position, scanning serially along the
// axis. Not work-efficient (no Hillis-Steele scan) but correct and simple.
extern "C" __global__ void cumsum_f32(
    const float *a, float *out, unsigned long outer, unsigned long axSize, unsigned long inner) {
  unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = outer * inner;
  if (tid >= total) return;
  unsigned long o = tid / inner;
  unsigned long in = tid % inner;
  unsigned long base = o * axSize * inner + in;
  float acc = 0.0f;
  for (unsigned long j = 0; j < axSize; j++) {
    acc += a[base + j * inner];
    out[base + j * inner] = acc;
  }
}

extern "C" __global__ void cumprod_f32(
    const float *a, float *out, unsigned long outer, unsigned long axSize, unsigned long inner) {
  unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = outer * inner;
  if (tid >= total) return;
  unsigned long o = tid / inner;
  unsigned long in = tid % inner;
  unsigned long base = o * axSize * inner + in;
  float acc = 1.0f;
  for (unsigned long j = 0; j < axSize; j++) {
    acc *= a[base + j * inner];
    out[base + j * inner] = acc;
  }
}

// ------------------- Conv2d (naive, one thread per output element) -------------------
// input:   (N, Cin, H, W) row-major
// weight:  (Cout, Cin, kH, kW)
// bias:    (Cout,) or nullptr
// output:  (N, Cout, Hout, Wout)
extern "C" __global__ void conv2d_f32(
    const float *input, const float *weight, const float *bias, float *output,
    unsigned long N, unsigned long Cin, unsigned long H, unsigned long W,
    unsigned long Cout, unsigned long kH, unsigned long kW,
    unsigned long Hout, unsigned long Wout,
    int sH, int sW, int pH, int pW, int hasBias) {
  unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = N * Cout * Hout * Wout;
  if (tid >= total) return;
  unsigned long wOut = tid % Wout;
  unsigned long hOut = (tid / Wout) % Hout;
  unsigned long co = (tid / (Wout * Hout)) % Cout;
  unsigned long n = tid / (Cout * Hout * Wout);
  float acc = hasBias ? bias[co] : 0.0f;
  unsigned long inStrideN = Cin * H * W;
  unsigned long inStrideC = H * W;
  unsigned long wStrideCO = Cin * kH * kW;
  unsigned long wStrideCI = kH * kW;
  for (unsigned long ci = 0; ci < Cin; ci++) {
    for (unsigned long ky = 0; ky < kH; ky++) {
      long ih = (long)hOut * sH - pH + (long)ky;
      if (ih < 0 || ih >= (long)H) continue;
      for (unsigned long kx = 0; kx < kW; kx++) {
        long iw = (long)wOut * sW - pW + (long)kx;
        if (iw < 0 || iw >= (long)W) continue;
        acc += input[n * inStrideN + ci * inStrideC + (unsigned long)ih * W + (unsigned long)iw]
             * weight[co * wStrideCO + ci * wStrideCI + ky * kW + kx];
      }
    }
  }
  output[tid] = acc;
}

// ------------------- Conv1d -------------------
extern "C" __global__ void conv1d_f32(
    const float *input, const float *weight, const float *bias, float *output,
    unsigned long N, unsigned long Cin, unsigned long L,
    unsigned long Cout, unsigned long kL, unsigned long Lout,
    int stride, int pad, int hasBias) {
  unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = N * Cout * Lout;
  if (tid >= total) return;
  unsigned long o = tid % Lout;
  unsigned long co = (tid / Lout) % Cout;
  unsigned long n = tid / (Cout * Lout);
  float acc = hasBias ? bias[co] : 0.0f;
  for (unsigned long ci = 0; ci < Cin; ci++) {
    for (unsigned long k = 0; k < kL; k++) {
      long pos = (long)o * stride - pad + (long)k;
      if (pos < 0 || pos >= (long)L) continue;
      acc += input[n * Cin * L + ci * L + (unsigned long)pos]
           * weight[co * Cin * kL + ci * kL + k];
    }
  }
  output[tid] = acc;
}

// ------------------- Upsample nearest -------------------
extern "C" __global__ void upsample_nearest_f32(
    const float *input, float *output,
    unsigned long N, unsigned long C, unsigned long H, unsigned long W,
    unsigned long Hout, unsigned long Wout, int sH, int sW) {
  unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = N * C * Hout * Wout;
  if (tid >= total) return;
  unsigned long x = tid % Wout;
  unsigned long h = (tid / Wout) % Hout;
  unsigned long c = (tid / (Wout * Hout)) % C;
  unsigned long n = tid / (C * Hout * Wout);
  unsigned long srcH = h / sH;
  unsigned long srcW = x / sW;
  output[tid] = input[n * C * H * W + c * H * W + srcH * W + srcW];
}

// ------------------- MaxPool2d -------------------
extern "C" __global__ void maxpool2d_f32(
    const float *input, float *output,
    unsigned long N, unsigned long C, unsigned long H, unsigned long W,
    unsigned long Hout, unsigned long Wout,
    int kH, int kW, int sH, int sW, int pH, int pW) {
  unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = N * C * Hout * Wout;
  if (tid >= total) return;
  unsigned long wOut = tid % Wout;
  unsigned long hOut = (tid / Wout) % Hout;
  unsigned long c = (tid / (Wout * Hout)) % C;
  unsigned long n = tid / (C * Hout * Wout);
  float best = -INFINITY;
  bool found = false;
  unsigned long inStrideN = C * H * W;
  unsigned long inStrideC = H * W;
  for (int ky = 0; ky < kH; ky++) {
    long ih = (long)hOut * sH - pH + ky;
    if (ih < 0 || ih >= (long)H) continue;
    for (int kx = 0; kx < kW; kx++) {
      long iw = (long)wOut * sW - pW + kx;
      if (iw < 0 || iw >= (long)W) continue;
      float v = input[n * inStrideN + c * inStrideC + (unsigned long)ih * W + (unsigned long)iw];
      if (!found || v > best) {
        best = v;
        found = true;
      }
    }
  }
  output[tid] = found ? best : 0.0f;
}

// ------------------- AvgPool2d -------------------
extern "C" __global__ void avgpool2d_f32(
    const float *input, float *output,
    unsigned long N, unsigned long C, unsigned long H, unsigned long W,
    unsigned long Hout, unsigned long Wout,
    int kH, int kW, int sH, int sW, int pH, int pW) {
  unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = N * C * Hout * Wout;
  if (tid >= total) return;
  unsigned long wOut = tid % Wout;
  unsigned long hOut = (tid / Wout) % Hout;
  unsigned long c = (tid / (Wout * Hout)) % C;
  unsigned long n = tid / (C * Hout * Wout);
  float acc = 0.0f;
  int count = 0;
  unsigned long inStrideN = C * H * W;
  unsigned long inStrideC = H * W;
  for (int ky = 0; ky < kH; ky++) {
    long ih = (long)hOut * sH - pH + ky;
    if (ih < 0 || ih >= (long)H) continue;
    for (int kx = 0; kx < kW; kx++) {
      long iw = (long)wOut * sW - pW + kx;
      if (iw < 0 || iw >= (long)W) continue;
      acc += input[n * inStrideN + c * inStrideC + (unsigned long)ih * W + (unsigned long)iw];
      count++;
    }
  }
  output[tid] = count > 0 ? acc / (float)count : 0.0f;
}


// ------------------- sign / erf / recip / sqr / mish / hard_* (2026-10-28) -------------------
UN_OP(sign_f32, float, av > 0.0f ? 1.0f : (av < 0.0f ? -1.0f : 0.0f))
UN_OP(sign_f64, double, av > 0.0 ? 1.0 : (av < 0.0 ? -1.0 : 0.0))
UN_OP(sign_i32, int, av > 0 ? 1 : (av < 0 ? -1 : 0))
UN_OP(sign_i64, long long, av > 0 ? 1 : (av < 0 ? -1 : 0))
UN_OP(erf_f32, float, erff(av))
UN_OP(erf_f64, double, erf(av))
UN_OP(recip_f32, float, 1.0f / av)
UN_OP(recip_f64, double, 1.0 / av)
UN_OP(sqr_f32, float, av *av)
UN_OP(sqr_f64, double, av *av)
// mish(x) = x · tanh(softplus(x)); softplus saturates to x above 20.
UN_OP(mish_f32, float, av *tanhf(av > 20.0f ? av : log1pf(expf(av))))
UN_OP(mish_f64, double, av *tanh(av > 20.0 ? av : log1p(exp(av))))
// hard_sigmoid(x) = clamp((x + 3) / 6, 0, 1); hard_swish(x) = x · hard_sigmoid(x).
UN_OP(hard_sigmoid_f32, float, fminf(fmaxf((av + 3.0f) / 6.0f, 0.0f), 1.0f))
UN_OP(hard_sigmoid_f64, double, fmin(fmax((av + 3.0) / 6.0, 0.0), 1.0))
UN_OP(hard_swish_f32, float, av *fminf(fmaxf((av + 3.0f) / 6.0f, 0.0f), 1.0f))
UN_OP(hard_swish_f64, double, av *fmin(fmax((av + 3.0) / 6.0, 0.0), 1.0))

// ------------------- parametrised unary: (a, out, n, p0, p1) -------------------
// Two scalar parameters p0, p1 at the tensor's dtype (the launcher passes
// both; unused ones are 0).
#define UN_OP_P(NAME, T, EXPR)                                                           \
  extern "C" __global__ void NAME(const T *a, T *out, unsigned long n, T p0, T p1) {     \
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;              \
    if (i < n) {                                                                         \
      T av = a[i];                                                                       \
      out[i] = (EXPR);                                                                   \
    }                                                                                    \
  }

UN_OP_P(clamp_f32, float, fminf(fmaxf(av, p0), p1))
UN_OP_P(clamp_f64, double, fmin(fmax(av, p0), p1))
UN_OP_P(elu_f32, float, av > 0.0f ? av : p0 *(expf(av) - 1.0f))
UN_OP_P(elu_f64, double, av > 0.0 ? av : p0 *(exp(av) - 1.0))
UN_OP_P(leaky_relu_f32, float, av > 0.0f ? av : p0 *av)
UN_OP_P(leaky_relu_f64, double, av > 0.0 ? av : p0 *av)
UN_OP_P(powf_f32, float, powf(av, p0))
UN_OP_P(powf_f64, double, pow(av, p0))
UN_OP_P(affine_f32, float, av *p0 + p1)
UN_OP_P(affine_f64, double, av *p0 + p1)

// ------------------- maximum / minimum (plain + strided) -------------------
BIN_OP(maximum_f32, float, fmaxf(av, bv))
BIN_OP(maximum_f64, double, fmax(av, bv))
BIN_OP(maximum_i32, int, av > bv ? av : bv)
BIN_OP(maximum_i64, long long, av > bv ? av : bv)
BIN_OP(minimum_f32, float, fminf(av, bv))
BIN_OP(minimum_f64, double, fmin(av, bv))
BIN_OP(minimum_i32, int, av < bv ? av : bv)
BIN_OP(minimum_i64, long long, av < bv ? av : bv)
BIN_STRIDED(maximum_strided_f32, float, fmaxf(av, bv))
BIN_STRIDED(maximum_strided_f64, double, fmax(av, bv))
BIN_STRIDED(maximum_strided_i32, int, av > bv ? av : bv)
BIN_STRIDED(maximum_strided_i64, long long, av > bv ? av : bv)
BIN_STRIDED(minimum_strided_f32, float, fminf(av, bv))
BIN_STRIDED(minimum_strided_f64, double, fmin(av, bv))
BIN_STRIDED(minimum_strided_i32, int, av < bv ? av : bv)
BIN_STRIDED(minimum_strided_i64, long long, av < bv ? av : bv)

// ------------------- compare / where: F64 / I32 / I64 / U8 -------------------
#define CMP_SET(SUFFIX, T)              \
  CMP_OP(lt_##SUFFIX, T, av < bv)       \
  CMP_OP(le_##SUFFIX, T, av <= bv)      \
  CMP_OP(gt_##SUFFIX, T, av > bv)       \
  CMP_OP(ge_##SUFFIX, T, av >= bv)      \
  CMP_OP(eq_##SUFFIX, T, av == bv)      \
  CMP_OP(ne_##SUFFIX, T, av != bv)
CMP_SET(f64, double)
CMP_SET(i32, int)
CMP_SET(i64, long long)
CMP_SET(u8, unsigned char)
WHERE_OP(where_i32, int)
WHERE_OP(where_i64, long long)
WHERE_OP(where_u8, unsigned char)

// ------------------- interleaved RoPE (GPT-J / Candle rope_i) -------------------
// Same layout contract as rope_f32: x is (rows, headDim) with rows =
// batch*seqLen, positions (seqLen,) I64. Pair p rotates dims (2p, 2p+1) by
// pos * base^(-2p/headDim). BF16 variant lives in bf16_ops.cu.
#define ROPE_I(NAME, T)                                                                 \
  extern "C" __global__ void NAME(const T *x, const long long *positions, T *out,       \
                                  unsigned long rows, unsigned long seqLen,             \
                                  unsigned long headDim, T base) {                      \
    unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;           \
    unsigned long halfDim = headDim / 2;                                                \
    if (tid >= rows * halfDim) return;                                                  \
    unsigned long row = tid / halfDim;                                                  \
    unsigned long pair = tid % halfDim;                                                 \
    T pos = (T)positions[row % seqLen];                                                 \
    T theta = pos * pow(base, -(T)(2 * pair) / (T)headDim);                             \
    T c = cos(theta), sn = sin(theta);                                                  \
    unsigned long off = row * headDim + 2 * pair;                                       \
    T x0 = x[off], x1 = x[off + 1];                                                     \
    out[off] = x0 * c - x1 * sn;                                                        \
    out[off + 1] = x0 * sn + x1 * c;                                                    \
  }
ROPE_I(rope_i_f32, float)
ROPE_I(rope_i_f64, double)

// ------------------- F8E4M3 casts (2026-10-28) -------------------
// Storage is one byte per element; conversion is round-to-nearest-even with
// saturation to ±448 (__NV_SATFINITE), matching dtype.F8FromF32 on the CPU.
__device__ __forceinline__ float f8e4m3_to_f(unsigned char v) {
  __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)v, __NV_E4M3);
  return __half2float(__half(h));
}
__device__ __forceinline__ unsigned char f_to_f8e4m3(float v) {
  return (unsigned char)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}
extern "C" __global__ void cast_f32_to_f8e4m3(const float *src, unsigned char *dst, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = f_to_f8e4m3(src[i]);
}
extern "C" __global__ void cast_f8e4m3_to_f32(const unsigned char *src, float *dst, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = f8e4m3_to_f(src[i]);
}
extern "C" __global__ void cast_bf16_to_f8e4m3(const __nv_bfloat16 *src, unsigned char *dst, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = f_to_f8e4m3(__bfloat162float(src[i]));
}
extern "C" __global__ void cast_f8e4m3_to_bf16(const unsigned char *src, __nv_bfloat16 *dst, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2bfloat16(f8e4m3_to_f(src[i]));
}
extern "C" __global__ void cast_f16_to_f8e4m3(const __half *src, unsigned char *dst, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = f_to_f8e4m3(__half2float(src[i]));
}
extern "C" __global__ void cast_f8e4m3_to_f16(const unsigned char *src, __half *dst, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(f8e4m3_to_f(src[i]));
}

// ------------------- Strided binary F16 (2026-10-28) -------------------
// Same meta contract as add_strided_f32; compute in F32.
#define BIN_STRIDED_F16(NAME, EXPR)                                                   \
  extern "C" __global__ void NAME(                                                    \
      const __half *a, const __half *b, __half *out, unsigned long n, int rank,       \
      const int *meta) {                                                              \
    unsigned long outIdx = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;      \
    if (outIdx >= n) return;                                                          \
    long aOff = 0, bOff = 0;                                                          \
    unsigned long rem = outIdx;                                                       \
    for (int d = rank - 1; d >= 0; d--) {                                             \
      int dim = meta[d];                                                              \
      int ix = (int)(rem % (unsigned long)dim);                                       \
      rem /= (unsigned long)dim;                                                      \
      aOff += (long)ix * (long)meta[rank + d];                                        \
      bOff += (long)ix * (long)meta[2 * rank + d];                                    \
    }                                                                                 \
    float av = __half2float(a[aOff]);                                                 \
    float bv = __half2float(b[bOff]);                                                 \
    out[outIdx] = __float2half(EXPR);                                                 \
  }
BIN_STRIDED_F16(add_strided_f16, av + bv)
BIN_STRIDED_F16(sub_strided_f16, av - bv)
BIN_STRIDED_F16(mul_strided_f16, av *bv)
BIN_STRIDED_F16(div_strided_f16, av / bv)
BIN_STRIDED_F16(maximum_strided_f16, fmaxf(av, bv))
BIN_STRIDED_F16(minimum_strided_f16, fminf(av, bv))

// rope_partial_f32: RoPE over the first rotDim dims of each head, the rest
// copied (ops.RoPEPartial). Same pairing and angles as rope_f32 with rotDim
// in place of headDim. One thread per pair: pairs below rotDim/2 rotate,
// the others copy two of the pass-through dims.
extern "C" __global__ void rope_partial_f32(
    const float *x, const long long *positions, float *out,
    unsigned long rows, unsigned long seqLen, unsigned long headDim, unsigned long rotDim, float base) {
  unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long halfDim = headDim / 2;
  if (tid >= rows * halfDim) return;
  unsigned long row = tid / halfDim;
  unsigned long pair = tid % halfDim;
  unsigned long halfRot = rotDim / 2;
  unsigned long rowOff = row * headDim;
  if (pair >= halfRot) {
    unsigned long i0 = rowOff + rotDim + (pair - halfRot);
    unsigned long i1 = i0 + (headDim - rotDim) / 2;
    out[i0] = x[i0];
    out[i1] = x[i1];
    return;
  }
  unsigned long s = row % seqLen;
  float pos = (float)positions[s];
  float i = (float)(pair * 2);
  float theta = pos * powf(base, -i / (float)rotDim);
  float c = cosf(theta);
  float sn = sinf(theta);
  unsigned long base_off = rowOff + pair;
  float x0 = (x[base_off]);
  float x1 = (x[base_off + halfRot]);
  out[base_off] = (x0 * c - x1 * sn);
  out[base_off + halfRot] = (x0 * sn + x1 * c);
}
