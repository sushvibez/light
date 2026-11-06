// BF16 versions of the transformer-block ops that don't reduce to a
// straight elementwise. Each kernel reads BF16 storage, computes in F32 for
// stability, and writes BF16 back — the standard pattern for fast BF16
// inference. RMSNorm's weight stays F32 (small per-layer tensor).

#include <cuda_runtime.h>
#include <cuda_bf16.h>

static __device__ __forceinline__ float bf16_to_f(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
static __device__ __forceinline__ __nv_bfloat16 f_to_bf16(float x) {
  return __float2bfloat16(x);
}

// softmax_bf16: same shape contract as softmax_f32 — one block per row,
// 256 threads, F32 reductions. Stable: subtract row max before exp.
extern "C" __global__ void softmax_bf16(
    const __nv_bfloat16 *a, __nv_bfloat16 *out,
    unsigned long rows, unsigned long cols) {
  extern __shared__ float sdata_sm[];
  unsigned long row = blockIdx.x;
  if (row >= rows) return;

  const __nv_bfloat16 *rowA = a + row * cols;
  __nv_bfloat16 *rowOut = out + row * cols;
  int tid = threadIdx.x;
  int bs = blockDim.x;

  // Row max.
  float thread_max = -INFINITY;
  for (unsigned long j = tid; j < cols; j += bs) {
    float v = bf16_to_f(rowA[j]);
    if (v > thread_max) thread_max = v;
  }
  sdata_sm[tid] = thread_max;
  __syncthreads();
  for (int s = bs / 2; s > 0; s >>= 1) {
    if (tid < s && sdata_sm[tid + s] > sdata_sm[tid]) sdata_sm[tid] = sdata_sm[tid + s];
    __syncthreads();
  }
  float maxVal = sdata_sm[0];
  __syncthreads();

  // exp(x - max), tracked sum.
  float thread_sum = 0.0f;
  for (unsigned long j = tid; j < cols; j += bs) {
    float e = expf(bf16_to_f(rowA[j]) - maxVal);
    rowOut[j] = f_to_bf16(e);
    thread_sum += e;
  }
  sdata_sm[tid] = thread_sum;
  __syncthreads();
  for (int s = bs / 2; s > 0; s >>= 1) {
    if (tid < s) sdata_sm[tid] += sdata_sm[tid + s];
    __syncthreads();
  }
  float invSum = 1.0f / sdata_sm[0];

  // Normalise.
  for (unsigned long j = tid; j < cols; j += bs) {
    rowOut[j] = f_to_bf16(bf16_to_f(rowOut[j]) * invSum);
  }
}

// rmsnorm_bf16: BF16 in/out, BF16 weight. Compute in F32 for stability.
// Weight stored at the same dtype as the input — caller is expected to cast
// at load time, not on every forward.
extern "C" __global__ void rmsnorm_bf16(
    const __nv_bfloat16 *x, const __nv_bfloat16 *weight, __nv_bfloat16 *out,
    unsigned long rows, unsigned long cols, float eps) {
  extern __shared__ float sdata_rms[];
  unsigned long row = blockIdx.x;
  if (row >= rows) return;

  const __nv_bfloat16 *rowX = x + row * cols;
  __nv_bfloat16 *rowOut = out + row * cols;
  int tid = threadIdx.x;
  int bs = blockDim.x;

  float thread_sum = 0.0f;
  for (unsigned long j = tid; j < cols; j += bs) {
    float v = bf16_to_f(rowX[j]);
    thread_sum += v * v;
  }
  sdata_rms[tid] = thread_sum;
  __syncthreads();
  for (int s = bs / 2; s > 0; s >>= 1) {
    if (tid < s) sdata_rms[tid] += sdata_rms[tid + s];
    __syncthreads();
  }
  float rms = rsqrtf(sdata_rms[0] / (float)cols + eps);

  for (unsigned long j = tid; j < cols; j += bs) {
    float v = bf16_to_f(rowX[j]) * rms * bf16_to_f(weight[j]);
    rowOut[j] = f_to_bf16(v);
  }
}

// rope_bf16: BF16 in/out, I64 positions. headDim must be even. Same layout
// contract as rope_f32 — leading "rows" dim is broadcast (each row picks its
// position from `positions[row % seqLen]`).
extern "C" __global__ void rope_bf16(
    const __nv_bfloat16 *x, const long long *positions, __nv_bfloat16 *out,
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
  float x0 = bf16_to_f(x[base_off]);
  float x1 = bf16_to_f(x[base_off + halfDim]);
  out[base_off] = f_to_bf16(x0 * c - x1 * sn);
  out[base_off + halfDim] = f_to_bf16(x0 * sn + x1 * c);
}

// index_select_dim0_bf16: gather rows from a (V, D) BF16 table by I64 indices.
// Plain memcpy semantics with element-wise copy; compute happens elsewhere.
extern "C" __global__ void index_select_dim0_bf16(
    const __nv_bfloat16 *src, const long long *indices, __nv_bfloat16 *out,
    unsigned long outRows, unsigned long rowSize) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= outRows * rowSize) return;
  unsigned long outRow = i / rowSize;
  unsigned long col = i % rowSize;
  long long srcRow = indices[outRow];
  out[i] = src[srcRow * rowSize + col];
}

// Strided binary BF16 kernels. Same meta layout as add_strided_f32 et al.:
// [outDims[0..rank), aStrides[0..rank), bStrides[0..rank)] int32, with
// aStrides/bStrides pre-zeroed on broadcast axes. Compute is in F32.
#define BIN_STRIDED_BF16(NAME, EXPR)                                              \
  extern "C" __global__ void NAME(                                                \
      const __nv_bfloat16 *a, const __nv_bfloat16 *b, __nv_bfloat16 *out,         \
      unsigned long n, int rank, const int *meta) {                               \
    unsigned long outIdx = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;  \
    if (outIdx >= n) return;                                                      \
    long aOff = 0, bOff = 0;                                                      \
    unsigned long rem = outIdx;                                                   \
    for (int d = rank - 1; d >= 0; d--) {                                         \
      int dim = meta[d];                                                          \
      int ix = (int)(rem % (unsigned long)dim);                                   \
      rem /= (unsigned long)dim;                                                  \
      aOff += (long)ix * (long)meta[rank + d];                                    \
      bOff += (long)ix * (long)meta[2 * rank + d];                                \
    }                                                                             \
    float av = bf16_to_f(a[aOff]);                                                \
    float bv = bf16_to_f(b[bOff]);                                                \
    out[outIdx] = f_to_bf16(EXPR);                                                \
  }

BIN_STRIDED_BF16(add_strided_bf16, av + bv)
BIN_STRIDED_BF16(sub_strided_bf16, av - bv)
BIN_STRIDED_BF16(mul_strided_bf16, av *bv)
BIN_STRIDED_BF16(div_strided_bf16, av / bv)

// gelu_bf16 — single-kernel tanh-approx GELU (PyTorch's `gelu(x, approximate="tanh")`).
// Compute in F32 for stability, store BF16. Avoids the autograd-decomposed
// `Mul/Add/Tanh` chain which would otherwise need an F32→BF16 cast per
// constant tensor and 5+ kernel launches.
extern "C" __global__ void gelu_bf16(const __nv_bfloat16 *a, __nv_bfloat16 *out, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float av = bf16_to_f(a[i]);
  float v = 0.5f * av * (1.0f + tanhf(0.7978845608028654f * (av + 0.044715f * av * av * av)));
  out[i] = f_to_bf16(v);
}

// geglu_fwd_{f32,bf16} — fused GeGLU activation. Input: (..., 2*C) where the
// first C channels are `hidden` and the trailing C are `gate`. Output:
// (..., C) = hidden * GELU(gate). One kernel, one pass — replaces a
// Narrow + Contiguous + GELU + Mul chain (4 launches, 2 transient
// tensors of size n) with a single kernel and the output buffer alone.
extern "C" __global__ void geglu_fwd_f32(const float *x, float *out, unsigned long rows, unsigned long c) {
  unsigned long idx = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = rows * c;
  if (idx >= total) return;
  unsigned long row = idx / c;
  unsigned long col = idx % c;
  float h = x[row * 2 * c + col];
  float g = x[row * 2 * c + c + col];
  float gelu = 0.5f * g * (1.0f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g)));
  out[idx] = h * gelu;
}

extern "C" __global__ void geglu_fwd_bf16(const __nv_bfloat16 *x, __nv_bfloat16 *out, unsigned long rows, unsigned long c) {
  unsigned long idx = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = rows * c;
  if (idx >= total) return;
  unsigned long row = idx / c;
  unsigned long col = idx % c;
  float h = bf16_to_f(x[row * 2 * c + col]);
  float g = bf16_to_f(x[row * 2 * c + c + col]);
  float gelu = 0.5f * g * (1.0f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g)));
  out[idx] = f_to_bf16(h * gelu);
}

// ─── conv2d_bf16 ──────────────────────────────────────────────────────────────
//
// Direct nested-loop conv (im2col-free), one thread per output element.
// Same shape contract as `conv2d_f32`: input (N, Cin, H, W), weight
// (Cout, Cin, kH, kW), optional bias (Cout). Compute and accumulator are
// F32; only the BF16↔F32 round-trip happens at the loads/stores. Bias is
// BF16 too — the loader casts once, the inference path stays single-dtype.
extern "C" __global__ void conv2d_bf16(
    const __nv_bfloat16 *input, const __nv_bfloat16 *weight, const __nv_bfloat16 *bias,
    __nv_bfloat16 *output,
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
  float acc = hasBias ? bf16_to_f(bias[co]) : 0.0f;
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
        float in_v = bf16_to_f(input[n * inStrideN + ci * inStrideC +
                                (unsigned long)ih * W + (unsigned long)iw]);
        float w_v = bf16_to_f(weight[co * wStrideCO + ci * wStrideCI + ky * kW + kx]);
        acc += in_v * w_v;
      }
    }
  }
  output[tid] = f_to_bf16(acc);
}

// ─── layernorm_bf16 ───────────────────────────────────────────────────────────
//
// Last-axis LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta.
// Same blocking contract as `layernorm_f32` — one block per row, F32
// reductions in shared memory. Gamma/beta are BF16 — caller is expected to
// cast at load time so the inference path stays single-dtype.
extern "C" __global__ void layernorm_bf16(
    const __nv_bfloat16 *x, const __nv_bfloat16 *gamma, const __nv_bfloat16 *beta,
    __nv_bfloat16 *out,
    unsigned long rows, unsigned long cols, float eps) {
  extern __shared__ float sdata_ln[];
  unsigned long row = blockIdx.x;
  if (row >= rows) return;
  const __nv_bfloat16 *rowX = x + row * cols;
  __nv_bfloat16 *rowO = out + row * cols;
  int tid = threadIdx.x;
  int bs = blockDim.x;
  float *sSum = sdata_ln;
  float *sSumSq = sdata_ln + bs;
  float sum = 0.0f, sumSq = 0.0f;
  for (unsigned long j = tid; j < cols; j += bs) {
    float v = bf16_to_f(rowX[j]);
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
  float invN = 1.0f / (float)cols;
  float mean = sSum[0] * invN;
  float var = sSumSq[0] * invN - mean * mean;
  if (var < 0.0f) var = 0.0f;
  float invSigma = rsqrtf(var + eps);
  for (unsigned long j = tid; j < cols; j += bs) {
    float v = (bf16_to_f(rowX[j]) - mean) * invSigma;
    rowO[j] = f_to_bf16(v * bf16_to_f(gamma[j]) + bf16_to_f(beta[j]));
  }
}

// ─── upsample_nearest_bf16 ────────────────────────────────────────────────────
//
// Element-wise nearest-neighbour upsample. Same shape contract as
// `upsample_nearest_f32`: input (N, C, H, W), output (N, C, H*sH, W*sW).
// One thread per output element. Pure memcpy semantics — no compute, just
// a BF16→BF16 copy with a stride map.
extern "C" __global__ void upsample_nearest_bf16(
    const __nv_bfloat16 *input, __nv_bfloat16 *output,
    unsigned long N, unsigned long C, unsigned long H, unsigned long W,
    unsigned long Hout, unsigned long Wout, int sH, int sW) {
  unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = N * C * Hout * Wout;
  if (tid >= total) return;
  unsigned long wOut = tid % Wout;
  unsigned long hOut = (tid / Wout) % Hout;
  unsigned long c = (tid / (Wout * Hout)) % C;
  unsigned long n = tid / (C * Wout * Hout);
  unsigned long hIn = hOut / sH;
  unsigned long wIn = wOut / sW;
  output[tid] = input[((n * C + c) * H + hIn) * W + wIn];
}


// ------------------- BF16 elementwise additions (2026-10-28) -------------------
// sign / erf / recip / sqr / mish / hard_*, parametrised clamp / elu /
// leaky_relu / powf / affine (float p0, p1), maximum / minimum (plain +
// strided), compare / where, and interleaved RoPE. Compute in F32.
#define UN_BF16(NAME, EXPR)                                                                   \
  extern "C" __global__ void NAME(const __nv_bfloat16 *a, __nv_bfloat16 *out, unsigned long n) { \
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;                   \
    if (i < n) {                                                                              \
      float av = bf16_to_f(a[i]);                                                             \
      out[i] = f_to_bf16(EXPR);                                                               \
    }                                                                                         \
  }
#define UN_BF16_P(NAME, EXPR)                                                                   \
  extern "C" __global__ void NAME(const __nv_bfloat16 *a, __nv_bfloat16 *out, unsigned long n, \
                                  float p0, float p1) {                                         \
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;                     \
    if (i < n) {                                                                                \
      float av = bf16_to_f(a[i]);                                                               \
      out[i] = f_to_bf16(EXPR);                                                                 \
    }                                                                                           \
  }
#define BIN_BF16(NAME, EXPR)                                                                   \
  extern "C" __global__ void NAME(const __nv_bfloat16 *a, const __nv_bfloat16 *b,              \
                                  __nv_bfloat16 *out, unsigned long n) {                       \
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;                    \
    if (i < n) {                                                                               \
      float av = bf16_to_f(a[i]);                                                              \
      float bv = bf16_to_f(b[i]);                                                              \
      out[i] = f_to_bf16(EXPR);                                                                \
    }                                                                                          \
  }
#define CMP_BF16(NAME, OP)                                                                      \
  extern "C" __global__ void NAME(const __nv_bfloat16 *a, const __nv_bfloat16 *b,               \
                                  unsigned char *out, unsigned long n) {                        \
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;                     \
    if (i < n) {                                                                                \
      float av = bf16_to_f(a[i]);                                                               \
      float bv = bf16_to_f(b[i]);                                                               \
      out[i] = (OP) ? 1 : 0;                                                                    \
    }                                                                                           \
  }

UN_BF16(sign_bf16, av > 0.0f ? 1.0f : (av < 0.0f ? -1.0f : 0.0f))
UN_BF16(erf_bf16, erff(av))
UN_BF16(recip_bf16, 1.0f / av)
UN_BF16(sqr_bf16, av *av)
UN_BF16(mish_bf16, av *tanhf(av > 20.0f ? av : log1pf(expf(av))))
UN_BF16(hard_sigmoid_bf16, fminf(fmaxf((av + 3.0f) / 6.0f, 0.0f), 1.0f))
UN_BF16(hard_swish_bf16, av *fminf(fmaxf((av + 3.0f) / 6.0f, 0.0f), 1.0f))

UN_BF16_P(clamp_bf16, fminf(fmaxf(av, p0), p1))
UN_BF16_P(elu_bf16, av > 0.0f ? av : p0 *(expf(av) - 1.0f))
UN_BF16_P(leaky_relu_bf16, av > 0.0f ? av : p0 *av)
UN_BF16_P(powf_bf16, powf(av, p0))
UN_BF16_P(affine_bf16, av *p0 + p1)

BIN_BF16(maximum_bf16, fmaxf(av, bv))
BIN_BF16(minimum_bf16, fminf(av, bv))
BIN_STRIDED_BF16(maximum_strided_bf16, fmaxf(av, bv))
BIN_STRIDED_BF16(minimum_strided_bf16, fminf(av, bv))

CMP_BF16(lt_bf16, av < bv)
CMP_BF16(le_bf16, av <= bv)
CMP_BF16(gt_bf16, av > bv)
CMP_BF16(ge_bf16, av >= bv)
CMP_BF16(eq_bf16, av == bv)
CMP_BF16(ne_bf16, av != bv)

extern "C" __global__ void where_bf16(const unsigned char *cond, const __nv_bfloat16 *a,
                                      const __nv_bfloat16 *b, __nv_bfloat16 *out, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = cond[i] ? a[i] : b[i];
}

// rope_i_bf16: interleaved pairs (2p, 2p+1); same contract as rope_i_f32 in
// elementwise.cu and the row/position broadcast of rope_bf16 above.
extern "C" __global__ void rope_i_bf16(const __nv_bfloat16 *x, const long long *positions,
                                       __nv_bfloat16 *out, unsigned long rows, unsigned long seqLen,
                                       unsigned long headDim, float base) {
  unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long halfDim = headDim / 2;
  if (tid >= rows * halfDim) return;
  unsigned long row = tid / halfDim;
  unsigned long pair = tid % halfDim;
  float pos = (float)positions[row % seqLen];
  float theta = pos * powf(base, -(float)(2 * pair) / (float)headDim);
  float c = cosf(theta), sn = sinf(theta);
  unsigned long off = row * headDim + 2 * pair;
  float x0 = bf16_to_f(x[off]), x1 = bf16_to_f(x[off + 1]);
  out[off] = f_to_bf16(x0 * c - x1 * sn);
  out[off + 1] = f_to_bf16(x0 * sn + x1 * c);
}
