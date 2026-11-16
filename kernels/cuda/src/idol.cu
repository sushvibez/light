// Fused elementwise kernels for Idol's transformer blocks (models/idol.go):
// each replaces a chain of F32 passes over (tokens, channels) activations
// between the FP16 tensor-core GEMMs (the GEMMs take F16 inputs, F32
// outputs without their bias, which the next kernel adds).
//
// idol_ln_mod_f16     LayerNorm (optional affine) and adaLN modulation
//                     x̂·(1 + scale) + shift, rounded to F16 for the GEMM.
// idol_head_f16      one of q / k / v from a projection's F32 output:
//                     bias, RePo rotary (angles from the RePo offsets and
//                     learned frequencies), MultiHeadRMSNorm, F16 out for
//                     the FP16 flash attention kernel. Head dim 64.
// idol_gate_res_f32   x + (y + bias)·gate: the gated residual.
// idol_bias_gelu_f16  tanh-GELU(y + bias) to F16 (the FFN's middle).
// idol_silu_mul_f16   silu(g + bg)·(u + bu) to F16 (RePo's gated MLP).
//
// Arithmetic follows the unfused kernels (elementwise.cu): F32 throughout,
// the same tanh-GELU, silu and cosf / sinf expressions, round-to-nearest
// casts.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Two / four consecutive values of an F32 or F16 array, as F32.
__device__ __forceinline__ float2 idol_ld2(const float *p) { return *reinterpret_cast<const float2 *>(p); }
__device__ __forceinline__ float2 idol_ld2(const __half *p) { return __half22float2(*reinterpret_cast<const __half2 *>(p)); }
__device__ __forceinline__ float4 idol_ld4(const float *p) { return *reinterpret_cast<const float4 *>(p); }
__device__ __forceinline__ float4 idol_ld4(const __half *p) {
  float2 a = idol_ld2(p), b = idol_ld2(p + 2);
  return make_float4(a.x, a.y, b.x, b.y);
}

__device__ __forceinline__ float idol_block_sum(float v, float *sh) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
  __syncthreads();
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  float s = lane < nw ? sh[lane] : 0.f;
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  return s;
}

// One block (256 threads) per row; cols a multiple of 4, at most 4096.
// w / b (cols) may be null (no affine); scale / shift (batches, cols) may
// be null (no modulation); row r uses batch r / rows_per_batch.
extern "C" __global__ void __launch_bounds__(256)
idol_ln_mod_f16(const float *x, const float *w, const float *b, const float *scale, const float *shift,
                __half *out, long rows, int cols, long rows_per_batch, float eps) {
  __shared__ float sh[32];
  long row = blockIdx.x;
  if (row >= rows) return;
  const float4 *xr = reinterpret_cast<const float4 *>(x + row * cols);
  int n4 = cols / 4;
  float4 v[4];
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int j = threadIdx.x + i * blockDim.x;
    if (j < n4) {
      v[i] = xr[j];
      s += v[i].x + v[i].y + v[i].z + v[i].w;
    }
  }
  float mean = idol_block_sum(s, sh) / (float)cols;
  float q = 0.f;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int j = threadIdx.x + i * blockDim.x;
    if (j < n4) {
      float a = v[i].x - mean, c = v[i].y - mean, d = v[i].z - mean, e = v[i].w - mean;
      q += a * a + c * c + d * d + e * e;
    }
  }
  float inv = rsqrtf(idol_block_sum(q, sh) / (float)cols + eps);
  long bi = row / rows_per_batch;
  const float *sc = scale ? scale + bi * cols : nullptr;
  const float *sf = shift ? shift + bi * cols : nullptr;
  __half2 *o = reinterpret_cast<__half2 *>(out + row * cols);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int j = threadIdx.x + i * blockDim.x;
    if (j < n4) {
      float y[4] = {(v[i].x - mean) * inv, (v[i].y - mean) * inv, (v[i].z - mean) * inv, (v[i].w - mean) * inv};
#pragma unroll
      for (int k = 0; k < 4; k++) {
        int c = 4 * j + k;
        if (w) y[k] = y[k] * w[c] + b[c];
        if (sc) y[k] = y[k] * (sc[c] + 1.0f) + sf[c];
      }
      o[2 * j] = __floats2half2_rn(y[0], y[1]);
      o[2 * j + 1] = __floats2half2_rn(y[2], y[3]);
    }
  }
}

// A warp per (row, head), lane j owning the pair (2j, 2j + 1) of the 64
// head dims. x row r, head h, dim c is x[r * x_stride + off + h * 64 + c]
// (+ bias[off + h * 64 + c] when bias is set). With rope set (rope_d
// (rows, H, 3) RePo offsets, + rope_b (H * 3) when set), the pair turns by
// angle (rope_d[r, h, axis[j]]) * freq[j] as interleaved complex pairs
// (apply_rotary_emb). With gamma set (H, 64), the head is then
// RMS-normalised: y · rsqrt(mean(y²) + 1e-30) · gamma. out (rows, H, 64) F16.
// x is F32 (idol_head_f16) or F16 (idol_head_f16_h).
template <typename T>
__device__ __forceinline__ void idol_head(const T *x, long x_stride, long off, const float *bias,
                                          const float *rope_d, const float *rope_b, const float *freq,
                                          const float *axis, const float *gamma, __half *out, long rows,
                                          int H) {
  long wid = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (wid >= rows * H) return;
  long r = wid / H;
  int h = (int)(wid % H);
  long c0 = off + h * 64 + 2 * lane;
  float2 p = idol_ld2(x + r * x_stride + c0);
  if (bias) {
    p.x += bias[c0];
    p.y += bias[c0 + 1];
  }
  if (rope_d) {
    int a = (int)axis[lane];
    float dv = rope_d[(r * H + h) * 3 + a];
    if (rope_b) dv += rope_b[h * 3 + a];
    float ang = dv * freq[lane];
    float cs = cosf(ang), sn = sinf(ang);
    float y0 = p.x * cs - p.y * sn;
    float y1 = p.x * sn + p.y * cs;
    p.x = y0;
    p.y = y1;
  }
  if (gamma) {
    float s = p.x * p.x + p.y * p.y;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    float rms = rsqrtf(s / 64.0f + 1e-30f);
    p.x = p.x * rms * gamma[h * 64 + 2 * lane];
    p.y = p.y * rms * gamma[h * 64 + 2 * lane + 1];
  }
  *reinterpret_cast<__half2 *>(out + wid * 64 + 2 * lane) = __floats2half2_rn(p.x, p.y);
}

#define IDOL_HEAD(name, T) \
  extern "C" __global__ void name(const T *x, long x_stride, long off, const float *bias, const float *rope_d, \
  const float *rope_b, const float *freq, const float *axis, const float *gamma, \
  __half *out, long rows, int H) { \
  idol_head<T>(x, x_stride, off, bias, rope_d, rope_b, freq, axis, gamma, out, rows, H); \
  }
IDOL_HEAD(idol_head_f16, float)
IDOL_HEAD(idol_head_f16_h, __half)

// out = x + (y + bias) * gate over (rows, cols), cols a multiple of 4;
// bias (cols) and gate (batches, cols) may be null; y F32 or F16 (_h).
template <typename T>
__device__ __forceinline__ void idol_gate_res(const float *x, const T *y, const float *bias, const float *gate,
                                             float *out, long rows, int cols, long rows_per_batch) {
  long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i >= rows * cols) return;
  long r = i / cols;
  int c = (int)(i % cols);
  float4 a = *reinterpret_cast<const float4 *>(x + i);
  float4 v = idol_ld4(y + i);
  if (bias) {
    float4 bb = *reinterpret_cast<const float4 *>(bias + c);
    v.x += bb.x; v.y += bb.y; v.z += bb.z; v.w += bb.w;
  }
  if (gate) {
    float4 g = *reinterpret_cast<const float4 *>(gate + (r / rows_per_batch) * cols + c);
    v.x *= g.x; v.y *= g.y; v.z *= g.z; v.w *= g.w;
  }
  a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
  *reinterpret_cast<float4 *>(out + i) = a;
}

#define IDOL_GATE_RES(name, T) \
  extern "C" __global__ void name(const float *x, const T *y, const float *bias, const float *gate, float *out, \
  long rows, int cols, long rows_per_batch) { \
  idol_gate_res<T>(x, y, bias, gate, out, rows, cols, rows_per_batch); \
  }
IDOL_GATE_RES(idol_gate_res_f32, float)
IDOL_GATE_RES(idol_gate_res_f32_h, __half)

__device__ __forceinline__ float idol_gelu(float av) {
  return 0.5f * av * (1.0f + tanhf(0.7978845608028654f * (av + 0.044715f * av * av * av)));
}

// out = f16(gelu_tanh(y + bias)) over (n / cols, cols), cols a multiple of 4;
// y F32 or F16 (_h).
template <typename T>
__device__ __forceinline__ void idol_bias_gelu(const T *y, const float *bias, __half *out, long n, int cols) {
  long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i >= n) return;
  int c = (int)(i % cols);
  float4 v = idol_ld4(y + i);
  if (bias) {
    float4 bb = *reinterpret_cast<const float4 *>(bias + c);
    v.x += bb.x; v.y += bb.y; v.z += bb.z; v.w += bb.w;
  }
  __half2 *o = reinterpret_cast<__half2 *>(out + i);
  o[0] = __floats2half2_rn(idol_gelu(v.x), idol_gelu(v.y));
  o[1] = __floats2half2_rn(idol_gelu(v.z), idol_gelu(v.w));
}

extern "C" __global__ void idol_bias_gelu_f16(const float *y, const float *bias, __half *out, long n, int cols) {
  idol_bias_gelu<float>(y, bias, out, n, cols);
}
extern "C" __global__ void idol_bias_gelu_f16_h(const __half *y, const float *bias, __half *out, long n, int cols) {
  idol_bias_gelu<__half>(y, bias, out, n, cols);
}

__device__ __forceinline__ float idol_silu(float a) { return a / (1.0f + expf(-a)); }

// out = f16(silu(g + bg) * (u + bu)), cols a multiple of 4; biases may be null.
extern "C" __global__ void idol_silu_mul_f16(const float *g, const float *u, const float *bg, const float *bu,
                                             __half *out, long n, int cols) {
  long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i >= n) return;
  int c = (int)(i % cols);
  float4 a = *reinterpret_cast<const float4 *>(g + i);
  float4 b = *reinterpret_cast<const float4 *>(u + i);
  if (bg) {
    float4 t = *reinterpret_cast<const float4 *>(bg + c);
    a.x += t.x; a.y += t.y; a.z += t.z; a.w += t.w;
  }
  if (bu) {
    float4 t = *reinterpret_cast<const float4 *>(bu + c);
    b.x += t.x; b.y += t.y; b.z += t.z; b.w += t.w;
  }
  __half2 *o = reinterpret_cast<__half2 *>(out + i);
  o[0] = __floats2half2_rn(idol_silu(a.x) * b.x, idol_silu(a.y) * b.y);
  o[1] = __floats2half2_rn(idol_silu(a.z) * b.z, idol_silu(a.w) * b.w);
}
