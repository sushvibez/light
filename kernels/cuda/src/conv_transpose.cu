// Transposed convolution and bilinear upsampling (2026-10-28).
//
// conv_transpose2d_{f32,bf16}: gather per output pixel (no atomics), same
// mapping as kernels/cpu/conv_transpose.go. input (N, Cin, H, W), weight
// (Cin, Cout/g, kH, kW), bias (Cout) or null, output (N, Cout, Hout, Wout).
//
// upsample_bilinear2d_{f32,bf16}: F.interpolate(mode="bilinear") with both
// align_corners modes; source index math in bilinear_src matches PyTorch.
//
// Every kernel is an unmangled `extern "C" __global__`; BF16 variants load/
// store BF16 and accumulate in F32.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define IDENT(v) (v)

#define CONV_TRANSPOSE2D(NAME, T, TO, FROM)                                                   \
  extern "C" __global__ void NAME(                                                            \
      const T *input, const T *weight, const T *bias, T *output,                              \
      unsigned long N, unsigned long Cin, unsigned long H, unsigned long W,                   \
      unsigned long Cout, unsigned long kH, unsigned long kW,                                 \
      unsigned long Hout, unsigned long Wout,                                                 \
      int sH, int sW, int pH, int pW, int dH, int dW, int groups, int hasBias) {              \
    unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;                 \
    unsigned long total = N * Cout * Hout * Wout;                                             \
    if (tid >= total) return;                                                                 \
    unsigned long ow = tid % Wout;                                                            \
    unsigned long oh = (tid / Wout) % Hout;                                                   \
    unsigned long co = (tid / (Wout * Hout)) % Cout;                                          \
    unsigned long n = tid / (Cout * Hout * Wout);                                             \
    unsigned long CoutG = Cout / groups, CinG = Cin / groups;                                 \
    unsigned long g = co / CoutG, cog = co % CoutG;                                           \
    float acc = hasBias ? TO(bias[co]) : 0.0f;                                                \
    for (unsigned long ky = 0; ky < kH; ky++) {                                               \
      long num = (long)oh + pH - (long)ky * dH;                                               \
      if (num < 0 || num % sH != 0) continue;                                                 \
      long ih = num / sH;                                                                     \
      if (ih >= (long)H) continue;                                                            \
      for (unsigned long kx = 0; kx < kW; kx++) {                                             \
        long numw = (long)ow + pW - (long)kx * dW;                                            \
        if (numw < 0 || numw % sW != 0) continue;                                             \
        long iw = numw / sW;                                                                  \
        if (iw >= (long)W) continue;                                                          \
        for (unsigned long cig = 0; cig < CinG; cig++) {                                      \
          unsigned long ci = g * CinG + cig;                                                  \
          acc += TO(input[((n * Cin + ci) * H + ih) * W + iw]) *                              \
                 TO(weight[((ci * CoutG + cog) * kH + ky) * kW + kx]);                        \
        }                                                                                     \
      }                                                                                       \
    }                                                                                         \
    output[tid] = FROM(acc);                                                                  \
  }
CONV_TRANSPOSE2D(conv_transpose2d_f32, float, IDENT, IDENT)
CONV_TRANSPOSE2D(conv_transpose2d_bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16)

__device__ __forceinline__ void bilinear_src(unsigned long d, unsigned long in, unsigned long out,
                                             int align, unsigned long *i0, unsigned long *i1, float *lam) {
  float src = 0.0f;
  if (align) {
    if (out > 1) src = (float)d * (float)(in - 1) / (float)(out - 1);
  } else {
    src = ((float)d + 0.5f) * (float)in / (float)out - 0.5f;
    if (src < 0.0f) src = 0.0f;
  }
  unsigned long a = (unsigned long)floorf(src);
  if (a > in - 1) a = in - 1;
  *i0 = a;
  *i1 = (a + 1 < in) ? a + 1 : in - 1;
  *lam = src - (float)a;
}

#define UPSAMPLE_BILINEAR(NAME, T, TO, FROM)                                                  \
  extern "C" __global__ void NAME(                                                            \
      const T *input, T *output, unsigned long N, unsigned long C,                            \
      unsigned long H, unsigned long W, unsigned long Hout, unsigned long Wout, int align) {  \
    unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;                 \
    unsigned long total = N * C * Hout * Wout;                                                \
    if (tid >= total) return;                                                                 \
    unsigned long ow = tid % Wout;                                                            \
    unsigned long oh = (tid / Wout) % Hout;                                                   \
    unsigned long nc = tid / (Wout * Hout);                                                   \
    unsigned long h0, h1, w0, w1;                                                             \
    float lh, lw;                                                                             \
    bilinear_src(oh, H, Hout, align, &h0, &h1, &lh);                                          \
    bilinear_src(ow, W, Wout, align, &w0, &w1, &lw);                                          \
    const T *base = input + nc * H * W;                                                       \
    float v = (1.0f - lh) * ((1.0f - lw) * TO(base[h0 * W + w0]) + lw * TO(base[h0 * W + w1])) + \
              lh * ((1.0f - lw) * TO(base[h1 * W + w0]) + lw * TO(base[h1 * W + w1]));         \
    output[tid] = FROM(v);                                                                    \
  }
UPSAMPLE_BILINEAR(upsample_bilinear2d_f32, float, IDENT, IDENT)
UPSAMPLE_BILINEAR(upsample_bilinear2d_bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16)
