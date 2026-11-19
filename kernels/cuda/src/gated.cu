// Gated-MLP activations fused with the rounding for the next GEMM.
//
// silu_mul_f32_bf16: out = bf16(silu(g) * u) over n elements, the SwiGLU
// product of a gated MLP (Z-Image, Llama-style w1 / w3), written straight
// in BF16 for the down projection's tensor-core GEMM: one pass instead of
// silu, multiply and cast kernels over F32 intermediates. silu is
// g / (1 + exp(-g)) in F32 as the unfused silu_f32 kernel computes it.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" __global__ void silu_mul_f32_bf16(const float *g, const float *u, __nv_bfloat16 *out, long n) {
  long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i + 3 < n) {
    float4 a = *reinterpret_cast<const float4 *>(g + i);
    float4 b = *reinterpret_cast<const float4 *>(u + i);
    __nv_bfloat162 lo = __floats2bfloat162_rn(a.x / (1.0f + expf(-a.x)) * b.x, a.y / (1.0f + expf(-a.y)) * b.y);
    __nv_bfloat162 hi = __floats2bfloat162_rn(a.z / (1.0f + expf(-a.z)) * b.z, a.w / (1.0f + expf(-a.w)) * b.w);
    *reinterpret_cast<__nv_bfloat162 *>(out + i) = lo;
    *reinterpret_cast<__nv_bfloat162 *>(out + i + 2) = hi;
  } else {
    for (; i < n; i++) {
      float a = g[i];
      out[i] = __float2bfloat16_rn(a / (1.0f + expf(-a)) * u[i]);
    }
  }
}
