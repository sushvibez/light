// RoPE with an explicit per-pair inverse-frequency table.
//
// Same layout contract as rope_f32 / rope_bf16 (elementwise.cu, bf16_ops.cu):
// x is (rows, headDim) with rows = prod(leading dims) * seqLen, each row picks
// its position from positions[row % seqLen]. Pairing is rotate-half (dim i
// with dim i + headDim/2), the HF Llama convention. Instead of computing
// base^(-2i/headDim) in-kernel, the angle for pair i is
//
//     theta = pos * invFreq[i]
//
// which lets the host apply Llama-3 / YaRN / linear frequency scaling once
// and reuse the table for every layer.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" __global__ void rope_scaled_f32(
    const float *x, const long long *positions, const float *invFreq, float *out,
    unsigned long rows, unsigned long seqLen, unsigned long headDim) {
  unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long halfDim = headDim / 2;
  unsigned long total = rows * halfDim;
  if (tid >= total) return;
  unsigned long row = tid / halfDim;
  unsigned long pair = tid % halfDim;
  unsigned long s = row % seqLen;
  float pos = (float)positions[s];
  float theta = pos * invFreq[pair];
  float c = cosf(theta);
  float sn = sinf(theta);
  // Rotate-half pairing: dim `pair` with dim `pair + halfDim`.
  unsigned long off = row * headDim + pair;
  float x0 = x[off];
  float x1 = x[off + halfDim];
  out[off] = x0 * c - x1 * sn;
  out[off + halfDim] = x0 * sn + x1 * c;
}

extern "C" __global__ void rope_scaled_bf16(
    const __nv_bfloat16 *x, const long long *positions, const float *invFreq, __nv_bfloat16 *out,
    unsigned long rows, unsigned long seqLen, unsigned long headDim) {
  unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long halfDim = headDim / 2;
  unsigned long total = rows * halfDim;
  if (tid >= total) return;
  unsigned long row = tid / halfDim;
  unsigned long pair = tid % halfDim;
  unsigned long s = row % seqLen;
  float pos = (float)positions[s];
  float theta = pos * invFreq[pair];
  float c = cosf(theta);
  float sn = sinf(theta);
  unsigned long off = row * headDim + pair;
  float x0 = __bfloat162float(x[off]);
  float x1 = __bfloat162float(x[off + halfDim]);
  out[off] = __float2bfloat16(x0 * c - x1 * sn);
  out[off + halfDim] = __float2bfloat16(x0 * sn + x1 * c);
}
