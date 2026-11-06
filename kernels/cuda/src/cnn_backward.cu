// Backward kernels for CNN ops: pool, conv2d. One-thread-per-input-cell.
//
// These mirror the CPU reference in light/ops/*_backward.go so the
// CPU↔CUDA parity tests are straightforward.

#include <cuda_runtime.h>

// ---- MaxPool2d backward ----
// One thread per input cell. For each (n, c, h, w), iterate over every
// output cell (oh, ow) whose window contains (h, w); if (h, w) is the
// arg-max for that window (reproducing the forward's first-in-row-major tie
// break), accumulate grad_out[n, c, oh, ow]. Forward kernel is in
// elementwise.cu — this must match its argmax convention.
extern "C" __global__ void maxpool2d_backward_f32(
    const float *input, const float *gradOut, float *gradIn,
    unsigned long N, unsigned long C, unsigned long H, unsigned long W,
    unsigned long Hout, unsigned long Wout,
    int kH, int kW, int sH, int sW, int pH, int pW) {
  unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = N * C * H * W;
  if (tid >= total) return;

  unsigned long w = tid % W;
  unsigned long h = (tid / W) % H;
  unsigned long c = (tid / (W * H)) % C;
  unsigned long n = tid / (C * H * W);

  float acc = 0.0f;
  unsigned long inStrideN = C * H * W;
  unsigned long inStrideC = H * W;
  unsigned long outStrideN = C * Hout * Wout;
  unsigned long outStrideC = Hout * Wout;

  for (unsigned long oh = 0; oh < Hout; oh++) {
    long startH = (long)oh * sH - pH;
    long endH = startH + kH;
    if ((long)h < startH || (long)h >= endH) continue;
    for (unsigned long ow = 0; ow < Wout; ow++) {
      long startW = (long)ow * sW - pW;
      long endW = startW + kW;
      if ((long)w < startW || (long)w >= endW) continue;

      // Find argmax in this window, matching forward's first-in-row-major tie break.
      float bestVal = 0.0f;
      bool found = false;
      long bestIH = -1, bestIW = -1;
      for (int ky = 0; ky < kH; ky++) {
        long ih = startH + ky;
        if (ih < 0 || ih >= (long)H) continue;
        for (int kx = 0; kx < kW; kx++) {
          long iw = startW + kx;
          if (iw < 0 || iw >= (long)W) continue;
          float v = input[n * inStrideN + c * inStrideC + (unsigned long)ih * W + (unsigned long)iw];
          if (!found || v > bestVal) {
            bestVal = v;
            bestIH = ih;
            bestIW = iw;
            found = true;
          }
        }
      }
      if (found && bestIH == (long)h && bestIW == (long)w) {
        acc += gradOut[n * outStrideN + c * outStrideC + oh * Wout + ow];
      }
    }
  }
  gradIn[tid] = acc;
}

// ---- AvgPool2d backward ----
// One thread per input cell. Each output cell (oh, ow) whose window contains
// (h, w) contributes grad_out[oh, ow] / count, where count is the number of
// unpadded input cells in that window (matching forward).
extern "C" __global__ void avgpool2d_backward_f32(
    const float *gradOut, float *gradIn,
    unsigned long N, unsigned long C, unsigned long H, unsigned long W,
    unsigned long Hout, unsigned long Wout,
    int kH, int kW, int sH, int sW, int pH, int pW) {
  unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = N * C * H * W;
  if (tid >= total) return;

  unsigned long w = tid % W;
  unsigned long h = (tid / W) % H;
  unsigned long c = (tid / (W * H)) % C;
  unsigned long n = tid / (C * H * W);

  float acc = 0.0f;
  unsigned long outStrideN = C * Hout * Wout;
  unsigned long outStrideC = Hout * Wout;

  for (unsigned long oh = 0; oh < Hout; oh++) {
    long startH = (long)oh * sH - pH;
    long endH = startH + kH;
    if ((long)h < startH || (long)h >= endH) continue;
    for (unsigned long ow = 0; ow < Wout; ow++) {
      long startW = (long)ow * sW - pW;
      long endW = startW + kW;
      if ((long)w < startW || (long)w >= endW) continue;

      // Count unpadded cells in this window (matches forward's behaviour).
      int count = 0;
      for (int ky = 0; ky < kH; ky++) {
        long ih = startH + ky;
        if (ih < 0 || ih >= (long)H) continue;
        for (int kx = 0; kx < kW; kx++) {
          long iw = startW + kx;
          if (iw < 0 || iw >= (long)W) continue;
          count++;
        }
      }
      if (count > 0) {
        acc += gradOut[n * outStrideN + c * outStrideC + oh * Wout + ow] / (float)count;
      }
    }
  }
  gradIn[tid] = acc;
}

// ---- Conv2d grad_input ----
// One thread per input cell. For each output cell whose window contains
// (h, w), add weight[co, ci, ky, kx] * grad_out[n, co, oh, ow] to grad_in.
extern "C" __global__ void conv2d_grad_input_f32(
    const float *weight, const float *gradOut, float *gradIn,
    unsigned long N, unsigned long Cin, unsigned long H, unsigned long W,
    unsigned long Cout, unsigned long Hout, unsigned long Wout,
    int kH, int kW, int sH, int sW, int pH, int pW) {
  unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = N * Cin * H * W;
  if (tid >= total) return;

  unsigned long w = tid % W;
  unsigned long h = (tid / W) % H;
  unsigned long ci = (tid / (W * H)) % Cin;
  unsigned long n = tid / (Cin * H * W);

  float acc = 0.0f;
  unsigned long outStrideN = Cout * Hout * Wout;
  unsigned long outStrideC = Hout * Wout;
  unsigned long wStrideCout = Cin * kH * kW;
  unsigned long wStrideCin = kH * kW;

  for (unsigned long co = 0; co < Cout; co++) {
    for (int ky = 0; ky < kH; ky++) {
      // oh * sH + ky - pH == h → oh = (h + pH - ky) / sH, must be non-negative and divisible by sH.
      long numH = (long)h + pH - ky;
      if (numH < 0 || numH % sH != 0) continue;
      long oh = numH / sH;
      if (oh < 0 || oh >= (long)Hout) continue;
      for (int kx = 0; kx < kW; kx++) {
        long numW = (long)w + pW - kx;
        if (numW < 0 || numW % sW != 0) continue;
        long ow = numW / sW;
        if (ow < 0 || ow >= (long)Wout) continue;
        float wv = weight[co * wStrideCout + ci * wStrideCin + (unsigned long)ky * kW + kx];
        float gv = gradOut[n * outStrideN + co * outStrideC + (unsigned long)oh * Wout + (unsigned long)ow];
        acc += wv * gv;
      }
    }
  }
  gradIn[tid] = acc;
}

// ---- Conv2d grad_weight ----
// One thread per weight element. Loops over (n, oh, ow) and accumulates
// input[n, ci, ih, iw] * grad_out[n, co, oh, ow].
extern "C" __global__ void conv2d_grad_weight_f32(
    const float *input, const float *gradOut, float *gradW,
    unsigned long N, unsigned long Cin, unsigned long H, unsigned long W,
    unsigned long Cout, unsigned long Hout, unsigned long Wout,
    int kH, int kW, int sH, int sW, int pH, int pW) {
  unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = Cout * Cin * kH * kW;
  if (tid >= total) return;

  unsigned long kx = tid % kW;
  unsigned long ky = (tid / kW) % kH;
  unsigned long ci = (tid / (kW * kH)) % Cin;
  unsigned long co = tid / (Cin * kH * kW);

  float acc = 0.0f;
  unsigned long inStrideN = Cin * H * W;
  unsigned long inStrideC = H * W;
  unsigned long outStrideN = Cout * Hout * Wout;
  unsigned long outStrideC = Hout * Wout;

  for (unsigned long n = 0; n < N; n++) {
    for (unsigned long oh = 0; oh < Hout; oh++) {
      long ih = (long)oh * sH + (long)ky - pH;
      if (ih < 0 || ih >= (long)H) continue;
      for (unsigned long ow = 0; ow < Wout; ow++) {
        long iw = (long)ow * sW + (long)kx - pW;
        if (iw < 0 || iw >= (long)W) continue;
        float xv = input[n * inStrideN + ci * inStrideC + (unsigned long)ih * W + (unsigned long)iw];
        float gv = gradOut[n * outStrideN + co * outStrideC + oh * Wout + ow];
        acc += xv * gv;
      }
    }
  }
  gradW[tid] = acc;
}

// ---- Conv2d grad_bias ----
// One thread per output channel; reduces over (n, oh, ow).
extern "C" __global__ void conv2d_grad_bias_f32(
    const float *gradOut, float *gradB,
    unsigned long N, unsigned long Cout, unsigned long Hout, unsigned long Wout) {
  unsigned long co = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (co >= Cout) return;
  float acc = 0.0f;
  unsigned long strideN = Cout * Hout * Wout;
  unsigned long strideC = Hout * Wout;
  for (unsigned long n = 0; n < N; n++) {
    for (unsigned long oh = 0; oh < Hout; oh++) {
      for (unsigned long ow = 0; ow < Wout; ow++) {
        acc += gradOut[n * strideN + co * strideC + oh * Wout + ow];
      }
    }
  }
  gradB[co] = acc;
}
