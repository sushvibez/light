// Naive 3-D convolution (F32): one thread per output element.
//
//   input  (N, Cin, D, H, W)   weight (Cout, Cin, kD, kH, kW)
//   bias   (Cout,) or null     output (N, Cout, Dout, Hout, Wout)

#include <cuda_runtime.h>

extern "C" __global__ void conv3d_f32(
    const float *in, const float *w, const float *bias, float *out,
    int N, int Cin, int D, int H, int W,
    int Cout, int kD, int kH, int kW,
    int Dout, int Hout, int Wout,
    int sD, int sH, int sW, int pD, int pH, int pW, int hasBias) {
  unsigned long idx = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = (unsigned long)N * Cout * Dout * Hout * Wout;
  if (idx >= total) return;

  int ow = (int)(idx % Wout);
  unsigned long r = idx / Wout;
  int oh = (int)(r % Hout);
  r /= Hout;
  int od = (int)(r % Dout);
  r /= Dout;
  int co = (int)(r % Cout);
  int n = (int)(r / Cout);

  float acc = hasBias ? bias[co] : 0.0f;
  for (int ci = 0; ci < Cin; ci++) {
    const float *inB = in + ((unsigned long)(n * Cin + ci)) * D * H * W;
    const float *wB = w + ((unsigned long)(co * Cin + ci)) * kD * kH * kW;
    for (int kd = 0; kd < kD; kd++) {
      int id = od * sD - pD + kd;
      if (id < 0 || id >= D) continue;
      for (int kh = 0; kh < kH; kh++) {
        int ih = oh * sH - pH + kh;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < kW; kw++) {
          int iw = ow * sW - pW + kw;
          if (iw < 0 || iw >= W) continue;
          acc += inB[(id * H + ih) * W + iw] * wB[(kd * kH + kh) * kW + kw];
        }
      }
    }
  }
  out[idx] = acc;
}
