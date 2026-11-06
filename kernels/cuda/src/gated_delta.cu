// Gated DeltaNet kernels (Qwen3-Next, Qwen3.5 / 3.6 / 3.8), F32.
// CUDA counterparts of kernels/cpu/gated_delta.go; see ops.GatedDeltaRule
// and ops.CausalConv1dDepthwise for the math and shapes.
//
// Built with scripts/nvrtc_ptx.py (or scripts/build_ptx.sh). Only
// fixed-width integer types: `long` is 32-bit on Windows and 64-bit on
// Linux, and the PTX parameter widths must match the Go launcher on both.

#include <cuda_runtime.h>

#define GD_MAX_DK 256

// gated_delta_recurrent_f32 runs the gated delta rule token by token.
//
//   q, k  (B, T, H, dk)   v, out (B, T, H, dv)   g, beta (B, T, H)
//   state_in / state_out  (B, H, dk, dv)
//
// grid = B·H blocks, block = dv threads (dv ≤ 1024, dk ≤ GD_MAX_DK). Thread
// j owns column j of the head's state S (dk values) for the whole sequence;
// the token's q and k are staged in shared memory (2·dk floats). Per token:
//   S ← e^g S;  δ = β (v − Sᵀk);  S ← S + k δᵀ;  o = Sᵀ q / √dk
// with q, k optionally L2-normalised first (eps 1e-6), as transformers'
// torch_recurrent_gated_delta_rule. It also serves prefill: the chunked
// form is the same function, only cheaper in FLOPs on long prompts.
extern "C" __global__ void gated_delta_recurrent_f32(
    const float *q, const float *k, const float *v, const float *g, const float *beta,
    const float *state_in, float *state_out, float *out,
    int B, int T, int H, int dk, int dv, int l2norm) {
  extern __shared__ float sh[];
  float *qs = sh;
  float *ks = sh + dk;
  const int bh = blockIdx.x;
  const int b = bh / H, h = bh % H;
  const int j = threadIdx.x;
  const bool active = j < dv;
  const long long sbase = (long long)bh * dk * dv;

  float s[GD_MAX_DK];
  if (active) {
    for (int i = 0; i < dk; i++) s[i] = state_in[sbase + (long long)i * dv + j];
  }
  const float inv_sqrt_dk = rsqrtf((float)dk);
  for (int t = 0; t < T; t++) {
    const long long tok = ((long long)b * T + t) * H + h;
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
      qs[i] = q[tok * dk + i];
      ks[i] = k[tok * dk + i];
    }
    __syncthreads();
    if (active) {
      float qq = 0.f, kk = 0.f;
      for (int i = 0; i < dk; i++) {
        qq += qs[i] * qs[i];
        kk += ks[i] * ks[i];
      }
      float qscale = inv_sqrt_dk, kscale = 1.f;
      if (l2norm) {
        qscale *= rsqrtf(qq + 1e-6f);
        kscale = rsqrtf(kk + 1e-6f);
      }
      const float decay = expf(g[tok]);
      const float bt = beta[tok];
      float mem = 0.f;
      for (int i = 0; i < dk; i++) {
        s[i] *= decay;
        mem += s[i] * ks[i];
      }
      const float delta = (v[tok * dv + j] - mem * kscale) * bt;
      float o = 0.f;
      for (int i = 0; i < dk; i++) {
        s[i] += ks[i] * kscale * delta;
        o += s[i] * qs[i];
      }
      out[tok * dv + j] = o * qscale;
    }
    __syncthreads();  // the next token overwrites qs / ks
  }
  if (active) {
    for (int i = 0; i < dk; i++) state_out[sbase + (long long)i * dv + j] = s[i];
  }
}

// causal_conv1d_depthwise_f32: out[b,c,t] = act(Σ_j w[c,j] · in(t − K + 1 + j)),
// where in(i) for i < 0 reads the carried state (B, C, K−1). One thread per
// output element; n = B·C·T.
extern "C" __global__ void causal_conv1d_depthwise_f32(
    const float *x, const float *w, const float *state_in, float *out,
    int B, int C, int T, int K, int silu) {
  const long long n = (long long)B * C * T;
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const int t = (int)(idx % T);
  const long long bc = idx / T;
  const int c = (int)(bc % C);
  const int P = K - 1;
  float acc = 0.f;
  for (int j = 0; j < K; j++) {
    const int i = t - P + j;
    const float xv = i < 0 ? state_in[bc * P + (P + i)] : x[bc * T + i];
    acc += w[(long long)c * K + j] * xv;
  }
  if (silu) acc = acc / (1.f + expf(-acc));
  out[idx] = acc;
}

// causal_conv1d_state_f32 writes the new state: the last K−1 inputs of
// [state_in, x] per (b, c). One thread per state element; n = B·C·(K−1).
extern "C" __global__ void causal_conv1d_state_f32(
    const float *x, const float *state_in, float *state_out, int B, int C, int T, int K) {
  const int P = K - 1;
  const long long n = (long long)B * C * P;
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n || P == 0) return;
  const int i = (int)(idx % P);
  const long long bc = idx / P;
  const int src = T - P + i;
  state_out[idx] = src < 0 ? state_in[bc * P + (P + src)] : x[bc * T + src];
}

// softplus_f32: log(1 + eˣ), x itself above 20 (torch's threshold).
extern "C" __global__ void softplus_f32(const float *a, float *out, long long n) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const float x = a[idx];
  out[idx] = x > 20.f ? x : log1pf(expf(x));
}
