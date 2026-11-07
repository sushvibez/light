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

// ---------------------------------------------------------------------------
// gated_delta_chunked_f32: the chunked form for prefill, specialised to
// dk = dv = 128 (every Qwen3.5 size). Same math as kernels/cpu/gated_delta.go
// chunked(), in blocks of GDC_CH tokens:
//
//   A = I + strictly-lower[(β_i k_i · k_j) e^{γ_i − γ_j}]       (CH × CH)
//   P = lower[(q_i · k_j) e^{γ_i − γ_j}]                          (CH × CH)
//   U = A⁻¹ (β v),  W = A⁻¹ (β k e^{γ})                            (forward substitution)
//   V' = U − W S
//   o_c = e^{γ_c} q_c S + Σ_{m ≤ c} P_cm V'_m
//   S ← e^{γ_last} S + Σ_c (k_c e^{γ_last − γ_c}) ⊗ V'_c
//
// grid = B·H blocks, block = 256 threads. The 128 × 128 state lives in
// registers for the whole sequence: thread t holds column j = t mod 128,
// rows [64·(t / 128), +64). Per chunk the block stages q, k (normalised),
// β v → U → V', W, A, P and one partial-sum buffer in shared memory:
// 5 · CH·128 + 2 · CH² + 2 · CH floats ≈ 43 KB for CH = 16, under the
// default 48 KB, so no opt-in is needed. Padding past T (β = 0, g = 0,
// q = k = v = 0) leaves the state unchanged, as in the reference.
// ---------------------------------------------------------------------------

#define GDC_CH 16
#define GDC_D 128
#define GDC_ROWS 64  // state rows per thread (two threads per column)

extern "C" __global__ void __launch_bounds__(256) gated_delta_chunked_f32(
    const float *q, const float *k, const float *v, const float *g, const float *beta,
    const float *state_in, float *state_out, float *out,
    int B, int T, int H, int l2norm) {
  __shared__ float qs[GDC_CH * GDC_D];   // normalised, scaled q
  __shared__ float ks[GDC_CH * GDC_D];   // normalised k
  __shared__ float us[GDC_CH * GDC_D];   // β v → U → V'
  __shared__ float ws[GDC_CH * GDC_D];   // β k e^γ → W
  __shared__ float xs[GDC_CH * GDC_D];   // partial sums from the second row half
  __shared__ float As[GDC_CH * GDC_CH];
  __shared__ float Ps[GDC_CH * GDC_CH];
  __shared__ float cum[GDC_CH];          // cumulative log decay γ within the chunk
  __shared__ float bt[GDC_CH];           // β
  __shared__ float ex[GDC_CH];           // e^{γ_last − γ_c}

  const int tid = threadIdx.x;
  const int bh = blockIdx.x;
  const int b = bh / H, h = bh % H;
  const int col = tid & (GDC_D - 1);     // state column j
  const int half = tid >> 7;             // which 64 state rows
  const int row0 = half * GDC_ROWS;
  const int warp = tid >> 5, lane = tid & 31;
  const long long sbase = (long long)bh * GDC_D * GDC_D;
  const float inv_sqrt_d = rsqrtf((float)GDC_D);

  float s[GDC_ROWS];
#pragma unroll
  for (int r = 0; r < GDC_ROWS; r++) s[r] = state_in[sbase + (long long)(row0 + r) * GDC_D + col];

  for (int t0 = 0; t0 < T; t0 += GDC_CH) {
    // 1. Stage the chunk (zero past T).
    for (int idx = tid; idx < GDC_CH * GDC_D; idx += blockDim.x) {
      const int c = idx / GDC_D, i = idx % GDC_D, t = t0 + c;
      if (t < T) {
        const long long base = (((long long)b * T + t) * H + h) * GDC_D + i;
        qs[idx] = q[base];
        ks[idx] = k[base];
        us[idx] = v[base];
      } else {
        qs[idx] = 0.f;
        ks[idx] = 0.f;
        us[idx] = 0.f;
      }
    }
    if (tid < GDC_CH) {
      const int t = t0 + tid;
      const long long gi = ((long long)b * T + t) * H + h;
      cum[tid] = t < T ? g[gi] : 0.f;
      bt[tid] = t < T ? beta[gi] : 0.f;
    }
    __syncthreads();

    // 2. Row norms (one warp per row, 2·CH rows over 8 warps) and the
    //    cumulative decay.
    for (int r = warp; r < 2 * GDC_CH; r += 8) {
      float *row = r < GDC_CH ? qs + r * GDC_D : ks + (r - GDC_CH) * GDC_D;
      float ss = 0.f;
      for (int i = lane; i < GDC_D; i += 32) ss += row[i] * row[i];
      for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      float sc = l2norm ? rsqrtf(ss + 1e-6f) : 1.f;
      if (r < GDC_CH) sc *= inv_sqrt_d;
      for (int i = lane; i < GDC_D; i += 32) row[i] *= sc;
    }
    if (tid == 0) {
      float acc = 0.f;
      for (int c = 0; c < GDC_CH; c++) {
        acc += cum[c];
        cum[c] = acc;
      }
      for (int c = 0; c < GDC_CH; c++) ex[c] = expf(acc - cum[c]);
    }
    __syncthreads();

    // 3. A (strictly lower) and P (lower incl. diagonal): one entry per thread.
    {
      const int i = tid / GDC_CH, j = tid % GDC_CH;
      float kk = 0.f, qk = 0.f;
      if (j <= i) {
        for (int x = 0; x < GDC_D; x++) {
          const float kj = ks[j * GDC_D + x];
          kk += ks[i * GDC_D + x] * kj;
          qk += qs[i * GDC_D + x] * kj;
        }
      }
      const float d = j <= i ? expf(cum[i] - cum[j]) : 0.f;
      As[tid] = j < i ? bt[i] * kk * d : 0.f;
      Ps[tid] = j <= i ? qk * d : 0.f;
    }
    __syncthreads();

    // 4. Forward substitution, one column per thread: threads 0..127 solve
    //    U = A⁻¹ (β v), threads 128..255 W = A⁻¹ (β k e^γ).
    {
      float *X = half ? ws : us;
      for (int c = 0; c < GDC_CH; c++) {
        float x = half ? ks[c * GDC_D + col] * bt[c] * expf(cum[c]) : us[c * GDC_D + col] * bt[c];
        for (int m = 0; m < c; m++) x -= As[c * GDC_CH + m] * X[m * GDC_D + col];
        X[c * GDC_D + col] = x;
      }
    }
    __syncthreads();

    // 5. V' = U − W S. Each thread sums its 64 rows; the second row half
    //    hands its partials to the first through xs.
    float part[GDC_CH];
#pragma unroll
    for (int c = 0; c < GDC_CH; c++) {
      float p = 0.f;
#pragma unroll
      for (int r = 0; r < GDC_ROWS; r++) p += ws[c * GDC_D + row0 + r] * s[r];
      part[c] = p;
    }
    if (half) {
#pragma unroll
      for (int c = 0; c < GDC_CH; c++) xs[c * GDC_D + col] = part[c];
    }
    __syncthreads();
    if (!half) {
#pragma unroll
      for (int c = 0; c < GDC_CH; c++) us[c * GDC_D + col] -= part[c] + xs[c * GDC_D + col];
    }
    __syncthreads();

    // 6. Outputs: o_c = e^{γ_c} (q_c S) + Σ_{m ≤ c} P_cm V'_m, same split.
#pragma unroll
    for (int c = 0; c < GDC_CH; c++) {
      float p = 0.f;
#pragma unroll
      for (int r = 0; r < GDC_ROWS; r++) p += qs[c * GDC_D + row0 + r] * s[r];
      part[c] = p;
    }
    if (half) {
#pragma unroll
      for (int c = 0; c < GDC_CH; c++) xs[c * GDC_D + col] = part[c];
    }
    __syncthreads();
    if (!half) {
      for (int c = 0; c < GDC_CH; c++) {
        const int t = t0 + c;
        if (t >= T) break;
        float o = expf(cum[c]) * (part[c] + xs[c * GDC_D + col]);
        for (int m = 0; m <= c; m++) o += Ps[c * GDC_CH + m] * us[m * GDC_D + col];
        out[(((long long)b * T + t) * H + h) * GDC_D + col] = o;
      }
    }

    // 7. Carry the state: S ← e^{γ_last} S + Σ_c (k_c e^{γ_last − γ_c}) ⊗ V'_c.
    const float decay = expf(cum[GDC_CH - 1]);
#pragma unroll
    for (int r = 0; r < GDC_ROWS; r++) {
      float acc = s[r] * decay;
      for (int c = 0; c < GDC_CH; c++) acc += ks[c * GDC_D + row0 + r] * ex[c] * us[c * GDC_D + col];
      s[r] = acc;
    }
    __syncthreads();  // the next chunk overwrites the shared buffers
  }
#pragma unroll
  for (int r = 0; r < GDC_ROWS; r++) state_out[sbase + (long long)(row0 + r) * GDC_D + col] = s[r];
}
