// FlashAttention-2 forward on BF16 or FP16 tensor cores (sm_80+), head dim 64 or 128.
//
// K / V tiles reach shared memory by cp.async (double-buffered at head
// dim 64) and the MMA B fragments come from it by ldmatrix (.trans for V).
//
// Layout: Q, K, V are BF16 with an F32 O (or all F16), each either head-major (B, H, S, D)
// (seq_major 0: one batch-head is a contiguous S*D slab) or the projections'
// own (B, S, H, D) (seq_major 1: a head's rows are H*D apart), so callers
// need no permute copies either way. A block owns 128 query rows of one
// batch-head (8 warps x 16 rows, or 4 warps x 32 rows for the F16 kernels)
// and streams the keys in tiles of FA16_BK = 64 through shared memory.
// Each warp keeps its query rows' A fragments, the online-softmax state
// (running max m, denominator l) and an F32 accumulator in registers; S = Q K^T and O += P V are
// mma.sync.m16n8k16 BF16 products with F32 accumulation. The m16n8k16
// fragment layouts (PTX ISA, "Matrix Fragments for mma.m16n8k16") give
// each thread rows lane/4 and lane/4 + 8, so the softmax rescale is per
// register, and a score tile's accumulator is exactly the A fragment of
// the P V product (FlashAttention-2's register reuse).
//
// S query rows attend to Sk key / value rows (Sk = S but for cross
// attention). scale multiplies the scores; causal masks key j > query i.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define FA16_BK 64
#define FA16_PAD 8                      // bf16 per smem row of padding (bank conflicts)

// Q, K, V are handled as raw 16-bit words (b16) up to the MMA, which reads
// them as BF16 or, with F16 set, as FP16; P is rounded to the same type.
typedef unsigned short fa16_b16;

template <bool F16>
__device__ __forceinline__ void fa16_mma(float *c, const unsigned *a, unsigned b0, unsigned b1) {
  if (F16)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  else
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <bool F16>
__device__ __forceinline__ unsigned fa16_pack(float lo, float hi) {
  if (F16) {
    __half2 v = __floats2half2_rn(lo, hi); // .x = lo (low 16 bits)
    return *reinterpret_cast<unsigned *>(&v);
  }
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<unsigned *>(&v);
}

// Stage rows [row0, row0 + FA16_BK) of a (S, D) BF16 slab into smem
// (D + FA16_PAD stride) with 16-byte cp.async copies, zero-filling rows
// past S (src-size 0). The caller commits and waits.
template <int D>
__device__ __forceinline__ void fa16_stage(fa16_b16 *dst, const fa16_b16 *src, int row0, int S, long ld) {
  const int chunks = D / 8; // 16-byte chunks per row
  for (int c = threadIdx.x; c < FA16_BK * chunks; c += blockDim.x) {
    int r = c / chunks, k = (c % chunks) * 8;
    bool in = row0 + r < S;
    const fa16_b16 *g = src + (long)(in ? row0 + r : 0) * ld + k;
    unsigned sa = (unsigned)__cvta_generic_to_shared(dst + r * (D + FA16_PAD) + k);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(g), "r"(in ? 16 : 0));
  }
}

__device__ __forceinline__ void fa16_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void fa16_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// Four 8x8 b16 matrices from smem: lane l supplies the address of row l % 8
// of matrix l / 8; r[i] gets matrix i's fragment (trans: transposed).
__device__ __forceinline__ void fa16_ldm4(unsigned *r, const fa16_b16 *p) {
  unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void fa16_ldm4t(unsigned *r, const fa16_b16 *p) {
  unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void fa16_store(float *p, float a, float b) { *reinterpret_cast<float2 *>(p) = make_float2(a, b); }
__device__ __forceinline__ void fa16_store(__half *p, float a, float b) { *reinterpret_cast<__half2 *>(p) = __floats2half2_rn(a, b); }

// STAGES key / value tiles in flight: 2 (double-buffered cp.async) for
// head dim 64; 1 for 128 (two would pass the 48 KB static smem limit).
// WARPS warps of MT m-tiles (16 query rows each): a block owns
// WARPS * 16 * MT query rows. MT = 2 shares every K / V fragment read
// from shared memory between two m-tiles (half the ldmatrix traffic per
// MMA), at twice the accumulator registers.
template <int D, int STAGES, bool F16, int MT, int WARPS, typename OutT>
__device__ __forceinline__ void fa16_body(const fa16_b16 *Q, const fa16_b16 *K, const fa16_b16 *V, OutT *O,
                     int S, int Sk, int H, int seq_major, float scale, int causal) {
  constexpr int LDS = D + FA16_PAD;
  constexpr int BQ = WARPS * 16 * MT;
  __shared__ __align__(16) fa16_b16 sK[STAGES][FA16_BK * LDS];
  __shared__ __align__(16) fa16_b16 sV[STAGES][FA16_BK * LDS];

  const int bh = blockIdx.y;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int g = lane / 4, t = lane % 4; // fragment row group, thread in group
  // Row stride and this batch-head's first element, in either layout.
  const long ld = seq_major ? (long)H * D : D;
  // S query rows, Sk key rows (cross attention; Sk = S otherwise).
  const long base = seq_major ? ((long)(bh / H) * S * H + bh % H) * D : (long)bh * S * D;
  const long kbase = seq_major ? ((long)(bh / H) * Sk * H + bh % H) * D : (long)bh * Sk * D;
  const int q0 = blockIdx.x * BQ + warp * 16 * MT; // this warp's first query row
  // Thread rows: m-tile mt owns rows q0 + 16 mt + g (r[mt][0]) and + 8 (r[mt][1]).
  int r[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; mt++) {
    r[mt][0] = q0 + 16 * mt + g;
    r[mt][1] = q0 + 16 * mt + g + 8;
  }

  int kEnd = Sk;
  if (causal) kEnd = min(Sk, (int)(blockIdx.x + 1) * BQ);
  const int nTiles = (kEnd + FA16_BK - 1) / FA16_BK;
  // Start the first tile's copies before reading Q.
  fa16_stage<D>(sK[0], K + kbase, 0, Sk, ld);
  fa16_stage<D>(sV[0], V + kbase, 0, Sk, ld);
  fa16_commit();

  // Q A-fragments for the k-steps of 16 over D, straight from global.
  unsigned qa[MT][D / 16][4];
  {
    const fa16_b16 *q = Q + base;
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
      const fa16_b16 *p0 = r[mt][0] < S ? q + r[mt][0] * ld : nullptr;
      const fa16_b16 *p1 = r[mt][1] < S ? q + r[mt][1] * ld : nullptr;
#pragma unroll
      for (int ks = 0; ks < D / 16; ks++) {
        int c = ks * 16 + t * 2;
        qa[mt][ks][0] = p0 ? *reinterpret_cast<const unsigned *>(p0 + c) : 0u;
        qa[mt][ks][1] = p1 ? *reinterpret_cast<const unsigned *>(p1 + c) : 0u;
        qa[mt][ks][2] = p0 ? *reinterpret_cast<const unsigned *>(p0 + c + 8) : 0u;
        qa[mt][ks][3] = p1 ? *reinterpret_cast<const unsigned *>(p1 + c + 8) : 0u;
      }
    }
  }

  float acc[MT][D / 8][4]; // n-tiles of 8 dims: c0,c1 row r[mt][0]; c2,c3 row r[mt][1]
  float m[MT][2], l[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; mt++) {
#pragma unroll
    for (int n = 0; n < D / 8; n++) acc[mt][n][0] = acc[mt][n][1] = acc[mt][n][2] = acc[mt][n][3] = 0.f;
    m[mt][0] = m[mt][1] = -INFINITY;
    l[mt][0] = l[mt][1] = 0.f;
  }
  const float sl2 = scale * 1.4426950408889634f; // scores in log2 units
  // ldmatrix row addresses: matrix lane / 8, row lane % 8.
  const int lm = lane / 8, lr = lane % 8;

  for (int it = 0; it < nTiles; it++) {
    const int k0 = it * FA16_BK;
    const int cur = STAGES == 2 ? (it & 1) : 0;
    if (STAGES == 2) {
      if (it + 1 < nTiles) { // prefetch the next tile into the other buffer
        fa16_stage<D>(sK[cur ^ 1], K + kbase, k0 + FA16_BK, Sk, ld);
        fa16_stage<D>(sV[cur ^ 1], V + kbase, k0 + FA16_BK, Sk, ld);
      }
      fa16_commit();
      fa16_wait<1>(); // this tile's group has landed
    } else {
      if (it > 0) {
        fa16_stage<D>(sK[0], K + kbase, k0, Sk, ld);
        fa16_stage<D>(sV[0], V + kbase, k0, Sk, ld);
        fa16_commit();
      }
      fa16_wait<0>();
    }
    __syncthreads();
    const fa16_b16 *tK = sK[cur], *tV = sV[cur];

    // S = Q K^T for this warp's rows x 64 keys: 8 n-tiles of 8 keys.
    // B(k, n) = K[key n][dim k]: matrices (keys n*8.., dims ks*16 + 0 / 8,
    // (ks+1)*16 + 0 / 8) give b0, b1 for two k-steps per ldmatrix.
    float s[MT][FA16_BK / 8][4];
#pragma unroll
    for (int n = 0; n < FA16_BK / 8; n++) {
#pragma unroll
      for (int mt = 0; mt < MT; mt++) s[mt][n][0] = s[mt][n][1] = s[mt][n][2] = s[mt][n][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < D / 16; ks += 2) {
        unsigned b[4];
        fa16_ldm4(b, tK + (n * 8 + lr) * LDS + ks * 16 + lm * 8);
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
          fa16_mma<F16>(s[mt][n], qa[mt][ks], b[0], b[1]);
          fa16_mma<F16>(s[mt][n], qa[mt][ks + 1], b[2], b[3]);
        }
      }
    }

    // Mask keys past S (and above the diagonal when causal; tiles wholly
    // inside both skip it), then the online softmax over each of the
    // thread's rows (reduced over the 4 lanes that share a row). Scores
    // stay unscaled: the max is taken on them (scale > 0) and the scale
    // folds into exp2's argument as one FMA.
    const bool full = k0 + FA16_BK <= Sk && (!causal || k0 + FA16_BK - 1 <= q0);
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
#pragma unroll
      for (int h = 0; h < 2; h++) { // row r[mt][h]: elements 2h, 2h + 1
        const int row = r[mt][h];
        float mx = -INFINITY;
#pragma unroll
        for (int n = 0; n < FA16_BK / 8; n++) {
#pragma unroll
          for (int e = 0; e < 2; e++) {
            if (!full) {
              int key = k0 + n * 8 + t * 2 + e;
              if (!(key < Sk && (!causal || key <= row))) s[mt][n][2 * h + e] = -INFINITY;
            }
            mx = fmaxf(mx, s[mt][n][2 * h + e]);
          }
        }
#pragma unroll
        for (int o = 1; o < 4; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
        float nm = fmaxf(m[mt][h], mx * sl2);
        // A row with every key masked so far keeps m = -inf: use 0 so exp2 gives 0, not NaN.
        float sub = nm == -INFINITY ? 0.f : nm;
        float al = exp2f(m[mt][h] - sub);
        m[mt][h] = nm;
        float rs = 0.f;
#pragma unroll
        for (int n = 0; n < FA16_BK / 8; n++) {
#pragma unroll
          for (int e = 0; e < 2; e++) {
            s[mt][n][2 * h + e] = exp2f(fmaf(s[mt][n][2 * h + e], sl2, -sub));
            rs += s[mt][n][2 * h + e];
          }
        }
#pragma unroll
        for (int o = 1; o < 4; o <<= 1) rs += __shfl_xor_sync(0xffffffff, rs, o);
        l[mt][h] = l[mt][h] * al + rs;
#pragma unroll
        for (int n = 0; n < D / 8; n++) {
          acc[mt][n][2 * h] *= al;
          acc[mt][n][2 * h + 1] *= al;
        }
      }
    }

    // O += P V: P's k-step j (keys 16j..16j+15) is score tiles 2j, 2j+1.
    // B(k, n) = V[key 16j + k][dim n]: transposed matrices (keys 16j + 0 / 8,
    // dims n*8 and n*8 + 8) give b0, b1 for two n-tiles per ldmatrix.
#pragma unroll
    for (int j = 0; j < FA16_BK / 16; j++) {
      unsigned pa[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; mt++) {
        pa[mt][0] = fa16_pack<F16>(s[mt][2 * j][0], s[mt][2 * j][1]);
        pa[mt][1] = fa16_pack<F16>(s[mt][2 * j][2], s[mt][2 * j][3]);
        pa[mt][2] = fa16_pack<F16>(s[mt][2 * j + 1][0], s[mt][2 * j + 1][1]);
        pa[mt][3] = fa16_pack<F16>(s[mt][2 * j + 1][2], s[mt][2 * j + 1][3]);
      }
#pragma unroll
      for (int n = 0; n < D / 8; n += 2) {
        unsigned b[4];
        fa16_ldm4t(b, tV + (16 * j + (lm & 1) * 8 + lr) * LDS + n * 8 + (lm >> 1) * 8);
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
          fa16_mma<F16>(acc[mt][n], pa[mt], b[0], b[1]);
          fa16_mma<F16>(acc[mt][n + 1], pa[mt], b[2], b[3]);
        }
      }
    }
    __syncthreads(); // the buffer is refilled next (STAGES 1) or after next (2)
  }

  // O = acc / l, F32 (or rounded to F16).
  OutT *o = O + base;
#pragma unroll
  for (int mt = 0; mt < MT; mt++) {
    float inv0 = l[mt][0] > 0.f ? 1.f / l[mt][0] : 0.f, inv1 = l[mt][1] > 0.f ? 1.f / l[mt][1] : 0.f;
    const int r0 = r[mt][0], r1 = r[mt][1];
#pragma unroll
    for (int n = 0; n < D / 8; n++) {
      int c = n * 8 + t * 2;
      if (r0 < S) fa16_store(o + r0 * ld + c, acc[mt][n][0] * inv0, acc[mt][n][1] * inv0);
      if (r1 < S) fa16_store(o + r1 * ld + c, acc[mt][n][2] * inv1, acc[mt][n][3] * inv1);
    }
  }
}

// Exported entry points: head dim 64 (Idol) or 128 (Z-Image, FLUX), BF16
// inputs with an F32 result (8 warps x 16 rows), or F16 inputs with an F16
// result (flash_attn_f16_d64_h: 4 warps x 32 rows; at head dim 128 that
// many accumulators spill). kernels/cuda/flash_attn_bf16.go
// launches each with its block's rows and threads.
#define FA16_ENTRY(name, D, STAGES, F16, MT, WARPS, OutT) \
  extern "C" __global__ void __launch_bounds__(WARPS * 32) \
  name(const fa16_b16 *Q, const fa16_b16 *K, const fa16_b16 *V, OutT *O, int S, int Sk, int H, int seq_major, \
       float scale, int causal) { \
    fa16_body<D, STAGES, F16, MT, WARPS>(Q, K, V, O, S, Sk, H, seq_major, scale, causal); \
  }

FA16_ENTRY(flash_attn_bf16_d64, 64, 2, false, 1, 8, float)
FA16_ENTRY(flash_attn_bf16_d128, 128, 1, false, 1, 8, float)
FA16_ENTRY(flash_attn_f16_d64_h, 64, 2, true, 2, 4, __half)
