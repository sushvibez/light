// FlashAttention-2 forward on BF16 tensor cores (sm_80+), head dim 128.
//
// Layout: Q, K, V are BF16 and O is F32, each either head-major (B, H, S, D)
// (seq_major 0: one batch-head is a contiguous S*D slab) or the projections'
// own (B, S, H, D) (seq_major 1: a head's rows are H*D apart), so callers
// need no permute copies either way. A block owns FA16_BQ = 128 query rows
// of one batch-head (8 warps x 16 rows) and streams the keys in tiles of
// FA16_BK = 64 through shared memory. Each warp keeps its 16 query rows'
// A fragments, the online-softmax state (running max m, denominator l) and
// a 16 x 128 F32 accumulator in registers; S = Q K^T and O += P V are
// mma.sync.m16n8k16 BF16 products with F32 accumulation. The m16n8k16
// fragment layouts (PTX ISA, "Matrix Fragments for mma.m16n8k16") give
// each thread rows lane/4 and lane/4 + 8, so the softmax rescale is per
// register, and a score tile's accumulator is exactly the A fragment of
// the P V product (FlashAttention-2's register reuse).
//
// scale multiplies the scores; causal masks key j > query i.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define FA16_D 128
#define FA16_BQ 128
#define FA16_BK 64
#define FA16_WARPS 8
#define FA16_PAD 8                      // bf16 per smem row of padding (bank conflicts)
#define FA16_LD (FA16_D + FA16_PAD)     // smem row stride in bf16

__device__ __forceinline__ void fa16_mma(float *c, const unsigned *a, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned fa16_pack(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi); // .x = lo (low 16 bits)
  return *reinterpret_cast<unsigned *>(&v);
}

__device__ __forceinline__ unsigned fa16_pair(const __nv_bfloat16 *lo, const __nv_bfloat16 *hi) {
  unsigned short a = *reinterpret_cast<const unsigned short *>(lo);
  unsigned short b = *reinterpret_cast<const unsigned short *>(hi);
  return (unsigned)a | ((unsigned)b << 16);
}

// Stage rows [row0, row0 + FA16_BK) of a (S, D) BF16 slab into smem
// (FA16_LD stride), zero-filling rows past S. 16-byte loads.
__device__ __forceinline__ void fa16_stage(__nv_bfloat16 *dst, const __nv_bfloat16 *src, int row0, int S, long ld) {
  const int chunks = FA16_D / 8; // uint4 per row
  for (int c = threadIdx.x; c < FA16_BK * chunks; c += blockDim.x) {
    int r = c / chunks, k = (c % chunks) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (row0 + r < S) v = *reinterpret_cast<const uint4 *>(src + (row0 + r) * ld + k);
    *reinterpret_cast<uint4 *>(dst + r * FA16_LD + k) = v;
  }
}

extern "C" __global__ void __launch_bounds__(FA16_WARPS * 32)
flash_attn_bf16_d128(const __nv_bfloat16 *Q, const __nv_bfloat16 *K, const __nv_bfloat16 *V, float *O,
                     int S, int H, int seq_major, float scale, int causal) {
  __shared__ __align__(16) __nv_bfloat16 sK[FA16_BK * FA16_LD];
  __shared__ __align__(16) __nv_bfloat16 sV[FA16_BK * FA16_LD];

  const int bh = blockIdx.y;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int g = lane / 4, t = lane % 4; // fragment row group, thread in group
  // Row stride and this batch-head's first element, in either layout.
  const long ld = seq_major ? (long)H * FA16_D : FA16_D;
  const long base = seq_major ? ((long)(bh / H) * S * H + bh % H) * FA16_D : (long)bh * S * FA16_D;
  const int q0 = blockIdx.x * FA16_BQ + warp * 16; // this warp's first query row
  const int r0 = q0 + g, r1 = q0 + g + 8;          // the two rows this thread owns

  // Q A-fragments for the 8 k-steps of 16 over D, straight from global.
  unsigned qa[FA16_D / 16][4];
  {
    const __nv_bfloat16 *q = Q + base;
#pragma unroll
    for (int ks = 0; ks < FA16_D / 16; ks++) {
      int c = ks * 16 + t * 2;
      const __nv_bfloat16 *p0 = r0 < S ? q + r0 * ld : nullptr;
      const __nv_bfloat16 *p1 = r1 < S ? q + r1 * ld : nullptr;
      qa[ks][0] = p0 ? *reinterpret_cast<const unsigned *>(p0 + c) : 0u;
      qa[ks][1] = p1 ? *reinterpret_cast<const unsigned *>(p1 + c) : 0u;
      qa[ks][2] = p0 ? *reinterpret_cast<const unsigned *>(p0 + c + 8) : 0u;
      qa[ks][3] = p1 ? *reinterpret_cast<const unsigned *>(p1 + c + 8) : 0u;
    }
  }

  float acc[FA16_D / 8][4]; // 16 n-tiles of 8 dims: c0,c1 row r0; c2,c3 row r1
#pragma unroll
  for (int n = 0; n < FA16_D / 8; n++) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const float sl2 = scale * 1.4426950408889634f; // scores in log2 units

  int kEnd = S;
  if (causal) kEnd = min(S, (int)(blockIdx.x + 1) * FA16_BQ);
  for (int k0 = 0; k0 < kEnd; k0 += FA16_BK) {
    __syncthreads(); // the previous tile is no longer read
    fa16_stage(sK, K + base, k0, S, ld);
    fa16_stage(sV, V + base, k0, S, ld);
    __syncthreads();

    // S = Q K^T for this warp's 16 rows x 64 keys: 8 n-tiles of 8 keys.
    float s[FA16_BK / 8][4];
#pragma unroll
    for (int n = 0; n < FA16_BK / 8; n++) {
      s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
      const __nv_bfloat16 *krow = sK + (n * 8 + g) * FA16_LD + t * 2; // B(k, n) = K[key n][d k]
#pragma unroll
      for (int ks = 0; ks < FA16_D / 16; ks++) {
        unsigned b0 = *reinterpret_cast<const unsigned *>(krow + ks * 16);
        unsigned b1 = *reinterpret_cast<const unsigned *>(krow + ks * 16 + 8);
        fa16_mma(s[n], qa[ks], b0, b1);
      }
    }

    // Mask keys past S (and above the diagonal when causal), then the
    // online softmax over the thread's two rows (reduced over the 4 lanes
    // that share a row).
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int n = 0; n < FA16_BK / 8; n++) {
#pragma unroll
      for (int e = 0; e < 2; e++) {
        int key = k0 + n * 8 + t * 2 + e;
        bool ok0 = key < S && (!causal || key <= r0);
        bool ok1 = key < S && (!causal || key <= r1);
        s[n][e] = ok0 ? s[n][e] * sl2 : -INFINITY;
        s[n][2 + e] = ok1 ? s[n][2 + e] * sl2 : -INFINITY;
        mx0 = fmaxf(mx0, s[n][e]);
        mx1 = fmaxf(mx1, s[n][2 + e]);
      }
    }
#pragma unroll
    for (int o = 1; o < 4; o <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, o));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, o));
    }
    float nm0 = fmaxf(m0, mx0), nm1 = fmaxf(m1, mx1);
    // A row with every key masked so far keeps m = -inf: use 0 so exp2 gives 0, not NaN.
    float sub0 = nm0 == -INFINITY ? 0.f : nm0, sub1 = nm1 == -INFINITY ? 0.f : nm1;
    float al0 = exp2f(m0 - sub0), al1 = exp2f(m1 - sub1);
    m0 = nm0;
    m1 = nm1;
    float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
    for (int n = 0; n < FA16_BK / 8; n++) {
#pragma unroll
      for (int e = 0; e < 2; e++) {
        s[n][e] = exp2f(s[n][e] - sub0);
        s[n][2 + e] = exp2f(s[n][2 + e] - sub1);
        rs0 += s[n][e];
        rs1 += s[n][2 + e];
      }
    }
#pragma unroll
    for (int o = 1; o < 4; o <<= 1) {
      rs0 += __shfl_xor_sync(0xffffffff, rs0, o);
      rs1 += __shfl_xor_sync(0xffffffff, rs1, o);
    }
    l0 = l0 * al0 + rs0;
    l1 = l1 * al1 + rs1;
#pragma unroll
    for (int n = 0; n < FA16_D / 8; n++) {
      acc[n][0] *= al0;
      acc[n][1] *= al0;
      acc[n][2] *= al1;
      acc[n][3] *= al1;
    }

    // O += P V: P's k-step j (keys 16j..16j+15) is score tiles 2j, 2j+1.
#pragma unroll
    for (int j = 0; j < FA16_BK / 16; j++) {
      unsigned pa[4];
      pa[0] = fa16_pack(s[2 * j][0], s[2 * j][1]);
      pa[1] = fa16_pack(s[2 * j][2], s[2 * j][3]);
      pa[2] = fa16_pack(s[2 * j + 1][0], s[2 * j + 1][1]);
      pa[3] = fa16_pack(s[2 * j + 1][2], s[2 * j + 1][3]);
      // B(k, n) = V[key 16j + k][dim n]: two keys per register.
      const __nv_bfloat16 *v0 = sV + (16 * j + t * 2) * FA16_LD + g;
#pragma unroll
      for (int n = 0; n < FA16_D / 8; n++) {
        const __nv_bfloat16 *v = v0 + n * 8;
        unsigned b0 = fa16_pair(v, v + FA16_LD);
        unsigned b1 = fa16_pair(v + 8 * FA16_LD, v + 9 * FA16_LD);
        fa16_mma(acc[n], pa, b0, b1);
      }
    }
  }

  // O = acc / l, F32.
  float inv0 = l0 > 0.f ? 1.f / l0 : 0.f, inv1 = l1 > 0.f ? 1.f / l1 : 0.f;
  float *o = O + base;
#pragma unroll
  for (int n = 0; n < FA16_D / 8; n++) {
    int c = n * 8 + t * 2;
    if (r0 < S) *reinterpret_cast<float2 *>(o + r0 * ld + c) = make_float2(acc[n][0] * inv0, acc[n][1] * inv0);
    if (r1 < S) *reinterpret_cast<float2 *>(o + r1 * ld + c) = make_float2(acc[n][2] * inv1, acc[n][3] * inv1);
  }
}
