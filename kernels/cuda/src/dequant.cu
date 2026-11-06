// GGUF block-quant dequantization kernels (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 → F32).
//
// Each block encodes 32 logical elements. Kernels launch one thread per output
// element; the thread derives its block index and intra-block position, reads
// the F16 scale (and optional F16 min / 32-bit qh bitmap), and decodes its one
// element.
//
// All scalar loads from the raw byte buffer go through unsigned char* to avoid
// alignment assumptions on the device side. F16 bytes are decoded into float
// via __half_raw to stay strict-aliasing-safe.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float decode_half_le(const unsigned char *p) {
  unsigned short u = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
  __half_raw hr;
  hr.x = u;
  return __half2float((__half)hr);
}

// ---- Q4_0 (18B) ----
// bytes:  [ F16 d | qs[0..15] ]
// y[j]       = d * ((qs[j] & 0x0F) - 8)         j in [0,16)
// y[j + 16]  = d * ((qs[j] >> 4)   - 8)         j in [0,16)
extern "C" __global__ void dequant_q4_0_f32(
    const unsigned char *src, float *out, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long b = i / 32;
  unsigned j = (unsigned)(i - b * 32);
  const unsigned char *blk = src + b * 18;
  float d = decode_half_le(blk);
  int q;
  if (j < 16) q = (int)(blk[2 + j] & 0x0F) - 8;
  else        q = (int)(blk[2 + (j - 16)] >> 4) - 8;
  out[i] = d * (float)q;
}

// ---- Q4_1 (20B) ----
// bytes:  [ F16 d | F16 m | qs[0..15] ]
// y[i] = d * q4[i] + m, q4 in [0,15]
extern "C" __global__ void dequant_q4_1_f32(
    const unsigned char *src, float *out, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long b = i / 32;
  unsigned j = (unsigned)(i - b * 32);
  const unsigned char *blk = src + b * 20;
  float d = decode_half_le(blk);
  float m = decode_half_le(blk + 2);
  int q;
  if (j < 16) q = (int)(blk[4 + j] & 0x0F);
  else        q = (int)(blk[4 + (j - 16)] >> 4);
  out[i] = d * (float)q + m;
}

// ---- Q5_0 (22B) ----
// bytes:  [ F16 d | qh[4] | qs[16] ]
// q5 = (low_nibble) | (bit_from_qh << 4); y[i] = d * (q5 - 16)
extern "C" __global__ void dequant_q5_0_f32(
    const unsigned char *src, float *out, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long b = i / 32;
  unsigned j = (unsigned)(i - b * 32);
  const unsigned char *blk = src + b * 22;
  float d = decode_half_le(blk);
  unsigned int qh = (unsigned int)blk[2]
                  | ((unsigned int)blk[3] << 8)
                  | ((unsigned int)blk[4] << 16)
                  | ((unsigned int)blk[5] << 24);
  int q;
  if (j < 16) {
    unsigned lo4 = blk[6 + j] & 0x0F;
    unsigned hi5 = (qh >> j) & 1u;
    q = (int)(lo4 | (hi5 << 4)) - 16;
  } else {
    unsigned jj = j - 16;
    unsigned hi4 = blk[6 + jj] >> 4;
    unsigned hi5 = (qh >> (jj + 16)) & 1u;
    q = (int)(hi4 | (hi5 << 4)) - 16;
  }
  out[i] = d * (float)q;
}

// ---- Q5_1 (24B) ----
// bytes:  [ F16 d | F16 m | qh[4] | qs[16] ]
// q5 in [0,31]; y[i] = d * q5 + m
extern "C" __global__ void dequant_q5_1_f32(
    const unsigned char *src, float *out, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long b = i / 32;
  unsigned j = (unsigned)(i - b * 32);
  const unsigned char *blk = src + b * 24;
  float d = decode_half_le(blk);
  float m = decode_half_le(blk + 2);
  unsigned int qh = (unsigned int)blk[4]
                  | ((unsigned int)blk[5] << 8)
                  | ((unsigned int)blk[6] << 16)
                  | ((unsigned int)blk[7] << 24);
  int q;
  if (j < 16) {
    unsigned lo4 = blk[8 + j] & 0x0F;
    unsigned hi5 = (qh >> j) & 1u;
    q = (int)(lo4 | (hi5 << 4));
  } else {
    unsigned jj = j - 16;
    unsigned hi4 = blk[8 + jj] >> 4;
    unsigned hi5 = (qh >> (jj + 16)) & 1u;
    q = (int)(hi4 | (hi5 << 4));
  }
  out[i] = d * (float)q + m;
}

// ---- Q8_0 (34B) ----
// bytes:  [ F16 d | int8 qs[32] ]
// y[i] = d * qs[i]
extern "C" __global__ void dequant_q8_0_f32(
    const unsigned char *src, float *out, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long b = i / 32;
  unsigned j = (unsigned)(i - b * 32);
  const unsigned char *blk = src + b * 34;
  float d = decode_half_le(blk);
  signed char q = (signed char)blk[2 + j];
  out[i] = d * (float)q;
}

// ---- Q8_1 (36B) ----
// bytes:  [ F16 d | F16 sum_meta (unused here) | int8 qs[32] ]
// y[i] = d * qs[i]
extern "C" __global__ void dequant_q8_1_f32(
    const unsigned char *src, float *out, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long b = i / 32;
  unsigned j = (unsigned)(i - b * 32);
  const unsigned char *blk = src + b * 36;
  float d = decode_half_le(blk);
  signed char q = (signed char)blk[4 + j];
  out[i] = d * (float)q;
}

// ---- Q4_K (144B per super-block, 256 elements) ----
// bytes:  [ F16 d | F16 dmin | u8 scales[12] | u8 qs[128] ]
// 8 sub-blocks of 32. Sub-block j has a 6-bit scale sc_j and 6-bit min m_j
// unpacked from scales[12] per ggml get_scale_min_k4:
//   j<4: sc = scales[j] & 63; m = scales[j+4] & 63
//   else: sc = (scales[j+4] & 0xF) | ((scales[j-4]>>6)<<4)
//         m  = (scales[j+4] >> 4)  | ((scales[j]   >>6)<<4)
// Element layout: pair p = idx/64; r = idx%64; sub j = 2p + (r>>5); l = r&31;
// qs byte at qs[p*32+l]; nibble low if r<32 else high.
// y = d*sc_j*q4 - dmin*m_j
__device__ __forceinline__ void q4k_scale_min(
    int j, const unsigned char *scales, unsigned &sc, unsigned &m) {
  if (j < 4) {
    sc = (unsigned)(scales[j] & 63);
    m  = (unsigned)(scales[j + 4] & 63);
  } else {
    sc = (unsigned)((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4));
    m  = (unsigned)((scales[j + 4] >> 4)  | ((scales[j]     >> 6) << 4));
  }
}

extern "C" __global__ void dequant_q4_k_f32(
    const unsigned char *src, float *out, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long b = i / 256;
  unsigned idx = (unsigned)(i - b * 256);
  const unsigned char *blk = src + b * 144;
  float d = decode_half_le(blk);
  float dmin = decode_half_le(blk + 2);
  const unsigned char *scales = blk + 4;
  const unsigned char *qs = blk + 16;
  unsigned p = idx >> 6;        // 0..3
  unsigned r = idx & 63;        // 0..63
  unsigned subJ = (p << 1) + (r >> 5);
  unsigned l = r & 31;
  unsigned char byteVal = qs[p * 32 + l];
  unsigned q4 = (r < 32) ? (byteVal & 0x0F) : (byteVal >> 4);
  unsigned sc, m;
  q4k_scale_min((int)subJ, scales, sc, m);
  out[i] = d * (float)sc * (float)q4 - dmin * (float)m;
}

// ---- Q3_K (110B per super-block, 256 elements) ----
// bytes:  [ u8 hmask[32] | u8 qs[64] | u8 scales[12] | F16 d ]
// 3-bit quants: low2 from qs via shift=2j, high bit from hmask[bit=half].
//   q = low2 - (hmask_bit ? 0 : 4)   → signed [-4, 3]
// Scales: 16 signed 6-bit values packed per ggml Q3_K shuffle into 12 bytes.
// y = d * (scale_raw - 32) * q.
__device__ __forceinline__ int q3k_scale(int is, const unsigned char *packed) {
  // Unpack one of 16 scales from the 12-byte packed layout.
  // Reference: ggml's uint32 shuffle
  //   aux[0..2] = input bytes 0..11 as uint32s (LE)
  //   then two combined fields:
  //     low4(out[0..3])  = aux[0] low4 per byte
  //     low4(out[4..7])  = aux[1] low4 per byte
  //     low4(out[8..11]) = (aux[0] >> 4) low4 per byte
  //     low4(out[12..15])= (aux[1] >> 4) low4 per byte
  //     high2(out[0..3]) = aux[2] low2 per byte
  //     high2(out[4..7]) = (aux[2] >> 2) low2 per byte
  //     high2(out[8..11])= (aux[2] >> 4) low2 per byte
  //     high2(out[12..15])=(aux[2] >> 6) low2 per byte
  int group = is >> 2;           // 0..3
  int within = is & 3;            // 0..3
  unsigned a, b;
  if (group == 0) {
    a = packed[within];              // low-4 in aux[0]
    b = packed[8 + within];          // low-2 in aux[2]
  } else if (group == 1) {
    a = packed[4 + within];          // low-4 in aux[1]
    b = packed[8 + within] >> 2;
  } else if (group == 2) {
    a = packed[within] >> 4;         // high-4 of aux[0]
    b = packed[8 + within] >> 4;
  } else {
    a = packed[4 + within] >> 4;     // high-4 of aux[1]
    b = packed[8 + within] >> 6;
  }
  int raw = (int)((a & 0x0Fu) | ((b & 0x03u) << 4));
  return raw - 32;
}

extern "C" __global__ void dequant_q3_k_f32(
    const unsigned char *src, float *out, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long b = i / 256;
  unsigned idx = (unsigned)(i - b * 256);
  const unsigned char *blk = src + b * 110;
  const unsigned char *hmask = blk;
  const unsigned char *qs = blk + 32;
  const unsigned char *scales = blk + 96;
  float d = decode_half_le(blk + 108);

  unsigned half = idx >> 7;
  unsigned wh = idx & 0x7F;
  unsigned j = wh >> 5;
  unsigned withinJ = wh & 0x1F;
  unsigned subIdx = withinJ >> 4;
  unsigned l = withinJ & 0x0F;
  unsigned shift = 2u * j;
  unsigned scIdx = half * 8u + j * 2u + subIdx;
  int sc = q3k_scale((int)scIdx, scales);

  unsigned qsByte = qs[half * 32u + subIdx * 16u + l];
  int q = (int)((qsByte >> shift) & 3u);
  unsigned hmByte = hmask[subIdx * 16u + l];
  if (((hmByte >> half) & 1u) == 0u) q -= 4;
  out[i] = d * (float)sc * (float)q;
}

// ---- Q2_K (84B per super-block, 256 elements) ----
// bytes:  [ u8 scales[16] | u8 qs[64] | F16 d | F16 dmin ]
// scales[i]: low nibble = sub-block scale; high nibble = sub-block min.
// Element layout (per ggml):
//   half = idx/128; wh = idx%128; j = wh/32; within_j = wh%32;
//   sub_index = within_j/16; l = within_j%16; shift = 2*j.
//   scale_idx = half*8 + j*2 + sub_index; qs_byte = qs[half*32+sub_index*16+l].
//   q2 = (qs_byte >> shift) & 3; y = d*sc*q2 - dmin*m.
extern "C" __global__ void dequant_q2_k_f32(
    const unsigned char *src, float *out, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long b = i / 256;
  unsigned idx = (unsigned)(i - b * 256);
  const unsigned char *blk = src + b * 84;
  const unsigned char *scales = blk;
  const unsigned char *qs = blk + 16;
  float d = decode_half_le(blk + 80);
  float dmin = decode_half_le(blk + 82);

  unsigned half = idx >> 7;
  unsigned wh = idx & 0x7F;
  unsigned j = wh >> 5;
  unsigned withinJ = wh & 0x1F;
  unsigned subIdx = withinJ >> 4;
  unsigned l = withinJ & 0x0F;
  unsigned shift = 2u * j;
  unsigned scIdx = half * 8u + j * 2u + subIdx;
  unsigned qsByte = qs[half * 32u + subIdx * 16u + l];
  unsigned q2 = (qsByte >> shift) & 3u;
  unsigned char sc = scales[scIdx];
  unsigned dlSc = sc & 0x0Fu;
  unsigned mlMin = sc >> 4;
  out[i] = d * (float)dlSc * (float)q2 - dmin * (float)mlMin;
}

// ---- Q8_K (292B per super-block, 256 elements) ----
// bytes:  [ F32 d | int8 qs[256] | int16 bsums[16] ]
// y[i] = d * qs[i]. bsums are metadata (unused for pure dequant).
extern "C" __global__ void dequant_q8_k_f32(
    const unsigned char *src, float *out, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long b = i / 256;
  unsigned idx = (unsigned)(i - b * 256);
  const unsigned char *blk = src + b * 292;
  // F32 scale — copy 4 bytes into a float via memcpy-style aliasing.
  unsigned int bits = (unsigned int)blk[0]
                   | ((unsigned int)blk[1] << 8)
                   | ((unsigned int)blk[2] << 16)
                   | ((unsigned int)blk[3] << 24);
  float d = __int_as_float((int)bits);
  signed char q = (signed char)blk[4 + idx];
  out[i] = d * (float)q;
}

// ---- Q5_K (176B per super-block, 256 elements) ----
// bytes:  [ F16 d | F16 dmin | u8 scales[12] | u8 qh[32] | u8 qs[128] ]
// Same scale/min packing as Q4_K. Each element has a 5th high bit in qh:
//   qh[l] bit (2p)   is element (p*64 + l) bit 4
//   qh[l] bit (2p+1) is element (p*64 + 32 + l) bit 4
// q5 ∈ [0, 31]; y = d*sc*q5 - dmin*m.
extern "C" __global__ void dequant_q5_k_f32(
    const unsigned char *src, float *out, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long b = i / 256;
  unsigned idx = (unsigned)(i - b * 256);
  const unsigned char *blk = src + b * 176;
  float d = decode_half_le(blk);
  float dmin = decode_half_le(blk + 2);
  const unsigned char *scales = blk + 4;
  const unsigned char *qh = blk + 16;
  const unsigned char *qs = blk + 48;

  unsigned p = idx >> 6;        // 0..3 (pair index)
  unsigned r = idx & 63;        // 0..63
  unsigned subJ = (p << 1) + (r >> 5);
  unsigned l = r & 31;
  unsigned qhBit = (r < 32) ? (2u * p) : (2u * p + 1u);
  unsigned q5 = (r < 32)
      ? (unsigned)(qs[p * 32 + l] & 0x0F)
      : (unsigned)(qs[p * 32 + l] >> 4);
  if (qh[l] & (1u << qhBit)) q5 |= 0x10u;

  unsigned sc, m;
  q4k_scale_min((int)subJ, scales, sc, m);
  out[i] = d * (float)sc * (float)q5 - dmin * (float)m;
}

// ---- Q6_K (210B per super-block, 256 elements) ----
// bytes:  [ ql[128] | qh[64] | int8 scales[16] | F16 d ]
// 256 elements split into 2 halves of 128. Within a half, four 32-element
// groups derive from: (ql_low, qh bits 0-1), (ql_high at l+32, qh bits 2-3),
// (ql_high, qh bits 4-5), (ql_high at l+32, qh bits 6-7). Raw 6-bit value is
// q6 - 32; scale is signed. y = d * scale * (raw - 32).
extern "C" __global__ void dequant_q6_k_f32(
    const unsigned char *src, float *out, unsigned long n) {
  unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long b = i / 256;
  unsigned idx = (unsigned)(i - b * 256);
  const unsigned char *blk = src + b * 210;
  const unsigned char *ql = blk;
  const unsigned char *qh = blk + 128;
  const signed char   *sc = (const signed char *)(blk + 192);
  float d = decode_half_le(blk + 208);

  unsigned half  = idx >> 7;         // 0 or 1
  unsigned wh    = idx & 0x7F;       // 0..127
  unsigned group = wh >> 5;          // 0..3
  unsigned l     = wh & 0x1F;        // 0..31
  unsigned is    = l >> 4;           // 0 or 1
  unsigned qlBase = half * 64;
  unsigned qhBase = half * 32;
  unsigned scBase = half * 8;
  unsigned qhByte = qh[qhBase + l];

  int raw;
  int scaleIdx;
  if (group == 0) {
    raw = (int)(ql[qlBase + l] & 0x0F) | ((int)((qhByte >> 0) & 3u) << 4);
    scaleIdx = (int)(scBase + 0u + is);
  } else if (group == 1) {
    raw = (int)(ql[qlBase + l + 32] & 0x0F) | ((int)((qhByte >> 2) & 3u) << 4);
    scaleIdx = (int)(scBase + 2u + is);
  } else if (group == 2) {
    raw = (int)(ql[qlBase + l] >> 4) | ((int)((qhByte >> 4) & 3u) << 4);
    scaleIdx = (int)(scBase + 4u + is);
  } else {
    raw = (int)(ql[qlBase + l + 32] >> 4) | ((int)((qhByte >> 6) & 3u) << 4);
    scaleIdx = (int)(scBase + 6u + is);
  }
  int q = raw - 32;
  out[i] = d * (float)sc[scaleIdx] * (float)q;
}
