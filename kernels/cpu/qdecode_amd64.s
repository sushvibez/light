//go:build amd64

#include "textflag.h"

// AVX2 decoders for quantized weight rows (see qdecode_amd64.go). Each
// produces exactly the floats of the Go decoders in quant_matmul.go: the
// integer code is built per lane, converted to float32, then scaled with
// the same operations in the same order.

DATA lanes0<>+0(SB)/4, $0
DATA lanes0<>+4(SB)/4, $1
DATA lanes0<>+8(SB)/4, $2
DATA lanes0<>+12(SB)/4, $3
DATA lanes0<>+16(SB)/4, $4
DATA lanes0<>+20(SB)/4, $5
DATA lanes0<>+24(SB)/4, $6
DATA lanes0<>+28(SB)/4, $7
GLOBL lanes0<>(SB), RODATA, $32

DATA nib4<>+0(SB)/4, $0
DATA nib4<>+4(SB)/4, $4
DATA nib4<>+8(SB)/4, $8
DATA nib4<>+12(SB)/4, $12
DATA nib4<>+16(SB)/4, $16
DATA nib4<>+20(SB)/4, $20
DATA nib4<>+24(SB)/4, $24
DATA nib4<>+28(SB)/4, $28
GLOBL nib4<>(SB), RODATA, $32

#define BCAST(val, reg) MOVL $val, AX; VMOVD AX, X0; VPBROADCASTD X0, reg

// func q8_0RowAVX2(raw *byte, d *float32, dst *float32, nb int)
// 34-byte blocks: f16 d (pre-converted into d[]), 32 × int8.
TEXT ·q8_0RowAVX2(SB), NOSPLIT, $0-32
	MOVQ raw+0(FP), SI
	MOVQ d+8(FP), DX
	MOVQ dst+16(FP), DI
	MOVQ nb+24(FP), CX
q8loop:
	TESTQ CX, CX
	JZ    q8done
	VBROADCASTSS (DX), Y13
	VPMOVSXBD 2(SI), Y1
	VPMOVSXBD 10(SI), Y2
	VPMOVSXBD 18(SI), Y3
	VPMOVSXBD 26(SI), Y4
	VCVTDQ2PS Y1, Y1
	VCVTDQ2PS Y2, Y2
	VCVTDQ2PS Y3, Y3
	VCVTDQ2PS Y4, Y4
	VMULPS Y1, Y13, Y1
	VMULPS Y2, Y13, Y2
	VMULPS Y3, Y13, Y3
	VMULPS Y4, Y13, Y4
	VMOVUPS Y1, (DI)
	VMOVUPS Y2, 32(DI)
	VMOVUPS Y3, 64(DI)
	VMOVUPS Y4, 96(DI)
	ADDQ $34, SI
	ADDQ $4, DX
	ADDQ $128, DI
	DECQ CX
	JMP  q8loop
q8done:
	VZEROUPPER
	RET

// Q4_0 / Q5_0 half: nibbles of 8 bytes at off(SI) → lanes lo (elements
// j0..j0+7) and hi (16+j0..), minus the offset in Y14, times d (Y13).
#define Q4HALF(off, dstlo, dsthi) \
	VPMOVZXBD off(SI), Y1; \
	VPAND     Y15, Y1, Y2; \
	VPSRLD    $4, Y1, Y3; \
	VPSUBD    Y14, Y2, Y2; \
	VPSUBD    Y14, Y3, Y3; \
	VCVTDQ2PS Y2, Y2; \
	VCVTDQ2PS Y3, Y3; \
	VMULPS    Y2, Y13, Y2; \
	VMULPS    Y3, Y13, Y3; \
	VMOVUPS   Y2, dstlo(DI); \
	VMOVUPS   Y3, dsthi(DI)

// func q4_0RowAVX2(raw *byte, d *float32, dst *float32, nb int)
// 18-byte blocks: f16 d, 16 bytes of nibbles (low: 0..15, high: 16..31), offset 8.
TEXT ·q4_0RowAVX2(SB), NOSPLIT, $0-32
	MOVQ raw+0(FP), SI
	MOVQ d+8(FP), DX
	MOVQ dst+16(FP), DI
	MOVQ nb+24(FP), CX
	BCAST(15, Y15)
	BCAST(8, Y14)
q4loop:
	TESTQ CX, CX
	JZ    q4done
	VBROADCASTSS (DX), Y13
	Q4HALF(2, 0, 64)
	Q4HALF(10, 32, 96)
	ADDQ $18, SI
	ADDQ $4, DX
	ADDQ $128, DI
	DECQ CX
	JMP  q4loop
q4done:
	VZEROUPPER
	RET

// Q5_0 half: as Q4HALF, with the fifth bit of each element from qh (Y4):
// bit (j0 + lane) for lo, (16 + j0 + lane) for hi; shift vectors in shlo / shhi.
#define Q5HALF(off, shlo, shhi, dstlo, dsthi) \
	VPMOVZXBD off(SI), Y1; \
	VPAND     Y15, Y1, Y2; \
	VPSRLD    $4, Y1, Y3; \
	VPSRLVD   shlo, Y4, Y5; \
	VPSRLVD   shhi, Y4, Y6; \
	VPAND     Y12, Y5, Y5; \
	VPAND     Y12, Y6, Y6; \
	VPSLLD    $4, Y5, Y5; \
	VPSLLD    $4, Y6, Y6; \
	VPOR      Y5, Y2, Y2; \
	VPOR      Y6, Y3, Y3; \
	VPSUBD    Y14, Y2, Y2; \
	VPSUBD    Y14, Y3, Y3; \
	VCVTDQ2PS Y2, Y2; \
	VCVTDQ2PS Y3, Y3; \
	VMULPS    Y2, Y13, Y2; \
	VMULPS    Y3, Y13, Y3; \
	VMOVUPS   Y2, dstlo(DI); \
	VMOVUPS   Y3, dsthi(DI)

// func q5_0RowAVX2(raw *byte, d *float32, dst *float32, nb int)
// 22-byte blocks: f16 d, u32 qh, 16 bytes of nibbles, offset 16.
TEXT ·q5_0RowAVX2(SB), NOSPLIT, $0-32
	MOVQ raw+0(FP), SI
	MOVQ d+8(FP), DX
	MOVQ dst+16(FP), DI
	MOVQ nb+24(FP), CX
	BCAST(15, Y15)
	BCAST(16, Y14)
	BCAST(1, Y12)
	VMOVDQU lanes0<>(SB), Y8 // shifts 0..7
	BCAST(8, Y0)
	VPADDD  Y0, Y8, Y9       // 8..15
	BCAST(16, Y0)
	VPADDD  Y0, Y8, Y10      // 16..23
	BCAST(24, Y0)
	VPADDD  Y0, Y8, Y11      // 24..31
q5loop:
	TESTQ CX, CX
	JZ    q5done
	VBROADCASTSS (DX), Y13
	VPBROADCASTD 2(SI), Y4
	Q5HALF(6, Y8, Y10, 0, 64)
	Q5HALF(14, Y9, Y11, 32, 96)
	ADDQ $22, SI
	ADDQ $4, DX
	ADDQ $128, DI
	DECQ CX
	JMP  q5loop
q5done:
	VZEROUPPER
	RET

// Q4_K chunk: nibbles of 8 bytes at off(SI); lo = q·Y13 − Y12 at dlo,
// hi = q·Y11 − Y10 at dhi.
#define Q4KCHUNK(off, dlo, dhi) \
	VPMOVZXBD off(SI), Y1; \
	VPAND     Y15, Y1, Y2; \
	VPSRLD    $4, Y1, Y3; \
	VCVTDQ2PS Y2, Y2; \
	VCVTDQ2PS Y3, Y3; \
	VMULPS    Y2, Y13, Y2; \
	VMULPS    Y3, Y11, Y3; \
	VSUBPS    Y12, Y2, Y2; \
	VSUBPS    Y10, Y3, Y3; \
	VMOVUPS   Y2, dlo(DI); \
	VMOVUPS   Y3, dhi(DI)

// func q4KBlockAVX2(qs *byte, sc *float32, mn *float32, dst *float32)
// One 256-element super-block: qs[128] nibbles; sc[j] = d·scale_j and
// mn[j] = dmin·min_j for the 8 sub-blocks (unpacked in Go).
TEXT ·q4KBlockAVX2(SB), NOSPLIT, $0-32
	MOVQ qs+0(FP), SI
	MOVQ sc+8(FP), DX
	MOVQ mn+16(FP), BX
	MOVQ dst+24(FP), DI
	BCAST(15, Y15)
	MOVQ $4, CX
q4kloop:
	VBROADCASTSS (DX), Y13
	VBROADCASTSS (BX), Y12
	VBROADCASTSS 4(DX), Y11
	VBROADCASTSS 4(BX), Y10
	Q4KCHUNK(0, 0, 128)
	Q4KCHUNK(8, 32, 160)
	Q4KCHUNK(16, 64, 192)
	Q4KCHUNK(24, 96, 224)
	ADDQ $32, SI
	ADDQ $8, DX
	ADDQ $8, BX
	ADDQ $256, DI
	DECQ CX
	JNZ  q4kloop
	VZEROUPPER
	RET

// Q6_K quarter: q = (nib) | (((H >> hs) & 3) << 4) − 32, times the scale at
// soff(R8), stored at doff(DI). nib is in Y2 on entry.
#define Q6Q(hs, soff, doff) \
	VPSRLD    $hs, Y7, Y5; \
	VPAND     Y14, Y5, Y5; \
	VPSLLD    $4, Y5, Y5; \
	VPOR      Y5, Y2, Y2; \
	VPSUBD    Y13, Y2, Y2; \
	VCVTDQ2PS Y2, Y2; \
	VBROADCASTSS soff(R8), Y6; \
	VMULPS    Y2, Y6, Y2; \
	VMOVUPS   Y2, doff(DI)

// Q6_K chunk of 8 l-positions: byte offsets of ql[l0], ql[l0+32], qh[l0];
// scale byte offsets for the four quarters; output byte offsets.
#define Q6CHUNK(qa, qb, hq, s0, s1, s2, s3, d0, d1, d2, d3) 	VPMOVZXBD qa(SI), Y3; 	VPMOVZXBD qb(SI), Y4; 	VPMOVZXBD hq(BX), Y7; 	VPAND     Y15, Y3, Y2; 	Q6Q(0, s0, d0); 	VPAND     Y15, Y4, Y2; 	Q6Q(2, s1, d1); 	VPSRLD    $4, Y3, Y2; 	Q6Q(4, s2, d2); 	VPSRLD    $4, Y4, Y2; 	Q6Q(6, s3, d3)

// func q6KBlockAVX2(ql *byte, qh *byte, sc *float32, dst *float32)
// One 256-element super-block: ql[128], qh[64]; sc[16] = d·int8(scale).
TEXT ·q6KBlockAVX2(SB), NOSPLIT, $0-32
	MOVQ ql+0(FP), SI
	MOVQ qh+8(FP), BX
	MOVQ sc+16(FP), R8
	MOVQ dst+24(FP), DI
	BCAST(15, Y15)
	BCAST(3, Y14)
	BCAST(32, Y13)
	MOVQ $2, CX
q6kloop:
	Q6CHUNK(0, 32, 0, 0, 8, 16, 24, 0, 128, 256, 384)
	Q6CHUNK(8, 40, 8, 0, 8, 16, 24, 32, 160, 288, 416)
	Q6CHUNK(16, 48, 16, 4, 12, 20, 28, 64, 192, 320, 448)
	Q6CHUNK(24, 56, 24, 4, 12, 20, 28, 96, 224, 352, 480)
	ADDQ $64, SI
	ADDQ $32, BX
	ADDQ $32, R8
	ADDQ $512, DI
	DECQ CX
	JNZ  q6kloop
	VZEROUPPER
	RET

// func int4RowAVX2(codes *uint32, s *float32, z *float32, dst *float32, groups, wpg int)
// GPTQ / AWQ canonical rows: word w holds elements 8w..8w+7 (nibble j =
// element 8w+j); value s_g · (q − z_g); wpg words per group.
TEXT ·int4RowAVX2(SB), NOSPLIT, $0-48
	MOVQ codes+0(FP), SI
	MOVQ s+8(FP), DX
	MOVQ z+16(FP), BX
	MOVQ dst+24(FP), DI
	MOVQ groups+32(FP), CX
	MOVQ wpg+40(FP), R9
	BCAST(15, Y15)
	VMOVDQU nib4<>(SB), Y14
i4group:
	TESTQ CX, CX
	JZ    i4done
	VBROADCASTSS (DX), Y13
	VBROADCASTSS (BX), Y12
	MOVQ R9, R10
i4word:
	VPBROADCASTD (SI), Y1
	VPSRLVD   Y14, Y1, Y1
	VPAND     Y15, Y1, Y1
	VCVTDQ2PS Y1, Y1
	VSUBPS    Y12, Y1, Y1
	VMULPS    Y1, Y13, Y1
	VMOVUPS   Y1, (DI)
	ADDQ $4, SI
	ADDQ $32, DI
	DECQ R10
	JNZ  i4word
	ADDQ $4, DX
	ADDQ $4, BX
	DECQ CX
	JMP  i4group
i4done:
	VZEROUPPER
	RET

// func bf16RowAVX2(src *uint16, dst *float32, n int)
// BF16 → F32 (the bits shifted into the high half); n a multiple of 8.
TEXT ·bf16RowAVX2(SB), NOSPLIT, $0-24
	MOVQ src+0(FP), SI
	MOVQ dst+8(FP), DI
	MOVQ n+16(FP), CX
bf16loop:
	CMPQ CX, $16
	JL   bf16tail
	VPMOVZXWD (SI), Y1
	VPMOVZXWD 16(SI), Y2
	VPSLLD    $16, Y1, Y1
	VPSLLD    $16, Y2, Y2
	VMOVUPS   Y1, (DI)
	VMOVUPS   Y2, 32(DI)
	ADDQ $32, SI
	ADDQ $64, DI
	SUBQ $16, CX
	JMP  bf16loop
bf16tail:
	CMPQ CX, $8
	JL   bf16done
	VPMOVZXWD (SI), Y1
	VPSLLD    $16, Y1, Y1
	VMOVUPS   Y1, (DI)
bf16done:
	VZEROUPPER
	RET

// func f16RowF16C(src *uint16, dst *float32, n int)
// F16 → F32 (exact) with VCVTPH2PS; n a multiple of 8.
TEXT ·f16RowF16C(SB), NOSPLIT, $0-24
	MOVQ src+0(FP), SI
	MOVQ dst+8(FP), DI
	MOVQ n+16(FP), CX
f16loop:
	CMPQ CX, $16
	JL   f16tail
	VCVTPH2PS (SI), Y1
	VCVTPH2PS 16(SI), Y2
	VMOVUPS   Y1, (DI)
	VMOVUPS   Y2, 32(DI)
	ADDQ $32, SI
	ADDQ $64, DI
	SUBQ $16, CX
	JMP  f16loop
f16tail:
	CMPQ CX, $8
	JL   f16done
	VCVTPH2PS (SI), Y1
	VMOVUPS   Y1, (DI)
f16done:
	VZEROUPPER
	RET
