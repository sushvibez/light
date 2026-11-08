//go:build amd64

#include "textflag.h"

// func dotAVX2(a, b *float32, n int) float32
TEXT ·dotAVX2(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), SI
	MOVQ b+8(FP), DI
	MOVQ n+16(FP), CX
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
dot32:
	CMPQ CX, $32
	JL   dot8
	VMOVUPS (SI), Y4
	VMOVUPS 32(SI), Y5
	VMOVUPS 64(SI), Y6
	VMOVUPS 96(SI), Y7
	VFMADD231PS (DI), Y4, Y0
	VFMADD231PS 32(DI), Y5, Y1
	VFMADD231PS 64(DI), Y6, Y2
	VFMADD231PS 96(DI), Y7, Y3
	ADDQ $128, SI
	ADDQ $128, DI
	SUBQ $32, CX
	JMP  dot32
dot8:
	CMPQ CX, $8
	JL   dotreduce
	VMOVUPS (SI), Y4
	VFMADD231PS (DI), Y4, Y0
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  dot8
dotreduce:
	VADDPS Y1, Y0, Y0
	VADDPS Y3, Y2, Y2
	VADDPS Y2, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
dottail:
	CMPQ CX, $0
	JE   dotdone
	VMOVSS (SI), X1
	VFMADD231SS (DI), X1, X0
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JMP  dottail
dotdone:
	VZEROUPPER
	MOVSS X0, ret+24(FP)
	RET

// func axpyAVX2(alpha float32, x, y *float32, n int)
TEXT ·axpyAVX2(SB), NOSPLIT, $0-32
	VBROADCASTSS alpha+0(FP), Y0
	MOVQ x+8(FP), SI
	MOVQ y+16(FP), DI
	MOVQ n+24(FP), CX
axpy32:
	CMPQ CX, $32
	JL   axpy8
	VMOVUPS (DI), Y1
	VMOVUPS 32(DI), Y2
	VMOVUPS 64(DI), Y3
	VMOVUPS 96(DI), Y4
	VFMADD231PS (SI), Y0, Y1
	VFMADD231PS 32(SI), Y0, Y2
	VFMADD231PS 64(SI), Y0, Y3
	VFMADD231PS 96(SI), Y0, Y4
	VMOVUPS Y1, (DI)
	VMOVUPS Y2, 32(DI)
	VMOVUPS Y3, 64(DI)
	VMOVUPS Y4, 96(DI)
	ADDQ $128, SI
	ADDQ $128, DI
	SUBQ $32, CX
	JMP  axpy32
axpy8:
	CMPQ CX, $8
	JL   axpytail
	VMOVUPS (DI), Y1
	VFMADD231PS (SI), Y0, Y1
	VMOVUPS Y1, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  axpy8
axpytail:
	CMPQ CX, $0
	JE   axpydone
	VMOVSS (DI), X1
	VFMADD231SS (SI), X0, X1
	VMOVSS X1, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JMP  axpytail
axpydone:
	VZEROUPPER
	RET

// func cpuid(leaf, sub uint32) (eax, ebx, ecx, edx uint32)
TEXT ·cpuid(SB), NOSPLIT, $0-24
	MOVL leaf+0(FP), AX
	MOVL sub+4(FP), CX
	CPUID
	MOVL AX, eax+8(FP)
	MOVL BX, ebx+12(FP)
	MOVL CX, ecx+16(FP)
	MOVL DX, edx+20(FP)
	RET

// func xgetbv() (eax, edx uint32)
TEXT ·xgetbv(SB), NOSPLIT, $0-8
	MOVL $0, CX
	XGETBV
	MOVL AX, eax+0(FP)
	MOVL DX, edx+4(FP)
	RET

// Horizontal sum of Y(acc) into the float32 at off(BX); clobbers Y15.
#define HSUM(acc, xacc, off) \
	VEXTRACTF128 $1, acc, X15; \
	VADDPS X15, xacc, xacc; \
	VHADDPS xacc, xacc, xacc; \
	VHADDPS xacc, xacc, xacc; \
	VMOVSS xacc, off(BX)

// One A row (pointer reg) against the three W vectors in Y12..Y14.
#define ROW3(areg, c0, c1, c2) \
	VMOVUPS (areg), Y15; \
	VFMADD231PS Y12, Y15, c0; \
	VFMADD231PS Y13, Y15, c1; \
	VFMADD231PS Y14, Y15, c2

// func dot4x3AVX2(a *float32, lda int, w *float32, ldw int, k int, out *float32)
// out[3·i + j] = Σ_t a[i·lda + t] · w[j·ldw + t] for i < 4, j < 3, t < k;
// k a multiple of 8, strides in elements.
TEXT ·dot4x3AVX2(SB), NOSPLIT, $0-48
	MOVQ a+0(FP), SI
	MOVQ lda+8(FP), AX
	SHLQ $2, AX
	MOVQ w+16(FP), DI
	MOVQ ldw+24(FP), DX
	SHLQ $2, DX
	MOVQ k+32(FP), CX
	MOVQ out+40(FP), BX
	LEAQ (SI)(AX*1), R8
	LEAQ (R8)(AX*1), R9
	LEAQ (R9)(AX*1), R10
	LEAQ (DI)(DX*1), R11
	LEAQ (R11)(DX*1), R12
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7
	VXORPS Y8, Y8, Y8
	VXORPS Y9, Y9, Y9
	VXORPS Y10, Y10, Y10
	VXORPS Y11, Y11, Y11
tile:
	TESTQ CX, CX
	JZ    tiledone
	VMOVUPS (DI), Y12
	VMOVUPS (R11), Y13
	VMOVUPS (R12), Y14
	ROW3(SI, Y0, Y1, Y2)
	ROW3(R8, Y3, Y4, Y5)
	ROW3(R9, Y6, Y7, Y8)
	ROW3(R10, Y9, Y10, Y11)
	ADDQ $32, SI
	ADDQ $32, R8
	ADDQ $32, R9
	ADDQ $32, R10
	ADDQ $32, DI
	ADDQ $32, R11
	ADDQ $32, R12
	SUBQ $8, CX
	JMP  tile
tiledone:
	HSUM(Y0, X0, 0)
	HSUM(Y1, X1, 4)
	HSUM(Y2, X2, 8)
	HSUM(Y3, X3, 12)
	HSUM(Y4, X4, 16)
	HSUM(Y5, X5, 20)
	HSUM(Y6, X6, 24)
	HSUM(Y7, X7, 28)
	HSUM(Y8, X8, 32)
	HSUM(Y9, X9, 36)
	HSUM(Y10, X10, 40)
	HSUM(Y11, X11, 44)
	VZEROUPPER
	RET
