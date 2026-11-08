//go:build (linux || android || darwin || freebsd) && arm64

#include "textflag.h"

// call6 is entered by runtime.cgocall on the g0 stack with the C calling
// convention: R0 = *callArgs. It loads X0–X5 from the block, X6/X7 and
// six stack words from the CallStack words, calls fn, and stores X0/X1
// back. The non-NOFRAME $0 frame makes the assembler save
// LR/FP around the BL, as purego's trampoline relies on too.
TEXT call6(SB), NOSPLIT, $0
	SUB  $80, RSP         // 8 stack words + the block pointer; 16-byte aligned
	MOVD R0, 72(RSP)
	MOVD R0, R9
	// AAPCS64 has eight integer argument registers: the 7th and 8th
	// arguments (stack[0], stack[1]) go in X6/X7, the rest on the stack,
	// copied whether or not the callee reads them.
	MOVD 72(R9), R6
	MOVD 80(R9), R7
	MOVD 88(R9), R10
	MOVD R10, 0(RSP)
	MOVD 96(R9), R10
	MOVD R10, 8(RSP)
	MOVD 104(R9), R10
	MOVD R10, 16(RSP)
	MOVD 112(R9), R10
	MOVD R10, 24(RSP)
	MOVD 120(R9), R10
	MOVD R10, 32(RSP)
	MOVD 128(R9), R10
	MOVD R10, 40(RSP)
	FMOVD 136(R9), F0     // the eight vector registers, from the float words
	FMOVD 144(R9), F1
	FMOVD 152(R9), F2
	FMOVD 160(R9), F3
	FMOVD 168(R9), F4
	FMOVD 176(R9), F5
	FMOVD 184(R9), F6
	FMOVD 192(R9), F7
	MOVD 0(R9), R10       // fn
	MOVD 8(R9), R0        // a1
	MOVD 16(R9), R1       // a2
	MOVD 24(R9), R2       // a3
	MOVD 32(R9), R3       // a4
	MOVD 40(R9), R4       // a5
	MOVD 48(R9), R5       // a6
	BL   (R10)
	MOVD 72(RSP), R2
	ADD  $80, RSP
	MOVD R0, 56(R2)       // r1
	MOVD R1, 64(R2)       // r2
	RET

GLOBL ·call6ABI0(SB), NOPTR|RODATA, $8
DATA  ·call6ABI0(SB)/8, $call6(SB)
