//go:build (linux || android || darwin || freebsd) && amd64

#include "textflag.h"

// call6 is entered by runtime.cgocall on the g0 stack with the C calling
// convention: DI = *callArgs. It loads the six SysV integer argument
// registers from the block, lays its eight stack words out at SP, calls
// fn, and stores RAX/RDX back.
TEXT call6(SB), NOSPLIT|NOFRAME, $0
	PUSHQ BP
	MOVQ  SP, BP
	SUBQ  $80, SP          // 8 stack words + the block pointer; 16-byte aligned
	MOVQ  DI, 72(SP)
	MOVQ  DI, R11
	MOVQ  72(R11), AX      // stack words, copied whether or not the callee reads them
	MOVQ  AX, 0(SP)
	MOVQ  80(R11), AX
	MOVQ  AX, 8(SP)
	MOVQ  88(R11), AX
	MOVQ  AX, 16(SP)
	MOVQ  96(R11), AX
	MOVQ  AX, 24(SP)
	MOVQ  104(R11), AX
	MOVQ  AX, 32(SP)
	MOVQ  112(R11), AX
	MOVQ  AX, 40(SP)
	MOVQ  120(R11), AX
	MOVQ  AX, 48(SP)
	MOVQ  128(R11), AX
	MOVQ  AX, 56(SP)
	MOVQ  136(R11), X0     // the eight vector registers, from the float words
	MOVQ  144(R11), X1
	MOVQ  152(R11), X2
	MOVQ  160(R11), X3
	MOVQ  168(R11), X4
	MOVQ  176(R11), X5
	MOVQ  184(R11), X6
	MOVQ  192(R11), X7
	MOVQ  0(R11), R10      // fn
	MOVQ  8(R11), DI       // a1
	MOVQ  16(R11), SI      // a2
	MOVQ  24(R11), DX      // a3
	MOVQ  32(R11), CX      // a4
	MOVQ  40(R11), R8      // a5
	MOVQ  48(R11), R9      // a6
	MOVL  $8, AX           // varargs convention: an upper bound on vector registers in use
	CALL  R10
	MOVQ  72(SP), R11
	MOVQ  AX, 56(R11)      // r1
	MOVQ  DX, 64(R11)      // r2
	MOVQ  BP, SP
	POPQ  BP
	RET

GLOBL ·call6ABI0(SB), NOPTR|RODATA, $8
DATA  ·call6ABI0(SB)/8, $call6(SB)
