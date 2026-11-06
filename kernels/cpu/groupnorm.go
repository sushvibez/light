package cpu

import (
	"fmt"
	"math"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func init() {
	registry.Register(registry.Key{Op: "groupnorm", Dev: device.KindCPU, DType: dtype.F32}, groupnormCPUF32)
	registry.Register(registry.Key{Op: "groupnorm", Dev: device.KindCPU, DType: dtype.F64}, groupnormCPUF64)
}

// groupnormCPUF32 normalises (N, C, H, W) F32 inputs by groups: for each
// (n, g) we compute mean and variance over the (C/G) * H * W elements that
// fall in that group, then apply (x - μ) / σ * γ + β with per-channel γ/β.
//
// This is what Stable Diffusion's UNet/VAE call after every Conv2d.
func groupnormCPUF32(ctx *registry.Context) error {
	x, gamma, beta, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	G, _ := ctx.Attrs["groups"].(int)
	eps, _ := ctx.Attrs["eps"].(float64)
	if x.Rank() != 4 {
		return fmt.Errorf("cpu groupnorm: input rank %d != 4", x.Rank())
	}
	N, C := x.Dim(0), x.Dim(1)
	HW := x.Dim(2) * x.Dim(3)
	if G <= 0 || C%G != 0 {
		return fmt.Errorf("cpu groupnorm: C (%d) not divisible by groups (%d)", C, G)
	}
	Cg := C / G
	M := Cg * HW
	invM := 1.0 / float64(M)
	xBuf := unsafeF32(x.Storage().CPUBytes())
	gBuf := unsafeF32(gamma.Storage().CPUBytes())
	bBuf := unsafeF32(beta.Storage().CPUBytes())
	oBuf := unsafeF32(out.Storage().CPUBytes())
	for n := range N {
		for g := range G {
			// Pass 1: mean / var.
			var sum, sumSq float64
			for cInG := range Cg {
				c := g*Cg + cInG
				rowOff := (n*C + c) * HW
				for k := 0; k < HW; k++ {
					v := float64(xBuf[rowOff+k])
					sum += v
					sumSq += v * v
				}
			}
			m := sum * invM
			variance := sumSq*invM - m*m
			if variance < 0 {
				variance = 0
			}
			invSigma := 1.0 / math.Sqrt(variance+eps)
			// Pass 2: apply.
			mF, isF := float32(m), float32(invSigma)
			for cInG := range Cg {
				c := g*Cg + cInG
				gv, bv := gBuf[c], bBuf[c]
				rowOff := (n*C + c) * HW
				for k := 0; k < HW; k++ {
					xh := (xBuf[rowOff+k] - mF) * isF
					oBuf[rowOff+k] = xh*gv + bv
				}
			}
		}
	}
	return nil
}

func groupnormCPUF64(ctx *registry.Context) error {
	x, gamma, beta, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	G, _ := ctx.Attrs["groups"].(int)
	eps, _ := ctx.Attrs["eps"].(float64)
	if x.Rank() != 4 {
		return fmt.Errorf("cpu groupnorm: input rank %d != 4", x.Rank())
	}
	N, C := x.Dim(0), x.Dim(1)
	HW := x.Dim(2) * x.Dim(3)
	if G <= 0 || C%G != 0 {
		return fmt.Errorf("cpu groupnorm: C (%d) not divisible by groups (%d)", C, G)
	}
	Cg := C / G
	M := Cg * HW
	invM := 1.0 / float64(M)
	xBuf := unsafeF64(x.Storage().CPUBytes())
	gBuf := unsafeF64(gamma.Storage().CPUBytes())
	bBuf := unsafeF64(beta.Storage().CPUBytes())
	oBuf := unsafeF64(out.Storage().CPUBytes())
	for n := range N {
		for g := range G {
			var sum, sumSq float64
			for cInG := range Cg {
				c := g*Cg + cInG
				rowOff := (n*C + c) * HW
				for k := 0; k < HW; k++ {
					v := xBuf[rowOff+k]
					sum += v
					sumSq += v * v
				}
			}
			m := sum * invM
			variance := sumSq*invM - m*m
			if variance < 0 {
				variance = 0
			}
			invSigma := 1.0 / math.Sqrt(variance+eps)
			for cInG := range Cg {
				c := g*Cg + cInG
				gv, bv := gBuf[c], bBuf[c]
				rowOff := (n*C + c) * HW
				for k := 0; k < HW; k++ {
					xh := (xBuf[rowOff+k] - m) * invSigma
					oBuf[rowOff+k] = xh*gv + bv
				}
			}
		}
	}
	return nil
}

func unsafeF32(b []byte) []float32 {
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}
func unsafeF64(b []byte) []float64 {
	return unsafe.Slice((*float64)(unsafe.Pointer(&b[0])), len(b)/8)
}
