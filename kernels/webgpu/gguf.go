//go:build !nowebgpu

package webgpu

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/internal/wgpudev"
	"light/tensor"
)

// GGUF block formats with WebGPU kernels: id (the shader's QTYPE, ggml's
// numbering), elements and bytes per block. Q4_K_M / Q5_K_M files mix
// these (small models fall back to Q5_0 / Q8_0 where a row is not a
// multiple of 256).
var ggufFormats = map[dtype.DType]struct{ id, be, bb int }{
	dtype.Q4_0: {2, 32, 18},
	dtype.Q5_0: {6, 32, 22},
	dtype.Q8_0: {8, 32, 34},
	dtype.Q4_K: {12, 256, 144},
	dtype.Q6_K: {14, 256, 210},
}

func init() {
	for dt := range ggufFormats {
		registry.Register(registry.Key{Op: "matmul_q", Dev: device.KindWebGPU, DType: dt}, matmulQ)
		registry.Register(registry.Key{Op: "dequant", Dev: device.KindWebGPU, DType: dt}, dequantQ)
		registry.Register(registry.Key{Op: "index_select", Dev: device.KindWebGPU, DType: dt}, indexSelectQ)
	}
}

// ggufShader is the WGSL for one format: its constants, the shared
// decoder, then the entry point.
func ggufShader(dt dtype.DType, entry string) (name, code string) {
	f := ggufFormats[dt]
	return entry + "_" + dt.String(), fmt.Sprintf("const QTYPE: u32 = %du;\nconst BE: u32 = %du;\nconst BB: u32 = %du;\n", f.id, f.be, f.bb) +
		shader("gguf_common") + shader(entry)
}

// dequantRows writes rows of w (all N in order, or the rows named by ids)
// as F32 into out at element offset off.
func dequantRows(d *wgpudev.Device, w, ids, out *tensor.Tensor, rows, K int) error {
	mode, idsBuf := uint32(0), raw(w) // the ids slot must be bound; any buffer does when unused
	if ids != nil {
		ids, err := contig(ids)
		if err != nil {
			return err
		}
		switch ids.DType() {
		case dtype.I64:
			mode = 1
		case dtype.I32:
			mode = 2
		default:
			return fmt.Errorf("webgpu gguf rows: ids must be I64 or I32, got %v", ids.DType())
		}
		idsBuf = raw(ids)
	}
	params, err := d.Uniform(uint32(rows), uint32(K), mode, 0)
	if err != nil {
		return err
	}
	defer params.Release()
	name, code := ggufShader(w.DType(), "gguf_dequant")
	x, y := d.Grid1D(uint64(rows*K), wgSize)
	return d.Run(name, code, []wgpudev.Binding{rd(raw(w)), rd(idsBuf), rw(raw(out)), uni(params)}, x, y, 1)
}

// dequantQ: quant.Dequantize on WebGPU (a quantized (N, K) tensor → F32).
func dequantQ(ctx *registry.Context) error {
	w, out := ctx.Inputs[0], ctx.Output
	w, err := contig(w)
	if err != nil {
		return err
	}
	d, err := dev()
	if err != nil {
		return err
	}
	K := w.Dim(-1)
	return dequantRows(d, w, nil, out, w.NumElements()/K, K)
}

// indexSelectQ gathers whole rows of a quantized table (the quantized
// embedding lookup; quant.Dequantize then decodes just those rows). The
// gather moves 32-bit words, so a row's block bytes must be a multiple of 4.
func indexSelectQ(ctx *registry.Context) error {
	w, ids, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	rowBytes := w.DType().ByteSize(w.Dim(-1))
	if rowBytes%4 != 0 {
		return fmt.Errorf("webgpu index_select %v: rows of %d bytes are not word-aligned", w.DType(), rowBytes)
	}
	var idxStride uint32 = 2
	if ids.DType() == dtype.I32 || ids.DType() == dtype.U32 {
		idxStride = 1
	}
	w, err := contig(w)
	if err != nil {
		return err
	}
	if ids, err = contig(ids); err != nil {
		return err
	}
	d, err := dev()
	if err != nil {
		return err
	}
	params, err := d.Uniform(uint32(ids.Dim(0)), uint32(rowBytes/4), idxStride, 0)
	if err != nil {
		return err
	}
	defer params.Release()
	return run1D(d, "indexselect", []wgpudev.Binding{rd(raw(w)), rd(raw(ids)), rw(raw(out)), uni(params)}, ids.Dim(0)*rowBytes/4)
}

// ggufGEMVMaxM is MAXM in gguf_gemv.wgsl.
const ggufGEMVMaxM = 8

// matmulQ: x (M, K) F32 times the quantized weight (N, K) transposed. Up to
// 8 rows run the fused GEMV, decoding the weight as it is read; more rows
// (prefill) dequantize the weight once and use the F32 matmul.
func matmulQ(ctx *registry.Context) error {
	x, w, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	M, K, N := x.Dim(0), x.Dim(1), w.Dim(0)
	d, err := dev()
	if err != nil {
		return err
	}
	if x, err = contig(x); err != nil {
		return err
	}
	if w, err = contig(w); err != nil {
		return err
	}
	if M <= ggufGEMVMaxM {
		params, err := d.Uniform(uint32(M), uint32(N), uint32(K), 0)
		if err != nil {
			return err
		}
		defer params.Release()
		name, code := ggufShader(w.DType(), "gguf_gemv")
		gx, gy := d.Rows(uint64(N))
		return d.Run(name, code, []wgpudev.Binding{rd(raw(w)), rd(raw(x)), rw(raw(out)), uni(params)}, gx, gy, 1)
	}
	dq, err := tensor.Zeros(dtype.F32, []int{N, K}, x.Device())
	if err != nil {
		return err
	}
	if err := dequantRows(d, w, nil, dq, N, K); err != nil {
		return err
	}
	dqT, err := dq.Transpose(0, 1)
	if err != nil {
		return err
	}
	return matmulF32(&registry.Context{Inputs: []*tensor.Tensor{x, dqT}, Output: out})
}
