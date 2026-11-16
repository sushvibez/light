package ops

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/quant"
	"light/tensor"
)

// DequantizeInt4 expands a canonical 4-bit weight (quant.Int4, from GPTQ /
// AWQ checkpoints) to an (N, K) tensor at outDT (F32 or BF16):
// W[n, k] = s[n, k/G] · (code(n, k) − z[n, k/G]), formed in F32, rounded once.
func DequantizeInt4(w quant.Int4, outDT dtype.DType) (*tensor.Tensor, error) {
	out, err := tensor.Zeros(outDT, []int{w.N(), w.K()}, w.Codes.Device())
	if err != nil {
		return nil, err
	}
	if err := dequantizeInt4Into(w, out); err != nil {
		return nil, err
	}
	return out, nil
}

func dequantizeInt4Into(w quant.Int4, out *tensor.Tensor) error {
	return dequantizeInt4At(w, out, 0)
}

// dequantizeInt4At writes the expanded weight into out's storage starting
// at element offset `off` (out must be a contiguous offset-0 buffer large
// enough); used to fill one slot of a stacked expert buffer.
func dequantizeInt4At(w quant.Int4, out *tensor.Tensor, off int) error {
	if w.Group <= 0 || w.K()%w.Group != 0 {
		return fmt.Errorf("ops.DequantizeInt4: group %d does not divide K %d", w.Group, w.K())
	}
	k, err := registry.Lookup(registry.Key{Op: "int4_dequant", Dev: w.Codes.Device().Kind(), DType: out.DType()})
	if err != nil {
		return err
	}
	codes, err := w.Codes.Contiguous()
	if err != nil {
		return err
	}
	s, err := w.Scales.Contiguous()
	if err != nil {
		return err
	}
	z, err := w.Zeros.Contiguous()
	if err != nil {
		return err
	}
	return k(&registry.Context{Inputs: []*tensor.Tensor{codes, s, z}, Output: out,
		Attrs: map[string]any{"group": w.Group, "offset": off}})
}

// int4GEMVMaxM is INT4_GEMV_MAXM in kernels/cuda/src/int4.cu.
const int4GEMVMaxM = 8

// MatMulInt4 computes x · Wᵀ for a 4-bit weight w: x is (M, K) F32 or BF16,
// the result (M, N) at x's dtype. Up to 8 rows (decode) on CUDA it runs a
// fused GEMV reading the packed codes directly (half a byte per weight);
// otherwise the weight is expanded into the shared scratch buffer
// (see fp8Scratch) and multiplied by the regular GEMM.
func MatMulInt4(x *tensor.Tensor, w quant.Int4) (*tensor.Tensor, error) {
	if x.Rank() != 2 || x.Dim(1) != w.K() {
		return nil, fmt.Errorf("ops.MatMulInt4: x %v, weight (%d, %d)", x.Shape(), w.N(), w.K())
	}
	if x.DType() != dtype.F32 && x.DType() != dtype.BF16 {
		return nil, fmt.Errorf("ops.MatMulInt4: activations %v, want F32 or BF16", x.DType())
	}
	M, N := x.Dim(0), w.N()
	xc, err := x.Contiguous()
	if err != nil {
		return nil, err
	}
	if w.Perm != nil {
		// Act-order GPTQ: the weight's K axis is stored sorted by group;
		// gather x's columns into the same order (x[:, Perm]).
		if xc, err = gatherColumns(xc, w.Perm); err != nil {
			return nil, err
		}
	}
	// The fused kernel: decode GEMV for M ≤ 8 on GPUs; on the CPU it is the
	// row-decode + SIMD dot kernel for any M.
	if (M <= int4GEMVMaxM || device.IsCPU(x.Device())) && w.K()%32 == 0 && w.Group%32 == 0 {
		if k, err := registry.Lookup(registry.Key{Op: "int4_gemv", Dev: x.Device().Kind(), DType: x.DType()}); err == nil {
			out, err := tensor.Zeros(x.DType(), []int{M, N}, x.Device())
			if err != nil {
				return nil, err
			}
			if err := k(&registry.Context{Inputs: []*tensor.Tensor{xc, w.Codes, w.Scales, w.Zeros}, Output: out,
				Attrs: map[string]any{"group": w.Group}}); err != nil {
				return nil, err
			}
			return out, nil
		}
	}
	gemm := func(dq *tensor.Tensor) (*tensor.Tensor, error) {
		wT, err := dq.Transpose(0, 1)
		if err != nil {
			return nil, err
		}
		if x.DType() == dtype.BF16 {
			return MatMulMixed(xc, wT)
		}
		return MatMul(xc, wT)
	}
	if x.Device().Kind() != device.KindCPU {
		fp8ScratchMu.Lock()
		defer fp8ScratchMu.Unlock()
		dq, err := weightScratch(x.Device(), x.DType(), N, w.K())
		if err != nil {
			return nil, err
		}
		if err := dequantizeInt4Into(w, dq); err != nil {
			return nil, err
		}
		return gemm(dq)
	}
	dq, err := DequantizeInt4(w, x.DType())
	if err != nil {
		return nil, err
	}
	return gemm(dq)
}

// gatherColumns returns x[:, idx] for a rank-2 x and an I64 index vector.
func gatherColumns(x, idx *tensor.Tensor) (*tensor.Tensor, error) {
	xt, err := x.Transpose(0, 1)
	if err != nil {
		return nil, err
	}
	if xt, err = xt.Contiguous(); err != nil {
		return nil, err
	}
	g, err := IndexSelect(xt, idx)
	if err != nil {
		return nil, err
	}
	gt, err := g.Transpose(0, 1)
	if err != nil {
		return nil, err
	}
	return gt.Contiguous()
}

// MoEInt4 runs a routed gated-MLP expert block whose experts are 4-bit
// (GPTQ / AWQ MoE checkpoints): x (T, H), per expert gateUp[e] (2I, H) —
// gate rows then up rows — and down[e] (H, I), the routing from the
// model's router, and act for the gate. Only the experts the routing uses
// are expanded, into the shared weight scratch (slots 1 and 2, under
// fp8ScratchMu, as for 4-bit / FP8 linears), and the regular grouped
// MoEGemm runs on the compact stacks with the routing remapped to them.
// Returns (T, H) at x's dtype.
func MoEInt4(x *tensor.Tensor, gateUp, down []quant.Int4, route *MoERouting, topk int,
	act func(*tensor.Tensor) (*tensor.Tensor, error)) (*tensor.Tensor, error) {
	if x.Rank() != 2 || len(gateUp) == 0 || len(gateUp) != len(down) {
		return nil, fmt.Errorf("ops.MoEInt4: x %v, %d gate-up / %d down experts", x.Shape(), len(gateUp), len(down))
	}
	T, H := x.Dim(0), x.Dim(1)
	I2, I := gateUp[0].N(), down[0].K()
	if gateUp[0].K() != H || down[0].N() != H || I2 != 2*I {
		return nil, fmt.Errorf("ops.MoEInt4: expert shapes gate-up (%d, %d), down (%d, %d) for hidden %d",
			I2, gateUp[0].K(), down[0].N(), I, H)
	}
	ids, err := route.TopKIDs.To(device.CPU())
	if err != nil {
		return nil, err
	}
	if ids, err = ids.Contiguous(); err != nil {
		return nil, err
	}
	ws, err := route.Weights.To(device.CPU())
	if err != nil {
		return nil, err
	}
	if ws, err = ws.Contiguous(); err != nil {
		return nil, err
	}
	idv, wv := tensor.MustSlice[int64](ids), tensor.MustSlice[float32](ws)
	slot := map[int64]int{}
	var used []int64
	compact := make([]int64, len(idv))
	for i, e := range idv {
		if e < 0 || int(e) >= len(gateUp) {
			return nil, fmt.Errorf("ops.MoEInt4: expert %d out of range", e)
		}
		s, ok := slot[e]
		if !ok {
			s = len(used)
			slot[e] = s
			used = append(used, e)
		}
		compact[i] = int64(s)
	}
	if device.IsCPU(x.Device()) && x.DType() == dtype.F32 && int4Fused(gateUp[used[0]]) && int4Fused(down[used[0]]) {
		return moeInt4Rows(x, gateUp, down, used, idv, wv, topk, act)
	}
	U, dt, dev := len(used), x.DType(), x.Device()
	fp8ScratchMu.Lock()
	defer fp8ScratchMu.Unlock()
	guBuf, err := weightScratchSlot(dev, dt, 1, U*I2, H)
	if err != nil {
		return nil, err
	}
	dnBuf, err := weightScratchSlot(dev, dt, 2, U*H, I)
	if err != nil {
		return nil, err
	}
	for s, e := range used {
		if err := dequantizeInt4At(gateUp[e], guBuf, s*I2*H); err != nil {
			return nil, err
		}
		if err := dequantizeInt4At(down[e], dnBuf, s*H*I); err != nil {
			return nil, err
		}
	}
	guStack, err := guBuf.Reshape(U, I2, H)
	if err != nil {
		return nil, err
	}
	dnStack, err := dnBuf.Reshape(U, H, I)
	if err != nil {
		return nil, err
	}
	r, err := MoERouteFromIDs(compact, wv, T, topk, dev)
	if err != nil {
		return nil, err
	}
	xc, err := x.Contiguous()
	if err != nil {
		return nil, err
	}
	gu, err := MoEGemm(xc, guStack, r, nil, topk) // (M, 2I)
	if err != nil {
		return nil, err
	}
	g, err := gu.Narrow(1, 0, I)
	if err != nil {
		return nil, err
	}
	u, err := gu.Narrow(1, I, I)
	if err != nil {
		return nil, err
	}
	if g, err = g.Contiguous(); err != nil {
		return nil, err
	}
	if g, err = act(g); err != nil {
		return nil, err
	}
	h, err := Mul(g, u)
	if err != nil {
		return nil, err
	}
	pair, err := MoEGemm(h, dnStack, r, r.Weights, topk)
	if err != nil {
		return nil, err
	}
	return MoECombine(pair, topk)
}

// int4Fused reports whether MatMulInt4 takes w through the fused
// row-decode kernel on the CPU (no act-order permutation, 32-aligned).
func int4Fused(w quant.Int4) bool {
	return w.Perm == nil && w.K()%32 == 0 && w.Group%32 == 0
}

// moeInt4Rows is MoEInt4 on the CPU with F32 activations: each used expert
// multiplies the rows routed to it through MatMulInt4's fused kernel, which
// decodes the packed codes row by row (half a byte per weight read, every
// core busy) instead of expanding every picked expert to F32 first. The
// weighted pairs are summed per token in routing order, as MoECombine does.
func moeInt4Rows(x *tensor.Tensor, gateUp, down []quant.Int4, used, idv []int64, wv []float32, topk int,
	act func(*tensor.Tensor) (*tensor.Tensor, error)) (*tensor.Tensor, error) {
	xc, err := x.Contiguous()
	if err != nil {
		return nil, err
	}
	xs := tensor.MustSlice[float32](xc)
	T, H, I := x.Dim(0), x.Dim(1), down[0].K()
	rows := make(map[int64][]int, len(used)) // expert → pair indices t·topk + j
	for i, e := range idv {
		rows[e] = append(rows[e], i)
	}
	pair := make([]float32, len(idv)*H)
	for _, e := range used {
		r := rows[e]
		xg := make([]float32, len(r)*H)
		for j, i := range r {
			t := i / topk
			copy(xg[j*H:(j+1)*H], xs[t*H:(t+1)*H])
		}
		xt, err := tensor.New(xg, []int{len(r), H}, x.Device())
		if err != nil {
			return nil, err
		}
		gu, err := MatMulInt4(xt, gateUp[e]) // (R, 2I)
		if err != nil {
			return nil, err
		}
		g, err := gu.Narrow(1, 0, I)
		if err != nil {
			return nil, err
		}
		u, err := gu.Narrow(1, I, I)
		if err != nil {
			return nil, err
		}
		if g, err = g.Contiguous(); err != nil {
			return nil, err
		}
		if g, err = act(g); err != nil {
			return nil, err
		}
		h, err := Mul(g, u)
		if err != nil {
			return nil, err
		}
		y, err := MatMulInt4(h, down[e]) // (R, H)
		if err != nil {
			return nil, err
		}
		ys := tensor.MustSlice[float32](y)
		for j, i := range r {
			w, dst, src := wv[i], pair[i*H:(i+1)*H], ys[j*H:(j+1)*H]
			for k := range dst {
				dst[k] = w * src[k]
			}
		}
	}
	out := make([]float32, T*H)
	for t := range T {
		o := out[t*H : (t+1)*H]
		for j := range topk {
			p := pair[(t*topk+j)*H:]
			for k := range o {
				o[k] += p[k]
			}
		}
	}
	return tensor.New(out, []int{T, H}, x.Device())
}
