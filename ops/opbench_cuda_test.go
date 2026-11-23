//go:build !nocuda && cgo

package ops_test

import (
	"fmt"
	"math/rand/v2"
	"os"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/nn"
	"light/ops"
	"light/quant"
	"light/storage"
	"light/tensor"
)

// TestOpBenchCUDA times single ops at FLUX.1's 1024² geometry (4096 image
// tokens, width 3072, 24 heads of 128) for docs/image-bench.md, next to
// scripts/verify/opbench_torch.py and scripts/verify/opbench_candle.
//
//	LIGHT_OPBENCH=1 go test ./ops -run TestOpBenchCUDA -v
func TestOpBenchCUDA(t *testing.T) {
	if os.Getenv("LIGHT_OPBENCH") == "" {
		t.Skip("set LIGHT_OPBENCH=1")
	}
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skip(err)
	}
	rng := rand.New(rand.NewPCG(1, 2))
	rnd := func(dev device.Device, shape ...int) *tensor.Tensor {
		n := 1
		for _, d := range shape {
			n *= d
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64()) * 0.1
		}
		x, err := tensor.New(v, shape, dev)
		if err != nil {
			t.Fatal(err)
		}
		return x
	}
	must := func(x *tensor.Tensor, err error) *tensor.Tensor {
		if err != nil {
			t.Fatal(err)
		}
		return x
	}
	var rows []string
	only := os.Getenv("LIGHT_OPBENCH_ONLY") // comma-separated substrings of row names
	bench := func(name string, flops, bytes float64, f func() error) {
		if only != "" {
			hit := false
			for _, o := range strings.Split(only, ",") {
				hit = hit || strings.Contains(name, o)
			}
			if !hit {
				return
			}
		}
		for range 3 {
			if err := f(); err != nil {
				t.Errorf("%s: %v", name, err)
				rows = append(rows, fmt.Sprintf("| %s | failed: %v | |", name, err))
				return
			}
		}
		// Warm the allocator: a pool miss is a cudaMalloc (~80 µs for a
		// 53 MB block), which would be a quarter of a sub-millisecond op;
		// outputs return to the pool only at a collection.
		for range 6 {
			m0 := storage.CUDAPoolStats().Mallocs
			for range 4 {
				_ = f()
			}
			runtime.GC()
			time.Sleep(5 * time.Millisecond)
			if storage.CUDAPoolStats().Mallocs == m0 {
				break
			}
		}
		_ = cudart.DeviceSynchronize()
		// Median of individually synchronized calls (a forced collection
		// inside the window otherwise lands on one op's mean).
		const iters = 21
		ds := make([]float64, iters)
		host := make([]float64, iters) // time to the call's return, before the device finishes
		m0 := storage.CUDAPoolStats().Mallocs
		for i := range ds {
			if i%3 == 0 { // return the outputs to the pool outside the timed call
				runtime.GC()
				time.Sleep(3 * time.Millisecond)
				_ = cudart.DeviceSynchronize()
			}
			t0 := time.Now()
			_ = f()
			host[i] = time.Since(t0).Seconds() * 1e3
			_ = cudart.DeviceSynchronize()
			ds[i] = time.Since(t0).Seconds() * 1e3
		}
		sort.Float64s(ds)
		ms := ds[iters/2]
		if n := storage.CUDAPoolStats().Mallocs - m0; n > iters/4 {
			t.Logf("%s: %d device mallocs inside the timed calls", name, n)
		}
		rate := ""
		if flops > 0 {
			rate = fmt.Sprintf("%.1f TFLOPS", flops/(ms*1e-3)/1e12)
		} else if bytes > 0 {
			rate = fmt.Sprintf("%.0f GB/s", bytes/(ms*1e-3)/1e9)
		}
		row := fmt.Sprintf("| %s | %.2f ms | %s |", name, ms, rate)
		if os.Getenv("LIGHT_OPBENCH_SPREAD") != "" {
			st := storage.CUDAPoolStats()
			sort.Float64s(host)
			t.Logf("%s: host-side median %.3f ms", name, host[iters/2])
			t.Logf("%s: min %.2f, median %.2f, max %.2f ms; %d mallocs in the timed calls; pool %+v", name, ds[0], ms, ds[iters-1], st.Mallocs-m0, st)
		}
		t.Log(row)
		rows = append(rows, row)
	}
	const M, K, H, Dh = 4096, 3072, 24, 128
	x := rnd(gpu, M, K)
	xb := must(ops.Cast(x, dtype.BF16))
	for _, N := range []int{3072, 12288} {
		w := rnd(gpu, N, K) // (N, K) as a Linear stores it
		wT := must(must(w.Transpose(0, 1)).Contiguous())
		wb := must(ops.Cast(w, dtype.BF16))
		wbT := must(wb.Transpose(0, 1)) // the transposed view of the (N, K) weight, as nn.Linear passes it
		w8 := must(ops.Cast(w, dtype.F8E4M3))
		fl := 2 * float64(M) * float64(K) * float64(N)
		tag := fmt.Sprintf("(%d x %d) · (%d x %d)ᵀ", M, K, N, K)
		bench("GEMM F32 "+tag, fl, 0, func() error { _, err := ops.MatMul(x, wT); return err })
		bench("GEMM BF16 in, F32 out (mixed) "+tag, fl, 0, func() error { _, err := ops.MatMulMixed(x, wbT); return err })
		bench("GEMM BF16 in, BF16 out "+tag, fl, 0, func() error { _, err := ops.MatMulMixed(xb, wbT); return err })
		bench("GEMM F32 x, BF16 weight exact (W16) "+tag, fl, 0, func() error { _, err := ops.MatMulW16(x, wb); return err })
		bench("FP8 weight → BF16 + GEMM "+tag, fl, 0, func() error {
			wide, err := ops.Cast(w8, dtype.BF16)
			if err != nil {
				return err
			}
			wideT, err := wide.Transpose(0, 1)
			if err != nil {
				return err
			}
			_, err = ops.MatMulMixed(x, wideT)
			return err
		})
		if N == 3072 {
			wc := must(w.To(device.CPU()))
			for _, q := range []dtype.DType{dtype.Q8_0, dtype.Q4_0} {
				wq := must(must(quant.Quantize(wc, q)).To(gpu))
				bench(fmt.Sprintf("GGUF %v weight, dequant + GEMM %s", q, tag), fl, 0, func() error { _, err := ops.MatMulQ(x, wq); return err })
				bench(fmt.Sprintf("GGUF %v weight, dequant → BF16 + GEMM %s", q, tag), fl, 0, func() error { _, err := ops.MatMulQMixed(x, wq); return err })
			}
		}
	}
	S := 4096 + 256
	q, k, v := rnd(gpu, 1, H, S, Dh), rnd(gpu, 1, H, S, Dh), rnd(gpu, 1, H, S, Dh)
	qb, kb, vb := must(ops.Cast(q, dtype.BF16)), must(ops.Cast(k, dtype.BF16)), must(ops.Cast(v, dtype.BF16))
	afl := 4 * float64(H) * float64(S) * float64(S) * float64(Dh)
	atag := fmt.Sprintf("(1, %d, %d, %d)", H, S, Dh)
	bench("Attention F32 "+atag, afl, 0, func() error { _, err := nn.SDPA(q, k, v, nil); return err })
	bench("Attention F32, flash kernel "+atag, afl, 0, func() error { _, err := ops.FlashAttention(q, k, v, false); return err })
	bench("Attention BF16 "+atag, afl, 0, func() error { _, err := ops.FlashAttentionBF16(qb, kb, vb, false); return err })
	nn.SDPAFlashF32 = false
	bench("Attention F32, generic path (nn.SDPA) "+atag, afl, 0, func() error { _, err := nn.SDPA(q, k, v, nil); return err })
	nn.SDPAFlashF32 = true
	for _, s2 := range []int{1280, 256, 64} { // 512² image, short sequences
		q2, k2, v2 := rnd(gpu, 1, H, s2, Dh), rnd(gpu, 1, H, s2, Dh), rnd(gpu, 1, H, s2, Dh)
		f2 := 4 * float64(H) * float64(s2) * float64(s2) * float64(Dh)
		bench(fmt.Sprintf("Attention F32 flash (1, %d, %d, %d)", H, s2, Dh), f2, 0, func() error { _, err := ops.FlashAttention(q2, k2, v2, false); return err })
		bench(fmt.Sprintf("Attention F32 batched GEMM (1, %d, %d, %d)", H, s2, Dh), f2, 0, func() error { _, err := ops.AttentionGEMM(q2, k2, v2); return err })
	}
	h := rnd(gpu, 1, S, K)
	ones, zeros := must(tensor.Ones(dtype.F32, []int{K}, gpu)), must(tensor.Zeros(dtype.F32, []int{K}, gpu))
	by := float64(S*K) * 4 * 2 // read + write
	ntag := fmt.Sprintf("(1, %d, %d)", S, K)
	bench("LayerNorm F32 "+ntag, 0, by, func() error { _, err := ops.LayerNorm(h, ones, zeros, 1e-6); return err })
	hq := rnd(gpu, 1, S, H, Dh)
	wq := must(tensor.Ones(dtype.F32, []int{Dh}, gpu))
	bench(fmt.Sprintf("RMSNorm F32 (1, %d, %d, %d)", S, H, Dh), 0, by, func() error { _, err := ops.RMSNorm(hq, wq, 1e-6); return err })
	g := rnd(gpu, 1, S, 4*K)
	bench(fmt.Sprintf("GELU(tanh) F32 (1, %d, %d)", S, 4*K), 0, float64(S*4*K)*8, func() error { _, err := ops.Gelu(g); return err })
	bench("SiLU F32 "+ntag, 0, by, func() error { _, err := ops.SiLU(h); return err })
	bench("Add F32 "+ntag, 0, float64(S*K)*12, func() error { _, err := ops.Add(h, h); return err })
	sm := rnd(gpu, H, 1024, 1024)
	bench("Softmax F32 (24, 1024, 1024)", 0, float64(H*1024*1024)*8, func() error { _, err := ops.Softmax(sm, -1); return err })
	img := rnd(gpu, 1, 128, 512, 512)
	cw := rnd(gpu, 128, 128, 3, 3)
	cfl := 2 * 128.0 * 128 * 9 * 512 * 512
	bench("Conv2d 3x3 F32 (1, 128, 512, 512) → 128", cfl, 0, func() error { _, err := ops.Conv2d(img, cw, nil, [2]int{1, 1}, [2]int{1, 1}); return err })
	imgb, cwb := must(ops.Cast(img, dtype.BF16)), must(ops.Cast(cw, dtype.BF16))
	bench("Conv2d 3x3 BF16 (1, 128, 512, 512) → 128", cfl, 0, func() error { _, err := ops.Conv2d(imgb, cwb, nil, [2]int{1, 1}, [2]int{1, 1}); return err })
	t.Logf("\n| Op | Light | rate |\n|---|---|---|\n%s", strings.Join(rows, "\n"))
}
