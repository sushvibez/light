//go:build !nocuda && cgo

package ops_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// CPU vs CUDA parity for rope_scaled, F32 and BF16, with a non-trivial
// (scaled) frequency table and a batch of rows sharing positions.
func TestCUDARoPEWithInvFreqParity(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	rng := rand.New(rand.NewPCG(7, 9))
	const B, H, S, D = 2, 3, 6, 16
	data := make([]float32, B*H*S*D)
	for i := range data {
		data[i] = rng.Float32()*2 - 1
	}
	inv := baseInvFreq(D, 500000)
	for i := range inv {
		if i >= D/4 {
			inv[i] /= 32 // Llama-3-style low-frequency scaling
		}
	}
	x, _ := tensor.New(data, []int{B, H, S, D}, cpu)
	pos, _ := tensor.New([]int64{10, 11, 12, 13, 14, 15}, []int{S}, cpu)
	invT, _ := tensor.New(inv, []int{D / 2}, cpu)
	cpuOut, err := ops.RoPEWithInvFreq(x, pos, invT)
	if err != nil {
		t.Fatal(err)
	}
	cs := tensor.MustSlice[float32](cpuOut)

	xG, _ := x.To(gpu)
	pG, _ := pos.To(gpu)
	iG, _ := invT.To(gpu)
	gpuOut, err := ops.RoPEWithInvFreq(xG, pG, iG)
	if err != nil {
		t.Fatal(err)
	}
	back, _ := gpuOut.To(cpu)
	gs := tensor.MustSlice[float32](back)
	for i := range cs {
		if math.Abs(float64(cs[i]-gs[i])) > 1e-4 {
			t.Fatalf("F32 [%d] CPU=%v GPU=%v", i, cs[i], gs[i])
		}
	}

	// BF16 path: compare against the F32 CPU result at BF16 tolerance.
	xB, err := ops.Cast(xG, dtype.BF16)
	if err != nil {
		t.Fatal(err)
	}
	outB, err := ops.RoPEWithInvFreq(xB, pG, iG)
	if err != nil {
		t.Fatal(err)
	}
	if outB.DType() != dtype.BF16 {
		t.Fatalf("BF16 output dtype = %v", outB.DType())
	}
	outF, err := ops.Cast(outB, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	backB, _ := outF.To(cpu)
	bs := tensor.MustSlice[float32](backB)
	for i := range cs {
		if math.Abs(float64(cs[i]-bs[i])) > 2e-2 {
			t.Fatalf("BF16 [%d] CPU=%v GPU=%v", i, cs[i], bs[i])
		}
	}
}
