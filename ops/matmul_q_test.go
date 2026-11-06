package ops_test

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/quant"
	"light/tensor"
)

// randomQuantBlocks builds a valid random (N, K) weight of the given GGUF
// dtype: random quant bytes, with the F16 scale fields forced into a small
// finite range so no block is NaN/Inf.
func randomQuantBlocks(rng *rand.Rand, dt dtype.DType, N, K int) *tensor.Tensor {
	be, bb := dt.BlockElems(), dt.BlockBytes()
	nb := N * K / be
	buf := make([]byte, nb*bb)
	for i := range buf {
		buf[i] = byte(rng.IntN(256))
	}
	small := func(b []byte) { // f16 in [2^-6, 2^-5) ≈ 0.0156..0.031
		binary.LittleEndian.PutUint16(b, 0x2400|uint16(rng.IntN(0x400)))
	}
	for b := 0; b < nb; b++ {
		blk := buf[b*bb : (b+1)*bb]
		switch dt {
		case dtype.Q4_0, dtype.Q5_0, dtype.Q8_0:
			small(blk[0:2])
		case dtype.Q4_1, dtype.Q5_1, dtype.Q8_1:
			small(blk[0:2])
			small(blk[2:4])
		case dtype.Q4_K, dtype.Q5_K:
			small(blk[0:2])
			small(blk[2:4])
		case dtype.Q2_K:
			small(blk[80:82])
			small(blk[82:84])
		case dtype.Q3_K:
			small(blk[108:110])
		case dtype.Q6_K:
			small(blk[208:210])
		case dtype.Q8_K:
			binary.LittleEndian.PutUint32(blk[0:4], math.Float32bits(0.02))
		}
	}
	t, err := tensor.FromBytes(buf, dt, []int{N, K}, device.CPU())
	if err != nil {
		panic(err)
	}
	return t
}

var matmulQTypes = []dtype.DType{dtype.Q4_0, dtype.Q5_0, dtype.Q8_0, dtype.Q4_K, dtype.Q5_K, dtype.Q6_K,
	dtype.Q4_1, dtype.Q5_1, dtype.Q2_K, dtype.Q3_K}

func maxRelErr(a, b []float32) float64 {
	var num, den float64
	for i := range a {
		num = math.Max(num, math.Abs(float64(a[i]-b[i])))
		den = math.Max(den, math.Abs(float64(b[i])))
	}
	if den == 0 {
		return num
	}
	return num / den
}

// On CPU MatMulQ is dequant + matmul by construction; this pins the
// reference used by the CUDA test and the block-size validation.
func TestMatMulQCPUMatchesDequant(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 9))
	const M, N, K = 5, 7, 512
	xd := make([]float32, M*K)
	for i := range xd {
		xd[i] = rng.Float32()*2 - 1
	}
	x, _ := tensor.New(xd, []int{M, K}, device.CPU())
	for _, dt := range matmulQTypes {
		w := randomQuantBlocks(rng, dt, N, K)
		got, err := ops.MatMulQ(x, w)
		if err != nil {
			t.Fatalf("%v: %v", dt, err)
		}
		wf, _ := quant.Dequantize(w)
		wT, _ := wf.Transpose(0, 1)
		want, _ := ops.MatMul(x, wT)
		if e := maxRelErr(tensor.MustSlice[float32](got), tensor.MustSlice[float32](want)); e > 1e-5 {
			t.Errorf("%v: max rel err %g", dt, e)
		}
	}
	if _, err := ops.MatMulQ(x, randomQuantBlocks(rng, dtype.Q4_0, N, 512)); err != nil {
		t.Fatal(err)
	}
	bad, _ := tensor.FromBytes(make([]byte, 18*N), dtype.Q4_0, []int{N, 32}, device.CPU())
	if _, err := ops.MatMulQ(x, bad); err == nil {
		t.Error("expected K mismatch error")
	}
}

func TestCUDAMatMulQParity(t *testing.T) {
	if !device.CUDAAvailable() {
		t.Skip("CUDA not available")
	}
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skip(err)
	}
	rng := rand.New(rand.NewPCG(5, 6))
	// M covers the 4-row tile plus a tail; K covers multiple blocks per lane
	// (>32 blocks of 32) and a non-multiple of 32 blocks for the K-quants.
	for _, M := range []int{1, 4, 9} {
		for _, K := range []int{1024, 1792} {
			const N = 13
			xd := make([]float32, M*K)
			for i := range xd {
				xd[i] = rng.Float32()*2 - 1
			}
			x, _ := tensor.New(xd, []int{M, K}, device.CPU())
			xG, _ := x.To(gpu)
			for _, dt := range matmulQTypes {
				if K%dt.BlockElems() != 0 {
					continue
				}
				w := randomQuantBlocks(rng, dt, N, K)
				want, err := ops.MatMulQ(x, w)
				if err != nil {
					t.Fatalf("%v cpu: %v", dt, err)
				}
				wG, _ := w.To(gpu)
				got, err := ops.MatMulQ(xG, wG)
				if err != nil {
					t.Fatalf("%v cuda: %v", dt, err)
				}
				gotC, _ := got.To(device.CPU())
				if e := maxRelErr(tensor.MustSlice[float32](gotC), tensor.MustSlice[float32](want)); e > 2e-4 {
					t.Errorf("%v M=%d K=%d: CUDA vs CPU max rel err %g", dt, M, K, e)
				}
			}
		}
	}
}
