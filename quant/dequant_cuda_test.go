//go:build !nocuda

package quant_test

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	_ "light/kernels/cuda" // register CUDA dequant hook
	"light/quant"
	"light/tensor"
)

// dequantParity takes raw block bytes plus the dtype, runs dequant on CPU and
// CUDA, and compares the F32 outputs bit-exactly (F16 scale decode is
// deterministic, so any discrepancy is a kernel bug).
func dequantParity(
	t *testing.T,
	raw []byte,
	dt dtype.DType,
	nElems int,
	dequant func(*tensor.Tensor) (*tensor.Tensor, error),
) {
	t.Helper()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}

	cpuIn, err := tensor.FromBytes(raw, dt, []int{nElems}, device.CPU())
	if err != nil {
		t.Fatalf("FromBytes CPU: %v", err)
	}
	cpuOut, err := dequant(cpuIn)
	if err != nil {
		t.Fatalf("CPU dequant: %v", err)
	}
	cpuVals := tensor.MustSlice[float32](cpuOut)

	gpuIn, err := tensor.FromBytes(raw, dt, []int{nElems}, gpu)
	if err != nil {
		t.Fatalf("FromBytes GPU: %v", err)
	}
	gpuOut, err := dequant(gpuIn)
	if err != nil {
		t.Fatalf("CUDA dequant: %v", err)
	}
	if gpuOut.Device().Kind() != device.KindCUDA {
		t.Fatal("GPU output should be on CUDA")
	}
	backCPU, err := gpuOut.To(device.CPU())
	if err != nil {
		t.Fatalf("To(CPU): %v", err)
	}
	gpuVals := tensor.MustSlice[float32](backCPU)

	if len(gpuVals) != len(cpuVals) {
		t.Fatalf("length mismatch: CPU=%d GPU=%d", len(cpuVals), len(gpuVals))
	}
	for i := range cpuVals {
		if math.Abs(float64(cpuVals[i]-gpuVals[i])) > 1e-5 {
			t.Errorf("[%d] CPU=%v GPU=%v", i, cpuVals[i], gpuVals[i])
		}
	}
}

func TestDequantQ4_0CUDAParity(t *testing.T) {
	var vals [32]int8
	for i := range vals {
		vals[i] = int8(i%16) - 8
	}
	raw := buildQ4_0Block(0.5, vals)
	dequantParity(t, raw, dtype.Q4_0, 32, quant.DequantQ4_0)
}

func TestDequantQ4_1CUDAParity(t *testing.T) {
	var vals [32]uint8
	for i := range vals {
		vals[i] = uint8(i % 16)
	}
	raw := buildQ4_1Block(0.25, -1.5, vals)
	dequantParity(t, raw, dtype.Q4_1, 32, quant.DequantQ4_1)
}

func TestDequantQ5_0CUDAParity(t *testing.T) {
	var vals [32]int8
	for i := range vals {
		vals[i] = int8(i%32) - 16
	}
	raw := buildQ5_0Block(0.125, vals)
	dequantParity(t, raw, dtype.Q5_0, 32, quant.DequantQ5_0)
}

func TestDequantQ5_1CUDAParity(t *testing.T) {
	var vals [32]uint8
	for i := range vals {
		vals[i] = uint8(i)
	}
	raw := buildQ5_1Block(0.5, 2.0, vals)
	dequantParity(t, raw, dtype.Q5_1, 32, quant.DequantQ5_1)
}

func TestDequantQ8_0CUDAParity(t *testing.T) {
	var vals [32]int8
	for i := range vals {
		vals[i] = int8(i - 16)
	}
	raw := buildQ8_0Block(0.03125, vals)
	dequantParity(t, raw, dtype.Q8_0, 32, quant.DequantQ8_0)
}

func TestDequantQ8_1CUDAParity(t *testing.T) {
	var vals [32]int8
	for i := range vals {
		vals[i] = int8((i * 3) - 48)
	}
	raw := buildQ8_1Block(0.0625, 0, vals)
	dequantParity(t, raw, dtype.Q8_1, 32, quant.DequantQ8_1)
}

func TestDequantQ4_KCUDAParity(t *testing.T) {
	var subScales, subMins [8]uint8
	for j := range 8 {
		subScales[j] = uint8(j*5 + 1)
		subMins[j] = uint8(j*3 + 2)
	}
	var quants [256]uint8
	for i := range quants {
		quants[i] = uint8((i*7 + 3) & 0x0F)
	}
	raw := buildQ4_KBlock(0.125, 0.0625, subScales, subMins, quants)
	dequantParity(t, raw, dtype.Q4_K, 256, quant.DequantQ4_K)
}

func TestDequantQ4_KCUDAMultiBlock(t *testing.T) {
	const nBlocks = 3
	buf := make([]byte, 0, nBlocks*quant.Q4_KBlockBytes)
	for b := range nBlocks {
		var subScales, subMins [8]uint8
		for j := range 8 {
			subScales[j] = uint8((j + b) & 0x3F)
			subMins[j] = uint8((j*2 + b) & 0x3F)
		}
		var quants [256]uint8
		for i := range quants {
			quants[i] = uint8((i + b*13) & 0x0F)
		}
		buf = append(buf, buildQ4_KBlock(float32(b+1)*0.1, float32(b+1)*0.05, subScales, subMins, quants)...)
	}
	dequantParity(t, buf, dtype.Q4_K, nBlocks*256, quant.DequantQ4_K)
}

func TestDequantQ2_KCUDAParity(t *testing.T) {
	var subScales, subMins [16]uint8
	for i := range subScales {
		subScales[i] = uint8((i + 3) & 0x0F)
		subMins[i] = uint8((i*2 + 1) & 0x0F)
	}
	var q2 [256]uint8
	for i := range q2 {
		q2[i] = uint8((i * 5) & 0x03)
	}
	block := buildQ2_KBlock(0.0625, 0.03125, subScales, subMins, q2)
	dequantParity(t, block, dtype.Q2_K, 256, quant.DequantQ2_K)
}

func TestDequantQ3_KCUDAParity(t *testing.T) {
	var rawScales [16]uint8
	for i := range rawScales {
		rawScales[i] = uint8((i*4 + 10) & 0x3F)
	}
	var rawQ2 [256]uint8
	var highBit [256]uint8
	for i := range rawQ2 {
		rawQ2[i] = uint8((i * 7) & 0x03)
		if i%3 == 0 {
			highBit[i] = 1
		}
	}
	block := buildQ3_KBlock(0.125, rawScales, rawQ2, highBit)
	dequantParity(t, block, dtype.Q3_K, 256, quant.DequantQ3_K)
}

func TestDequantQ8_KCUDAParity(t *testing.T) {
	var qs [256]int8
	for i := range qs {
		qs[i] = int8((i * 3) - 128)
	}
	block := buildQ8_KBlock(0.0078125, qs)
	dequantParity(t, block, dtype.Q8_K, 256, quant.DequantQ8_K)
}

func TestDequantQ5_KCUDAParity(t *testing.T) {
	var subScales, subMins [8]uint8
	for j := range 8 {
		subScales[j] = uint8(j*3 + 2)
		subMins[j] = uint8(j + 5)
	}
	var q5 [256]uint8
	for i := range q5 {
		q5[i] = uint8((i*5 + 7) & 0x1F)
	}
	block := buildQ5_KBlock(0.125, 0.0625, subScales, subMins, q5)
	dequantParity(t, block, dtype.Q5_K, 256, quant.DequantQ5_K)
}

func TestDequantQ5_KCUDAMultiBlock(t *testing.T) {
	const nBlocks = 3
	buf := make([]byte, 0, nBlocks*quant.Q5_KBlockBytes)
	for b := range nBlocks {
		var subScales, subMins [8]uint8
		for j := range 8 {
			subScales[j] = uint8((j + b*2) & 0x3F)
			subMins[j] = uint8((j*2 + b) & 0x3F)
		}
		var q5 [256]uint8
		for i := range q5 {
			q5[i] = uint8((i + b*19) & 0x1F)
		}
		buf = append(buf, buildQ5_KBlock(float32(b+1)*0.1, float32(b+1)*0.05, subScales, subMins, q5)...)
	}
	dequantParity(t, buf, dtype.Q5_K, nBlocks*256, quant.DequantQ5_K)
}

func TestDequantQ6_KCUDAParity(t *testing.T) {
	var scales [16]int8
	for i := range scales {
		scales[i] = int8(i - 8) // mix negative and positive
	}
	var raw [256]uint8
	for i := range raw {
		raw[i] = uint8((i * 3) & 0x3F)
	}
	block := buildQ6_KBlock(0.0625, scales, raw)
	dequantParity(t, block, dtype.Q6_K, 256, quant.DequantQ6_K)
}

func TestDequantQ6_KCUDAMultiBlock(t *testing.T) {
	const nBlocks = 3
	buf := make([]byte, 0, nBlocks*quant.Q6_KBlockBytes)
	for b := range nBlocks {
		var scales [16]int8
		for i := range scales {
			scales[i] = int8((i*2 + b*5) & 0x7F)
		}
		var raw [256]uint8
		for i := range raw {
			raw[i] = uint8((i + b*11) & 0x3F)
		}
		buf = append(buf, buildQ6_KBlock(float32(b+1)*0.1, scales, raw)...)
	}
	dequantParity(t, buf, dtype.Q6_K, nBlocks*256, quant.DequantQ6_K)
}

func TestDequantQ4_0CUDAMultiBlock(t *testing.T) {
	const nBlocks = 4
	buf := make([]byte, 0, nBlocks*int(quant.Q4_0BlockBytes))
	for b := range nBlocks {
		var vals [32]int8
		for i := range vals {
			vals[i] = int8((i+b)%16) - 8
		}
		buf = append(buf, buildQ4_0Block(float32(b+1)*0.1, vals)...)
	}
	dequantParity(t, buf, dtype.Q4_0, nBlocks*32, quant.DequantQ4_0)
}
