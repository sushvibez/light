//go:build !nocuda

package ops_test

import (
	"math"
	"testing"

	"light/device"
	_ "light/kernels/cuda"
	"light/ops"
	"light/tensor"
)

// TestCUDABatchNorm2dParity verifies native CUDA train-mode BatchNorm matches
// the CPU reference within F32 tolerance.
func TestCUDABatchNorm2dParity(t *testing.T) {
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	N, C, H, W := 4, 3, 5, 6
	size := N * C * H * W
	xData := make([]float32, size)
	for i := range xData {
		xData[i] = float32((i%37)-18) * 0.13
	}
	gammaData := []float32{0.8, 1.0, 1.2}
	betaData := []float32{-0.1, 0.0, 0.2}
	rmData := []float32{0, 0, 0}
	rvData := []float32{1, 1, 1}

	xCPU, _ := tensor.New(xData, []int{N, C, H, W}, device.CPU())
	gCPU, _ := tensor.New(gammaData, []int{C}, device.CPU())
	bCPU, _ := tensor.New(betaData, []int{C}, device.CPU())
	rmCPU, _ := tensor.New(rmData, []int{C}, device.CPU())
	rvCPU, _ := tensor.New(rvData, []int{C}, device.CPU())
	cpuRes, err := ops.BatchNorm2d(xCPU, gCPU, bCPU, rmCPU, rvCPU, ops.BatchNormOptions{
		Train: true, Momentum: 0.1, Eps: 1e-5,
	})
	if err != nil {
		t.Fatal(err)
	}
	cpuOut := tensor.MustSlice[float32](cpuRes.Out)

	// GPU run.
	xGPU, _ := xCPU.To(gpu)
	gGPU, _ := gCPU.To(gpu)
	bGPU, _ := bCPU.To(gpu)
	rmGPU, _ := rmCPU.To(gpu)
	rvGPU, _ := rvCPU.To(gpu)
	gpuRes, err := ops.BatchNorm2d(xGPU, gGPU, bGPU, rmGPU, rvGPU, ops.BatchNormOptions{
		Train: true, Momentum: 0.1, Eps: 1e-5,
	})
	if err != nil {
		t.Fatal(err)
	}
	gpuOutCPU, _ := gpuRes.Out.To(device.CPU())
	gotOut := tensor.MustSlice[float32](gpuOutCPU)

	for i := range cpuOut {
		if math.Abs(float64(gotOut[i]-cpuOut[i])) > 1e-4 {
			t.Fatalf("forward parity mismatch at %d: cpu=%v gpu=%v", i, cpuOut[i], gotOut[i])
		}
	}

	// Backward parity.
	gradData := make([]float32, size)
	for i := range gradData {
		gradData[i] = float32((i%13)-6) * 0.05
	}
	gCPUtensor, _ := tensor.New(gradData, []int{N, C, H, W}, device.CPU())
	gpuGrad, _ := gCPUtensor.To(gpu)

	giCPU, ggCPU, gbCPU, err := ops.BatchNorm2dBackward(gCPUtensor, cpuRes.XHat, cpuRes.Sigma, gCPU)
	if err != nil {
		t.Fatal(err)
	}
	giGPU, ggGPU, gbGPU, err := ops.BatchNorm2dBackward(gpuGrad, gpuRes.XHat, gpuRes.Sigma, gGPU)
	if err != nil {
		t.Fatal(err)
	}
	for _, pair := range []struct {
		name     string
		cpu, gpu *tensor.Tensor
	}{
		{"grad_input", giCPU, giGPU},
		{"grad_gamma", ggCPU, ggGPU},
		{"grad_beta", gbCPU, gbGPU},
	} {
		gpuBack, _ := pair.gpu.To(device.CPU())
		cpuVals := tensor.MustSlice[float32](pair.cpu)
		gpuVals := tensor.MustSlice[float32](gpuBack)
		for i := range cpuVals {
			if math.Abs(float64(gpuVals[i]-cpuVals[i])) > 1e-3 {
				t.Fatalf("%s mismatch at %d: cpu=%v gpu=%v", pair.name, i, cpuVals[i], gpuVals[i])
			}
		}
	}
}
