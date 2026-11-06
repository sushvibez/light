//go:build !nocuda

package ops_test

import (
	"math"
	"testing"

	"light/device"
	"light/ops"
	"light/tensor"
)

func TestGeluErfCUDAParity(t *testing.T) {
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	n := 4097
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(i-n/2) / 256
	}
	aCPU := mkF32(data, n)
	outCPU, err := ops.GeluErf(aCPU)
	if err != nil {
		t.Fatal(err)
	}
	aGPU, err := aCPU.To(gpu)
	if err != nil {
		t.Fatal(err)
	}
	outGPU, err := ops.GeluErf(aGPU)
	if err != nil {
		t.Fatalf("CUDA GeluErf: %v", err)
	}
	back, err := outGPU.To(device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	c, g := tensor.MustSlice[float32](outCPU), tensor.MustSlice[float32](back)
	for i := range c {
		if math.Abs(float64(c[i]-g[i])) > 1e-5 {
			t.Fatalf("[%d] x=%v CPU=%v GPU=%v", i, data[i], c[i], g[i])
		}
	}
}
