package ops_test

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"light/device"
	"light/ops"
	"light/tensor"
)

// DeformConv2d against torchvision.ops.deform_conv2d
// (scripts/verify/ref_deform_conv.py): kernel 1, 3, 7 with same padding,
// offsets that leave the image, a mask, bias or none, stride and dilation.
func TestDeformConv2dMatchesTorchvision(t *testing.T) {
	raw, err := os.ReadFile("testdata/deform_conv.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		K, Stride, Dilation, Padding int
		Dims                         []int
		X, Offset, Mask, Weight, Out []float32
		Bias                         []float32
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, c := range cases {
		B, C, H, W, O, Ho, Wo := c.Dims[0], c.Dims[1], c.Dims[2], c.Dims[3], c.Dims[4], c.Dims[5], c.Dims[6]
		for _, dev := range devs {
			t.Run(fmt.Sprintf("k%d_s%d_d%d/%v", c.K, c.Stride, c.Dilation, dev), func(t *testing.T) {
				nt := func(v []float32, dims ...int) *tensor.Tensor {
					if v == nil {
						return nil
					}
					x, err := tensor.New(v, dims, dev)
					if err != nil {
						t.Fatal(err)
					}
					return x
				}
				y, err := ops.DeformConv2d(nt(c.X, B, C, H, W), nt(c.Offset, B, 2*c.K*c.K, Ho, Wo), nt(c.Mask, B, c.K*c.K, Ho, Wo),
					nt(c.Weight, O, C, c.K, c.K), nt(c.Bias, O), ops.DeformConv2dOptions{
						Stride: [2]int{c.Stride, c.Stride}, Padding: [2]int{c.Padding, c.Padding}, Dilation: [2]int{c.Dilation, c.Dilation}})
				if err != nil {
					t.Fatal(err)
				}
				h, _ := y.To(device.CPU())
				got := tensor.MustSlice[float32](h)
				if len(got) != len(c.Out) {
					t.Fatalf("%d values, want %d", len(got), len(c.Out))
				}
				md := 0.0
				for i := range got {
					md = math.Max(md, math.Abs(float64(got[i]-c.Out[i])))
				}
				t.Logf("max diff %.3g", md)
				if md > 1e-4 {
					t.Errorf("differs by %.3g", md)
				}
			})
		}
	}
}
