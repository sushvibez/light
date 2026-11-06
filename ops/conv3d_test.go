package ops_test

import (
	"math"
	"testing"

	"light/device"
	"light/ops"
	"light/tensor"
)

// conv3dRef is an independent loop implementation used as the oracle.
func conv3dRef(in, w, bias []float32, N, Cin, D, H, W, Cout, kD, kH, kW int, stride, pad [3]int) ([]float32, [3]int) {
	Dout := (D+2*pad[0]-kD)/stride[0] + 1
	Hout := (H+2*pad[1]-kH)/stride[1] + 1
	Wout := (W+2*pad[2]-kW)/stride[2] + 1
	out := make([]float32, N*Cout*Dout*Hout*Wout)
	for n := range N {
		for co := range Cout {
			for od := range Dout {
				for oh := range Hout {
					for ow := range Wout {
						var acc float32
						if bias != nil {
							acc = bias[co]
						}
						for ci := range Cin {
							for kd := range kD {
								for kh := range kH {
									for kw := range kW {
										id, ih, iw := od*stride[0]-pad[0]+kd, oh*stride[1]-pad[1]+kh, ow*stride[2]-pad[2]+kw
										if id < 0 || id >= D || ih < 0 || ih >= H || iw < 0 || iw >= W {
											continue
										}
										acc += in[(((n*Cin+ci)*D+id)*H+ih)*W+iw] * w[(((co*Cin+ci)*kD+kd)*kH+kh)*kW+kw]
									}
								}
							}
						}
						out[(((n*Cout+co)*Dout+od)*Hout+oh)*Wout+ow] = acc
					}
				}
			}
		}
	}
	return out, [3]int{Dout, Hout, Wout}
}

func TestConv3dCPU(t *testing.T) {
	cpu := device.CPU()
	const N, Cin, D, H, W = 2, 3, 4, 5, 6
	const Cout, kD, kH, kW = 4, 2, 3, 3
	inData := make([]float32, N*Cin*D*H*W)
	for i := range inData {
		inData[i] = float32(i%13)*0.1 - 0.6
	}
	wData := make([]float32, Cout*Cin*kD*kH*kW)
	for i := range wData {
		wData[i] = float32(i%7)*0.05 - 0.15
	}
	bData := []float32{0.1, -0.2, 0.3, -0.4}
	in, _ := tensor.New(inData, []int{N, Cin, D, H, W}, cpu)
	w, _ := tensor.New(wData, []int{Cout, Cin, kD, kH, kW}, cpu)
	b, _ := tensor.New(bData, []int{Cout}, cpu)

	cases := []struct {
		name   string
		stride [3]int
		pad    [3]int
		bias   bool
	}{
		{"s1p0", [3]int{1, 1, 1}, [3]int{0, 0, 0}, false},
		{"s1p1bias", [3]int{1, 1, 1}, [3]int{1, 1, 1}, true},
		{"s2p1", [3]int{2, 2, 2}, [3]int{1, 1, 1}, true},
		{"mixed", [3]int{1, 2, 1}, [3]int{0, 1, 2}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var bt *tensor.Tensor
			var bs []float32
			if c.bias {
				bt, bs = b, bData
			}
			out, err := ops.Conv3d(in, w, bt, c.stride, c.pad)
			if err != nil {
				t.Fatal(err)
			}
			want, sp := conv3dRef(inData, wData, bs, N, Cin, D, H, W, Cout, kD, kH, kW, c.stride, c.pad)
			if out.Dim(2) != sp[0] || out.Dim(3) != sp[1] || out.Dim(4) != sp[2] {
				t.Fatalf("shape %v, want spatial %v", out.Shape(), sp)
			}
			got := tensor.MustSlice[float32](out)
			for i := range want {
				if math.Abs(float64(got[i]-want[i])) > 1e-4 {
					t.Fatalf("[%d] = %v, want %v", i, got[i], want[i])
				}
			}
		})
	}
	// F64 path.
	in64, _ := tensor.New(toF64(inData), []int{N, Cin, D, H, W}, cpu)
	w64, _ := tensor.New(toF64(wData), []int{Cout, Cin, kD, kH, kW}, cpu)
	out64, err := ops.Conv3d(in64, w64, nil, [3]int{1, 1, 1}, [3]int{0, 0, 0})
	if err != nil {
		t.Fatal(err)
	}
	want, _ := conv3dRef(inData, wData, nil, N, Cin, D, H, W, Cout, kD, kH, kW, [3]int{1, 1, 1}, [3]int{0, 0, 0})
	got := tensor.MustSlice[float64](out64)
	for i := range want {
		if math.Abs(got[i]-float64(want[i])) > 1e-4 {
			t.Fatalf("F64 [%d] = %v, want %v", i, got[i], want[i])
		}
	}
	// Errors.
	if _, err := ops.Conv3d(in, w, nil, [3]int{0, 1, 1}, [3]int{0, 0, 0}); err == nil {
		t.Error("expected stride error")
	}
	bad, _ := tensor.Zeros(in.DType(), []int{N, Cin, D, H}, cpu)
	if _, err := ops.Conv3d(bad, w, nil, [3]int{1, 1, 1}, [3]int{0, 0, 0}); err == nil {
		t.Error("expected rank error")
	}
}

func toF64(x []float32) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = float64(v)
	}
	return out
}
