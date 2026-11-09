package ops_test

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// Reference values from torch.nn.functional (see the python snippet in
// docs/history/checkpoints-2026.md, step 7.17): tensors filled with sin(seed + 0.7·i)·0.5.
func fillT(t *testing.T, dims []int, seed float64, dev device.Device) *tensor.Tensor {
	t.Helper()
	n := 1
	for _, d := range dims {
		n *= d
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(math.Sin(seed+0.7*float64(i)) * 0.5)
	}
	x, err := tensor.New(data, dims, dev)
	if err != nil {
		t.Fatal(err)
	}
	return x
}

var ct2dRef = []float32{0.154422, 0.38375, 0.192706, 0.317456, 0.209851, -0.123677, 0.416845, 0.203685, -0.092944, -0.425122, -0.489527, -0.301324, 0.197794, 0.014151, 0.162205, -0.08021, 0.119823, 0.270208, -0.586383, -0.705411, -0.337961, -0.225069, 0.138904, -0.022625, 0.09058, 0.046674, 0.088229, 0.208245, 0.113877, 0.295562, 0.570337, 0.811851, 0.587362, 0.622759, 0.397635, 0.270017, 0.426494, 0.651803, 0.459643, 0.579613, 0.474494, 0.142298, 0.694251, 0.477282, 0.18364, -0.152861, -0.215461, -0.020617, 0.464061, 0.280161, 0.433251, 0.193735, 0.396556, 0.54634, -0.315349, -0.434345, -0.069045, 0.046701, 0.40761, 0.247205, 0.371232, 0.331605, 0.369191, 0.491058, 0.391392, 0.572352, 0.840211, 1.080171, 0.857112, 0.889818, 0.668778, 0.538893}
var ct2dGroupsRef = []float32{0.048976, 0.101783, 0.13394, 0.337786, 0.106719, 0.061464, 0.382766, 0.247725, 0.155909, 0.414923, 0.104552, 0.296915, 0.478792, 0.317477, 0.349634, 0.237915, 0.085989, -0.060007, -0.02785, 0.053286, -0.177781, -0.211942, 0.10936, 0.114001, -0.128591, 0.141518, -0.168853, 0.163192, 0.345068, 0.386328, 0.418485, 0.476958}
var ct1dRef = []float32{-0.063042, -0.061405, -0.00871, 0.025672, 0.049719, 0.100674, 0.084764, 0.128328, 0.079943, 0.025454, 0.058153, 0.006768, 0.028994, -0.0151, -0.013802, -0.029867, -0.050106, -0.030587, 0.037342, 0.002688, 0.001876, -0.054946, -0.034472, -0.086739, -0.054608, -0.077737, -0.04906}
var bilinearAC0 = []float32{0.358678, 0.414706, 0.498747, 0.442048, 0.404248, 0.239151, 0.198988, 0.138744, 0.039347, -0.026917, 0.119625, -0.016729, -0.22126, -0.363354, -0.458083}
var bilinearAC1 = []float32{0.358678, 0.428713, 0.498747, 0.451498, 0.404248, 0.239151, 0.188947, 0.138744, 0.055913, -0.026917, 0.119625, -0.050818, -0.22126, -0.339672, -0.458083}
var bilinearDown = []float32{0.054809, -0.016517, -0.074897, 0.081147, 0.036494, -0.036765}

func testConvTranspose(t *testing.T, dev device.Device, dt dtype.DType, tol float64) {
	cast := func(x *tensor.Tensor) *tensor.Tensor {
		if dt == dtype.F32 {
			return x
		}
		c, err := ops.Cast(x, dt)
		if err != nil {
			t.Fatal(err)
		}
		return c
	}
	x := cast(fillT(t, []int{1, 2, 3, 3}, 0.1, dev))
	w := cast(fillT(t, []int{2, 2, 3, 3}, 0.2, dev))
	b := cast(fillT(t, []int{2}, 0.3, dev))
	y, err := ops.ConvTranspose2d(x, w, b, ops.ConvTranspose2dOpts{Stride: [2]int{2, 2}, Padding: [2]int{1, 1}, OutputPadding: [2]int{1, 1}})
	if err != nil {
		t.Fatal(err)
	}
	if d := y.Shape().Dims(); d[1] != 2 || d[2] != 6 || d[3] != 6 {
		t.Fatalf("shape %v", y.Shape())
	}
	expectRow(t, "ct2d s2 p1 op1", f32s(t, y), ct2dRef, tol)

	x2 := cast(fillT(t, []int{1, 4, 2, 2}, 0.5, dev))
	w2 := cast(fillT(t, []int{4, 1, 2, 2}, 0.4, dev))
	y2, err := ops.ConvTranspose2d(x2, w2, nil, ops.ConvTranspose2dOpts{Dilation: [2]int{2, 2}, Groups: 2})
	if err != nil {
		t.Fatal(err)
	}
	if d := y2.Shape().Dims(); d[1] != 2 || d[2] != 4 || d[3] != 4 {
		t.Fatalf("shape %v", y2.Shape())
	}
	expectRow(t, "ct2d groups dilation", f32s(t, y2), ct2dGroupsRef, tol)

	x3 := cast(fillT(t, []int{1, 2, 5}, 0.6, dev))
	w3 := cast(fillT(t, []int{2, 3, 3}, 0.7, dev))
	y3, err := ops.ConvTranspose1d(x3, w3, nil, 2, 1, 0, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if d := y3.Shape().Dims(); d[1] != 3 || d[2] != 9 {
		t.Fatalf("shape %v", y3.Shape())
	}
	expectRow(t, "ct1d", f32s(t, y3), ct1dRef, tol)

	xi := cast(fillT(t, []int{1, 1, 2, 3}, 0.8, dev))
	u0, err := ops.UpsampleBilinear2d(xi, 3, 5, false)
	if err != nil {
		t.Fatal(err)
	}
	expectRow(t, "bilinear align=false", f32s(t, u0), bilinearAC0, tol)
	u1, err := ops.UpsampleBilinear2d(xi, 3, 5, true)
	if err != nil {
		t.Fatal(err)
	}
	expectRow(t, "bilinear align=true", f32s(t, u1), bilinearAC1, tol)
	xd := cast(fillT(t, []int{1, 1, 4, 4}, 0.9, dev))
	ud, err := ops.UpsampleBilinear2d(xd, 2, 3, false)
	if err != nil {
		t.Fatal(err)
	}
	expectRow(t, "bilinear downsample", f32s(t, ud), bilinearDown, tol)
}

func TestConvTransposeBilinearCPU(t *testing.T) {
	testConvTranspose(t, device.CPU(), dtype.F32, 2e-6)
	testConvTranspose(t, device.CPU(), dtype.F64, 2e-6)
}

func TestConvTransposeBilinearCUDA(t *testing.T) {
	gpu := cudaOrSkip(t)
	testConvTranspose(t, gpu, dtype.F32, 2e-6)
	testConvTranspose(t, gpu, dtype.BF16, 2e-2)
}
