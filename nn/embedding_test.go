package nn_test

import (
	"math"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/quant"
	"light/tensor"
)

func TestEmbeddingForward(t *testing.T) {
	e, err := nn.NewEmbedding(10, 4, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	// indices of shape (3,) → output (3, 4)
	idx, _ := tensor.New([]int64{0, 5, 9}, []int{3}, device.CPU())
	y, err := e.Forward(idx)
	if err != nil {
		t.Fatal(err)
	}
	if y.Rank() != 2 || y.Dim(0) != 3 || y.Dim(1) != 4 {
		t.Errorf("shape = %v", y.Shape())
	}
	if e.NumEmbeddings() != 10 || e.EmbDim() != 4 {
		t.Errorf("accessors: num=%d dim=%d", e.NumEmbeddings(), e.EmbDim())
	}
}

func TestEmbeddingBatch(t *testing.T) {
	// indices (batch=2, seq=3) → output (2, 3, 4)
	e, _ := nn.NewEmbedding(10, 4, device.CPU())
	idx, _ := tensor.New([]int64{0, 1, 2, 3, 4, 5}, []int{2, 3}, device.CPU())
	y, err := e.Forward(idx)
	if err != nil {
		t.Fatal(err)
	}
	if y.Rank() != 3 || y.Dim(0) != 2 || y.Dim(1) != 3 || y.Dim(2) != 4 {
		t.Errorf("shape = %v", y.Shape())
	}
}

func TestEmbeddingLookup(t *testing.T) {
	// Set known weights and verify lookup returns the right rows.
	e, _ := nn.NewEmbedding(3, 2, device.CPU())
	e.Weight, _ = tensor.New([]float32{
		10, 20, // row 0
		30, 40, // row 1
		50, 60, // row 2
	}, []int{3, 2}, device.CPU())
	idx, _ := tensor.New([]int64{2, 0, 1}, []int{3}, device.CPU())
	y, _ := e.Forward(idx)
	got := tensor.MustSlice[float32](y)
	want := []float32{50, 60, 10, 20, 30, 40}
	if !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestRMSNormModule(t *testing.T) {
	n, err := nn.NewRMSNorm(4, 1e-5, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	x, _ := tensor.New([]float32{1, 2, 3, 4}, []int{1, 4}, device.CPU())
	y, err := n.Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](y)
	// mean(x²) = 30/4 = 7.5; rms = sqrt(7.5 + 1e-5) ≈ 2.7386
	rms := float32(math.Sqrt(7.5 + 1e-5))
	want := []float32{1 / rms, 2 / rms, 3 / rms, 4 / rms}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4 {
			t.Errorf("[%d] got %v want %v", i, got[i], want[i])
		}
	}
	if len(n.Parameters()) != 1 {
		t.Errorf("expected 1 param (weight), got %d", len(n.Parameters()))
	}
}

func TestLayerNormModule(t *testing.T) {
	n, err := nn.NewLayerNorm(3, 1e-5, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	x, _ := tensor.New([]float32{1, 2, 3}, []int{1, 3}, device.CPU())
	y, err := n.Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](y)
	// mean=2, var=2/3, std≈sqrt(2/3)
	std := float32(math.Sqrt(2.0/3.0 + 1e-5))
	want := []float32{-1 / std, 0, 1 / std}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-3 {
			t.Errorf("[%d] got %v want %v", i, got[i], want[i])
		}
	}
	if len(n.Parameters()) != 2 {
		t.Errorf("expected 2 params (gamma, beta), got %d", len(n.Parameters()))
	}
}

func TestEmbeddingAccessors(t *testing.T) {
	e, _ := nn.NewEmbedding(7, 3, device.CPU())
	if e.NumEmbeddings() != 7 {
		t.Errorf("num = %d", e.NumEmbeddings())
	}
	if e.EmbDim() != 3 {
		t.Errorf("dim = %d", e.EmbDim())
	}
	if len(e.Parameters()) != 1 {
		t.Errorf("params = %d", len(e.Parameters()))
	}
}

// nn.Embedding with a block-quantized table (GGUF path): lookup result equals
// the dequantized table's rows, on CPU and CUDA.
func TestEmbeddingQuantizedTable(t *testing.T) {
	const V, D = 20, 256
	data := make([]float32, V*D)
	for i := range data {
		data[i] = float32((i*7919)%1000)/500 - 1
	}
	f32, _ := tensor.New(data, []int{V, D}, device.CPU())
	q, err := quant.Quantize(f32, dtype.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	devs := []device.Device{device.CPU()}
	if device.CUDAAvailable() {
		gpu, _ := device.CUDA(0)
		devs = append(devs, gpu)
	}
	for _, dev := range devs {
		qd, _ := q.To(dev)
		emb, _ := nn.NewEmbeddingUninit(V, D, dtype.F32, dev)
		emb.Weight = qd
		ids, _ := tensor.New([]int64{3, 19, 0, 3}, []int{2, 2}, dev)
		out, err := emb.Forward(ids)
		if err != nil {
			t.Fatal(err)
		}
		if d := out.Shape().Dims(); len(d) != 3 || d[0] != 2 || d[1] != 2 || d[2] != D || out.DType() != dtype.F32 {
			t.Fatalf("%v: shape %v dtype %v", dev, out.Shape(), out.DType())
		}
		deq, _ := quant.Dequantize(qd)
		ref, _ := nn.NewEmbeddingUninit(V, D, dtype.F32, dev)
		ref.Weight = deq
		want, _ := ref.Forward(ids)
		o, _ := out.To(device.CPU())
		w, _ := want.To(device.CPU())
		os, ws := tensor.MustSlice[float32](o), tensor.MustSlice[float32](w)
		for i := range ws {
			if os[i] != ws[i] {
				t.Fatalf("%v: [%d] %v vs %v", dev, i, os[i], ws[i])
			}
		}
	}
}
