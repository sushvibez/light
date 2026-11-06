package ops_test

import (
	"math"
	"math/rand"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/quant"
	"light/tensor"
)

// Quantized embedding lookup: IndexSelect on a block-quantized table gathers
// whole rows of blocks, so dequantizing the gathered rows must equal
// gathering the dequantized table — on CPU and CUDA, for every GGUF format.
func TestIndexSelectQuantizedRows(t *testing.T) {
	const V, K = 37, 512
	rng := rand.New(rand.NewSource(7))
	data := make([]float32, V*K)
	for i := range data {
		data[i] = float32(rng.NormFloat64())
	}
	f32, _ := tensor.New(data, []int{V, K}, device.CPU())
	idsCPU, _ := tensor.New([]int64{0, 36, 5, 5, 17, 2}, []int{6}, device.CPU())

	tables := map[dtype.DType]*tensor.Tensor{}
	for _, dt := range []dtype.DType{dtype.Q4_0, dtype.Q5_0, dtype.Q8_0} {
		q, err := quant.Quantize(f32, dt)
		if err != nil {
			t.Fatal(err)
		}
		tables[dt] = q
	}
	// Formats without an encoder: random block bytes are valid codes; the
	// two paths must still agree byte for byte (NaN-aware).
	for _, dt := range []dtype.DType{dtype.Q4_1, dtype.Q5_1, dtype.Q8_1, dtype.Q2_K, dtype.Q3_K, dtype.Q4_K, dtype.Q5_K, dtype.Q6_K, dtype.Q8_K} {
		raw := make([]byte, dt.ByteSize(V*K))
		rng.Read(raw)
		q, err := tensor.FromBytes(raw, dt, []int{V, K}, device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		tables[dt] = q
	}

	devs := []device.Device{device.CPU()}
	if device.CUDAAvailable() {
		gpu, err := device.CUDA(0)
		if err != nil {
			t.Fatal(err)
		}
		devs = append(devs, gpu)
	}
	for dt, q := range tables {
		for _, dev := range devs {
			qd, _ := q.To(dev)
			ids, _ := idsCPU.To(dev)
			rows, err := ops.IndexSelect(qd, ids)
			if err != nil {
				t.Fatalf("%v %v: %v", dt, dev, err)
			}
			if rows.DType() != dt || rows.Dim(0) != 6 || rows.Dim(1) != K {
				t.Fatalf("%v: rows %v %v", dt, rows.DType(), rows.Shape())
			}
			got, err := quant.Dequantize(rows)
			if err != nil {
				t.Fatal(err)
			}
			full, err := quant.Dequantize(qd)
			if err != nil {
				t.Fatal(err)
			}
			want, err := ops.IndexSelect(full, ids)
			if err != nil {
				t.Fatal(err)
			}
			g, w := f32s(t, got), f32s(t, want)
			for i := range w {
				if g[i] != w[i] && !(math.IsNaN(float64(g[i])) && math.IsNaN(float64(w[i]))) {
					t.Fatalf("%v %v: [%d] got %v want %v", dt, dev, i, g[i], w[i])
				}
			}
			if dt == dtype.Q8_0 { // sanity vs the F32 source (Q8_0 is near-lossless)
				src, _ := ops.IndexSelect(f32, idsCPU)
				expectRow(t, "q8_0 vs f32", g, f32s(t, src), 2e-2)
			}
		}
	}
}
