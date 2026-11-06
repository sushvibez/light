package nn_test

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

func moeFill(t *testing.T, dims []int, seed float64) *tensor.Tensor {
	t.Helper()
	n := 1
	for _, d := range dims {
		n *= d
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(math.Sin(seed+0.7*float64(i)) * 0.5)
	}
	x, err := tensor.New(data, dims, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	return x
}

// Reference: HF-style loop over experts with index_select / index_add.
func sparseMoERef(t *testing.T, m *nn.SparseMoE, x *tensor.Tensor) []float32 {
	t.Helper()
	H, I, E := m.GateUp.Dim(2), m.GateUp.Dim(1)/2, m.GateUp.Dim(0)
	logits, _ := m.Gate.Forward(x)
	probs, _ := ops.Softmax(logits, -1)
	vals, ids, _ := ops.TopK(probs, m.TopK, true)
	den, _ := ops.Sum(vals, ops.ReduceOptions{Axes: []int{1}, KeepDims: true})
	vals, _ = ops.Div(vals, den)
	T := x.Dim(0)
	out := make([]float32, T*H)
	xs := tensor.MustSlice[float32](x)
	vs := tensor.MustSlice[float32](vals)
	is := tensor.MustSlice[int64](ids)
	gu := tensor.MustSlice[float32](m.GateUp)
	dn := tensor.MustSlice[float32](m.Down)
	silu := func(v float32) float32 { return v / (1 + float32(math.Exp(float64(-v)))) }
	for tok := range T {
		for s := range m.TopK {
			e := int(is[tok*m.TopK+s])
			_ = E
			h := make([]float32, I)
			for i := range I {
				var g, u float32
				for k := range H {
					g += xs[tok*H+k] * gu[(e*2*I+i)*H+k]
					u += xs[tok*H+k] * gu[(e*2*I+I+i)*H+k]
				}
				h[i] = silu(g) * u
			}
			for o := range H {
				var acc float32
				for i := range I {
					acc += h[i] * dn[(e*H+o)*I+i]
				}
				out[tok*H+o] += acc * vs[tok*m.TopK+s]
			}
		}
	}
	return out
}

func TestSparseMoEForward(t *testing.T) {
	H, I, E, topk, T := 16, 24, 4, 2, 9
	m, err := nn.NewSparseMoEUninit(H, I, E, topk, true, dtype.F32, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Gate.SetWeight(moeFill(t, []int{E, H}, 0.1)); err != nil {
		t.Fatal(err)
	}
	var gate, up, down []*tensor.Tensor
	for e := range E {
		gate = append(gate, moeFill(t, []int{I, H}, 0.2+float64(e)))
		up = append(up, moeFill(t, []int{I, H}, 0.3+float64(e)))
		down = append(down, moeFill(t, []int{H, I}, 0.4+float64(e)))
	}
	if err := m.StackExperts(gate, up, down); err != nil {
		t.Fatal(err)
	}
	x := moeFill(t, []int{1, T, H}, 0.7)
	x2, _ := x.Reshape(T, H)
	ref := sparseMoERef(t, m, x2)
	y, err := m.Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	if d := y.Shape().Dims(); d[0] != 1 || d[1] != T || d[2] != H {
		t.Fatalf("shape %v", y.Shape())
	}
	ys := tensor.MustSlice[float32](y)
	for i := range ref {
		if math.Abs(float64(ys[i]-ref[i])) > 1e-5 {
			t.Fatalf("[%d] = %v want %v", i, ys[i], ref[i])
		}
	}
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skip("no CUDA")
	}
	for _, dt := range []dtype.DType{dtype.F32, dtype.BF16, dtype.F16} {
		cast := func(x *tensor.Tensor) *tensor.Tensor {
			c, err := ops.Cast(x, dt)
			if err != nil {
				t.Fatal(err)
			}
			g, _ := c.To(gpu)
			return g
		}
		g, _ := nn.NewSparseMoEUninit(H, I, E, topk, true, dt, gpu)
		if err := g.Gate.SetWeight(cast(m.Gate.W)); err != nil {
			t.Fatal(err)
		}
		g.GateUp, g.Down = cast(m.GateUp), cast(m.Down)
		yg, err := g.Forward(cast(x))
		if err != nil {
			t.Fatalf("%v on CUDA: %v", dt, err)
		}
		yf, _ := ops.Cast(yg, dtype.F32)
		yc, _ := yf.To(device.CPU())
		gs := tensor.MustSlice[float32](yc)
		var num, den float64
		for i := range ref {
			num += float64((gs[i] - ref[i]) * (gs[i] - ref[i]))
			den += float64(ref[i] * ref[i])
		}
		tol := 1e-4
		if dt != dtype.F32 {
			tol = 5e-2
		}
		if e := math.Sqrt(num / den); e > tol {
			t.Errorf("%v SparseMoE CUDA rel err %.4g", dt, e)
		}
	}
}
