package nn_test

import (
	"math"
	"testing"

	"light/autograd"
	"light/device"
	"light/nn"
	"light/tensor"
)

// Deterministic weights shared with the PyTorch reference (see the values
// below, produced by torch.nn.LSTM / GRU with batch_first=True).
func rnnWeight(rows, cols int, seed float64, dev device.Device) *tensor.Tensor {
	data := make([]float32, rows*cols)
	for r := range rows {
		for c := range cols {
			data[r*cols+c] = float32(0.5 * math.Sin(seed+0.37*float64(r)+0.11*float64(c)))
		}
	}
	t, _ := tensor.New(data, []int{rows, cols}, dev)
	t.SetRequiresGrad(true)
	return t
}

func rnnBias(n int, seed float64, dev device.Device) *tensor.Tensor {
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(0.3 * math.Cos(seed+0.23*float64(i)))
	}
	t, _ := tensor.New(data, []int{n}, dev)
	t.SetRequiresGrad(true)
	return t
}

func rnnInput(dev device.Device) *tensor.Tensor {
	const B, T, I = 2, 3, 3
	data := make([]float32, B*T*I)
	for b := range B {
		for t := range T {
			for i := range I {
				data[(b*T+t)*I+i] = float32(0.1 * math.Sin(1+float64(b)+0.5*float64(t)+0.2*float64(i)))
			}
		}
	}
	x, _ := tensor.New(data, []int{B, T, I}, dev)
	return x
}

var lstmRefOut = []float32{-0.069505974650383, -0.094865582883358, -0.11471111327409744, -0.12973915040493011, -0.10827894508838654, -0.13241292536258698, -0.14722637832164764, -0.15512989461421967, -0.1251356601715088, -0.14112946391105652, -0.14867858588695526, -0.150844007730484, -0.06976594030857086, -0.09418265521526337, -0.11314720660448074, -0.1273098886013031, -0.11085551977157593, -0.1291280835866928, -0.13901765644550323, -0.14297151565551758, -0.1284652203321457, -0.13237155973911285, -0.13065102696418762, -0.1266988217830658}
var lstmRefC = []float32{-0.33316028118133545, -0.38284555077552795, -0.40511858463287354, -0.40643104910850525, -0.3179054856300354, -0.33176615834236145, -0.33091285824775696, -0.3229396939277649}
var gruRefOut = []float32{-0.11763693392276764, -0.18209342658519745, -0.2409278005361557, -0.2885383367538452, -0.1705394834280014, -0.22811642289161682, -0.27999618649482727, -0.32215943932533264, -0.19179826974868774, -0.23171402513980865, -0.2709004282951355, -0.306140661239624, -0.11773324012756348, -0.17981639504432678, -0.2362455129623413, -0.2816603183746338, -0.1708412766456604, -0.21218076348304749, -0.250440388917923, -0.2834506630897522, -0.19380925595760345, -0.20502758026123047, -0.21995395421981812, -0.23897887766361237}

func near(t *testing.T, name string, got *tensor.Tensor, want []float32, tol float64) {
	t.Helper()
	c, err := got.To(device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	g := tensor.MustSlice[float32](c)
	if len(g) != len(want) {
		t.Fatalf("%s: len %d want %d", name, len(g), len(want))
	}
	for i := range want {
		if math.Abs(float64(g[i]-want[i])) > tol {
			t.Errorf("%s[%d] = %.6f want %.6f", name, i, g[i], want[i])
		}
	}
}

func testRNN(t *testing.T, dev device.Device) {
	const I, H = 3, 4
	lstm := &nn.LSTM{Wih: rnnWeight(4*H, I, 0.1, dev), Whh: rnnWeight(4*H, H, 0.2, dev), Bih: rnnBias(4*H, 0.3, dev), Bhh: rnnBias(4*H, 0.4, dev), Hidden: H}
	out, st, err := lstm.Forward(rnnInput(dev))
	if err != nil {
		t.Fatal(err)
	}
	near(t, "lstm out", out, lstmRefOut, 1e-5)
	near(t, "lstm c", st.C, lstmRefC, 1e-5)

	gru := &nn.GRU{Wih: rnnWeight(3*H, I, 0.1, dev), Whh: rnnWeight(3*H, H, 0.2, dev), Bih: rnnBias(3*H, 0.3, dev), Bhh: rnnBias(3*H, 0.4, dev), Hidden: H}
	gout, _, err := gru.Forward(rnnInput(dev))
	if err != nil {
		t.Fatal(err)
	}
	near(t, "gru out", gout, gruRefOut, 1e-5)

	// Gradients flow back to every weight through the unrolled sequence.
	s, err := autograd.Sum(out)
	if err != nil {
		t.Fatal(err)
	}
	if err := autograd.Backward(s, nil); err != nil {
		t.Fatal(err)
	}
	for i, p := range lstm.Parameters() {
		if p.Grad() == nil {
			t.Errorf("lstm param %d has no grad", i)
		}
	}
}

func TestLSTMGRUMatchTorchCPU(t *testing.T) { testRNN(t, device.CPU()) }

func TestLSTMGRUMatchTorchCUDA(t *testing.T) {
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	testRNN(t, dev)
}

func TestNewLSTMGRUShapes(t *testing.T) {
	l, err := nn.NewLSTM(5, 7, true, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if len(l.Parameters()) != 4 || l.Wih.Dim(0) != 28 || l.Whh.Dim(1) != 7 {
		t.Fatalf("lstm params %d, wih %v", len(l.Parameters()), l.Wih.Shape())
	}
	g, err := nn.NewGRU(5, 7, false, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if len(g.Parameters()) != 2 || g.Wih.Dim(0) != 21 {
		t.Fatalf("gru params %d, wih %v", len(g.Parameters()), g.Wih.Shape())
	}
	x, _ := tensor.Zeros(l.Wih.DType(), []int{2, 4, 5}, device.CPU())
	out, _, err := l.Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	if d := out.Shape().Dims(); d[0] != 2 || d[1] != 4 || d[2] != 7 {
		t.Fatalf("out shape %v", out.Shape())
	}
	act := []nn.Module{nn.GELU{}, nn.ReLU6{}, nn.ELU{}, nn.LeakyReLU{}, nn.Mish{}, nn.HardSwish{}, nn.HardSigmoid{}, nn.SwiGLU{}}
	p, _ := nn.NewPReLU(0, device.CPU())
	act = append(act, p)
	in, _ := tensor.New([]float32{-1, 0.5, 2, -3}, []int{2, 2}, device.CPU())
	for i, m := range act {
		if _, err := m.Forward(in); err != nil {
			t.Errorf("module %d: %v", i, err)
		}
	}
}
