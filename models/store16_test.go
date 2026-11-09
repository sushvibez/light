package models

import (
	"slices"
	"testing"

	"light/device"
	"light/dtype"
	"light/nn"
)

// store16Case loads one cached checkpoint at F32 on a device.
type store16Case struct {
	repo string
	load func(dev device.Device) (CausalLM, error)
	ids  []int64
}

var qwenChatIDs = []int64{151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198}

var store16Cases = []store16Case{
	{"Qwen/Qwen2.5-0.5B-Instruct", func(d device.Device) (CausalLM, error) {
		return LoadCausalLMFromHF("Qwen/Qwen2.5-0.5B-Instruct", d, dtype.F32)
	}, qwenChatIDs},
	{"Qwen/Qwen3.5-0.8B", func(d device.Device) (CausalLM, error) {
		return LoadQwen35FromHF("Qwen/Qwen3.5-0.8B", d, dtype.F32)
	}, qwen35SmallChatIDs},
	{"unsloth/gemma-3-1b-it", func(d device.Device) (CausalLM, error) {
		return LoadGemma3FromHF("unsloth/gemma-3-1b-it", d, dtype.F32)
	}, []int64{2, 105, 2364, 107, 3689, 563, 506, 5279, 529, 7001, 236881, 106, 107, 105, 4368, 107}},
}

// compareStore16 runs prefill + 8 greedy steps on a reference model and a
// 16-bit-storage one and checks the logits and tokens agree.
func compareStore16(t *testing.T, ref, got CausalLM, refDev, gotDev device.Device, ids []int64, tol float64) {
	t.Helper()
	cr, cg := NewKVCache(ref.NumLayers()), NewKVCache(got.NumLayers())
	rr, rg := lastLogits(t, ref, ids, cr, refDev), lastLogits(t, got, ids, cg, gotDev)
	d := maxAbsDiff(rr, rg)
	t.Logf("prefill logits max diff %.3g", d)
	if d > tol {
		t.Fatalf("prefill logits differ by %.3g (tol %.3g)", d, tol)
	}
	var tr, tg []int64
	for range 8 {
		nr, ng := int64(top5(rr)[0]), int64(top5(rg)[0])
		tr, tg = append(tr, nr), append(tg, ng)
		rr, rg = lastLogits(t, ref, []int64{nr}, cr, refDev), lastLogits(t, got, []int64{ng}, cg, gotDev)
	}
	t.Logf("greedy ref %v\n       w16 %v", tr, tg)
	if !slices.Equal(tr, tg) {
		t.Errorf("greedy decode differs")
	}
}

// BF16 checkpoints narrowed back to BF16 storage are exact, so on the CPU
// the W16A32 model matches the F32 one up to summation order.
func TestStoreWeights16CPU(t *testing.T) {
	cpu := device.CPU()
	c := store16Cases[0]
	ref, err := c.load(cpu)
	if err != nil {
		t.Skip(err)
	}
	got, err := c.load(cpu)
	if err != nil {
		t.Fatal(err)
	}
	n, err := StoreWeights16(got, dtype.BF16)
	if err != nil {
		t.Fatal(err)
	}
	if n == 0 {
		t.Fatal("no weights narrowed")
	}
	t.Logf("%d tensors narrowed", n)
	compareStore16(t, ref, got, cpu, cpu, c.ids, 1e-3)
}

// WebGPU with BF16 weight storage against the CPU at F32.
func TestStoreWeights16WebGPU(t *testing.T) {
	g, err := device.WebGPU(0)
	if err != nil {
		t.Skip(err)
	}
	cpu := device.CPU()
	for _, c := range store16Cases {
		t.Run(c.repo, func(t *testing.T) {
			ref, err := c.load(cpu)
			if err != nil {
				t.Skip(err)
			}
			got, err := c.load(g)
			if err != nil {
				t.Fatal(err)
			}
			n, err := StoreWeights16(got, dtype.BF16)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("%d tensors narrowed", n)
			compareStore16(t, ref, got, cpu, g, c.ids, 1e-2)
		})
	}
}

// Loaded on the CPU, moved to WebGPU with BF16 weight storage (the engine's
// WebGPU bf16 path), against the CPU at F32.
func TestMoveToDevice16WebGPU(t *testing.T) {
	g, err := device.WebGPU(0)
	if err != nil {
		t.Skip(err)
	}
	cpu := device.CPU()
	for _, c := range store16Cases {
		t.Run(c.repo, func(t *testing.T) {
			ref, err := c.load(cpu)
			if err != nil {
				t.Skip(err)
			}
			got, err := c.load(cpu)
			if err != nil {
				t.Fatal(err)
			}
			n, err := MoveToDevice16(got, g, dtype.BF16)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("%d tensors narrowed", n)
			compareStore16(t, ref, got, cpu, g, c.ids, 1e-2)
		})
	}
}

// Loaded straight into BF16 weight storage (nn.WithWeightStorage16, the
// engine's w16 path) on the CPU and on WebGPU, against the CPU at F32.
func TestLoadWeightStorage16(t *testing.T) {
	cpu := device.CPU()
	devs := []device.Device{cpu}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, c := range store16Cases {
		t.Run(c.repo, func(t *testing.T) {
			ref, err := c.load(cpu)
			if err != nil {
				t.Skip(err)
			}
			for _, dev := range devs {
				var got CausalLM
				err := nn.WithWeightStorage16(dtype.BF16, func() error {
					var err error
					got, err = c.load(dev)
					return err
				})
				if err != nil {
					t.Fatal(err)
				}
				n, err := StoreWeights16(got, dtype.BF16)
				if err != nil {
					t.Fatal(err)
				}
				t.Logf("%v: %d tensors left to narrow after the load", dev, n)
				compareStore16(t, ref, got, cpu, dev, c.ids, 1e-2)
			}
		})
	}
}

// Q8_0 load-time quantization (the engine's q8_0 path: 16-bit load on the
// host, then MoveToDeviceQ8) on the CPU and WebGPU against the CPU at F32.
// Q8_0 is lossy: the prefill top-1 must agree and the logits stay close.
func TestMoveToDeviceQ8(t *testing.T) {
	cpu := device.CPU()
	devs := []device.Device{cpu}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, c := range store16Cases {
		t.Run(c.repo, func(t *testing.T) {
			ref, err := c.load(cpu)
			if err != nil {
				t.Skip(err)
			}
			rr := lastLogits(t, ref, c.ids, NewKVCache(ref.NumLayers()), cpu)
			for _, dev := range devs {
				var got CausalLM
				if err := nn.WithWeightStorage16(dtype.BF16, func() error {
					var err error
					got, err = c.load(cpu)
					return err
				}); err != nil {
					t.Fatal(err)
				}
				n, err := MoveToDeviceQ8(got, dev)
				if err != nil {
					t.Fatal(err)
				}
				rg := lastLogits(t, got, c.ids, NewKVCache(got.NumLayers()), dev)
				d := maxAbsDiff(rr, rg)
				t.Logf("%v: %d tensors quantized; prefill logits max diff %.3g; top-5 ref %v q8 %v", dev, n, d, top5(rr), top5(rg))
				if top5(rr)[0] != top5(rg)[0] {
					t.Errorf("%v: top-1 differs", dev)
				}
				if d > 1.5 {
					t.Errorf("%v: logits differ by %.3g", dev, d)
				}
			}
		})
	}
}
