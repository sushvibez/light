package models

import (
	"math"
	"testing"

	"light/autograd"
	"light/device"
	"light/tensor"
)

// GPT-2 small on WebGPU equals the CPU, stage by stage: embeddings, every
// block, the final norm and the logits (skips without a WebGPU adapter or
// the cached gpt2 checkpoint).
func TestGPT2WebGPUMatchesCPU(t *testing.T) {
	g, err := device.WebGPU(0)
	if err != nil {
		t.Skip(err)
	}
	cpu := device.CPU()
	mc, err := LoadGPT2FromHF("gpt2", cpu)
	if err != nil {
		t.Skip(err)
	}
	mg, err := LoadGPT2FromHF("gpt2", g)
	if err != nil {
		t.Fatal(err)
	}
	ids := []int64{464, 3139, 286, 4881, 318}
	diff := func(a, b *tensor.Tensor) float64 {
		ac, _ := a.To(cpu)
		bc, _ := b.To(cpu)
		ac, _ = ac.Contiguous()
		bc, _ = bc.Contiguous()
		x, y := tensor.MustSlice[float32](ac), tensor.MustSlice[float32](bc)
		var md float64
		for i := range x {
			d := math.Abs(float64(x[i] - y[i]))
			if math.IsNaN(d) {
				return math.NaN()
			}
			md = max(md, d)
		}
		return md
	}
	run := func(m *GPT2, dev device.Device) []*tensor.Tensor {
		tok, _ := tensor.New(ids, []int{1, len(ids)}, cpu)
		tok, _ = tok.To(dev)
		var out []*tensor.Tensor
		h, err := m.TokenEmb.Forward(tok)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, h)
		pos, _ := makePosIdx(len(ids), 1, dev)
		p, err := m.PosEmb.Forward(pos)
		if err != nil {
			t.Fatal(err)
		}
		if h, err = autograd.Add(h, p); err != nil {
			t.Fatal(err)
		}
		out = append(out, h)
		for _, b := range m.Blocks {
			n1, _ := b.Norm1.Forward(h)
			out = append(out, n1)
			a, err := b.Attn.Forward(n1)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, a)
			if h, err = b.Forward(h); err != nil {
				t.Fatal(err)
			}
			out = append(out, h)
		}
		f, _ := m.FinalNorm.Forward(h)
		out = append(out, f)
		l, err := m.LMHead.Forward(f)
		if err != nil {
			t.Fatal(err)
		}
		return append(out, l)
	}
	oc, og := run(mc, cpu), run(mg, g)
	names := []string{"tok emb", "+pos emb"}
	for i := range mc.Blocks {
		names = append(names, "block "+string(rune('0'+i/10))+string(rune('0'+i%10))+" norm1", "attn", "out")
	}
	names = append(names, "final norm", "logits")
	for i := range oc {
		d := diff(oc[i], og[i])
		t.Logf("%-18s max |cpu - webgpu| %.3g", names[i], d)
		if !(d < 1e-2) {
			t.Errorf("%s: WebGPU differs from CPU by %.3g", names[i], d)
			return
		}
	}
}
