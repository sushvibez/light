package models

import (
	"fmt"
	"os"
	"testing"

	"light/autograd"
	"light/device"
	"light/npy"
	"light/tensor"
)

// Stage-by-stage diagnostic against a transformers activation dump
// (scratch script mimi_dump.py; LIGHT_MIMI_DUMP=<npz>). Each Light stage is
// fed the reference input of that stage, so errors do not compound.
func TestMimiStagesAgainstDump(t *testing.T) {
	path := os.Getenv("LIGHT_MIMI_DUMP")
	if path == "" {
		t.Skip("set LIGHT_MIMI_DUMP")
	}
	cpu := device.CPU()
	ref, err := npy.ReadNpz(path, cpu)
	if err != nil {
		t.Fatal(err)
	}
	m := mimiLoad(t, cpu)
	audio := mimiAudio(t, cpu)
	report := func(name string, got *tensor.Tensor) {
		t.Helper()
		want := ref[name]
		if !got.Shape().Equal(want.Shape()) {
			t.Errorf("%s: shape %v want %v", name, got.Shape(), want.Shape())
			return
		}
		g, _ := got.Contiguous()
		d := maxAbsDiff(tensor.MustSlice[float32](g), tensor.MustSlice[float32](want))
		t.Logf("%-12s max abs diff %.3e", name, d)
	}
	x := audio
	for i, l := range m.Encoder {
		name := fmt.Sprintf("enc_l%d", i)
		if i > 0 {
			x = ref[fmt.Sprintf("enc_l%d", i-1)]
		}
		y, err := l.forward(x, nil)
		if err != nil {
			t.Fatal(err)
		}
		report(name, y)
	}
	tfIn, _ := autograd.Permute(ref["enc_seanet"], 0, 2, 1)
	tfIn, _ = autograd.Contiguous(tfIn)
	y, err := m.transformer(m.EncoderTF, tfIn, nil)
	if err != nil {
		t.Fatal(err)
	}
	y, _ = autograd.Permute(y, 0, 2, 1)
	report("enc_tf", y)
	lat, err := m.Downsample.forward(ref["enc_tf"], nil)
	if err != nil {
		t.Fatal(err)
	}
	report("latents", lat)
	sem, err := m.Semantic.encode(ref["latents"], 1)
	if err != nil {
		t.Fatal(err)
	}
	ac, err := m.Acoustic.encode(ref["latents"], 7)
	if err != nil {
		t.Fatal(err)
	}
	qs, _ := m.Semantic.decode(sem)
	qa, _ := m.Acoustic.decode(ac)
	q, _ := autograd.Add(qs, qa)
	report("quant", q)
	up, err := m.Upsample.forward(ref["quant"], nil)
	if err != nil {
		t.Fatal(err)
	}
	report("upsample", up)
	dIn, _ := autograd.Permute(ref["upsample"], 0, 2, 1)
	dIn, _ = autograd.Contiguous(dIn)
	y, err = m.transformer(m.DecoderTF, dIn, nil)
	if err != nil {
		t.Fatal(err)
	}
	y, _ = autograd.Permute(y, 0, 2, 1)
	report("dec_tf", y)
	x = ref["dec_tf"]
	for i, l := range m.Decoder {
		if i > 0 {
			x = ref[fmt.Sprintf("dec_l%d", i-1)]
		}
		y, err := l.forward(x, nil)
		if err != nil {
			t.Fatal(err)
		}
		report(fmt.Sprintf("dec_l%d", i), y)
	}
}
