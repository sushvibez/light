package models

import (
	"os"
	"testing"

	"light/device"
	"light/ops"
	"light/tensor"
)

// TestMimiStreamingMatchesFullSequence feeds the Front Center clip through
// MimiStream in chunks of 1 and 3 frames and checks that the codes and the
// decoded audio equal the full-sequence Encode / Decode of the same 17
// complete frames: conv memories, transposed-conv overlap carries and the
// sliding-window KV caches reproduce the batched causal computation.
func TestMimiStreamingMatchesFullSequence(t *testing.T) {
	for _, dn := range []string{"cpu", "cuda"} {
		dev := device.CPU()
		if dn == "cuda" {
			if !device.CUDAAvailable() {
				continue
			}
			var err error
			if dev, err = device.CUDA(0); err != nil {
				t.Fatal(err)
			}
		}
		t.Run(dn, func(t *testing.T) { mimiStreamCheck(t, dev) })
	}
}

func mimiStreamCheck(t *testing.T, dev device.Device) {
	os.Setenv("HF_HUB_OFFLINE", "1")
	m, err := LoadMimiFromHF("kyutai/mimi", dev)
	if err != nil {
		t.Skip(err)
	}
	hop := m.Cfg.HopLength()
	full := mimiAudio(t, dev)
	frames := full.Dim(2) / hop
	audio, err := full.Narrow(2, 0, frames*hop)
	if err != nil {
		t.Fatal(err)
	}
	if audio, err = audio.Contiguous(); err != nil {
		t.Fatal(err)
	}
	refCodes, err := m.Encode(audio, 8)
	if err != nil {
		t.Fatal(err)
	}
	if refCodes.Dim(2) != frames {
		t.Fatalf("full-sequence encode of %d frames gave %d", frames, refCodes.Dim(2))
	}
	refAudio, err := m.Decode(refCodes)
	if err != nil {
		t.Fatal(err)
	}
	refCodesHost := hostI64(t, refCodes)
	refAudioHost := hostF32(t, refAudio)
	for _, chunk := range []int{1, 3} {
		s := m.NewStream()
		var codes, out []*tensor.Tensor
		for f := 0; f < frames; f += chunk {
			n := min(chunk, frames-f)
			in, err := audio.Narrow(2, f*hop, n*hop)
			if err != nil {
				t.Fatal(err)
			}
			c, err := s.EncodeStep(in, 8)
			if err != nil {
				t.Fatal(err)
			}
			if c.Dim(2) != n {
				t.Fatalf("chunk %d: %d frames in, %d codes out", chunk, n, c.Dim(2))
			}
			codes = append(codes, c)
			a, err := s.DecodeStep(c)
			if err != nil {
				t.Fatal(err)
			}
			if a.Dim(2) != n*hop {
				t.Fatalf("chunk %d: %d codes in, %d samples out", chunk, n, a.Dim(2))
			}
			out = append(out, a)
		}
		allCodes, err := ops.Concat(codes, 2)
		if err != nil {
			t.Fatal(err)
		}
		allAudio, err := ops.Concat(out, 2)
		if err != nil {
			t.Fatal(err)
		}
		gotCodes := hostI64(t, allCodes)
		diff := 0
		for i := range refCodesHost {
			if gotCodes[i] != refCodesHost[i] {
				diff++
			}
		}
		if diff != 0 {
			t.Errorf("chunk %d: %d/%d codes differ from the full-sequence encode", chunk, diff, len(refCodesHost))
		}
		gotAudio := hostF32(t, allAudio)
		if d := maxAbsDiff(gotAudio, refAudioHost); d > 2e-4 {
			t.Errorf("chunk %d: decoded audio differs from full-sequence by %.2e", chunk, d)
		} else {
			t.Logf("chunk %d: %d codes identical, audio max abs diff %.1e over %d samples", chunk, len(refCodesHost), d, len(gotAudio))
		}
	}
	// Decoding alone from codes produced by the full-sequence encoder, one
	// frame at a time, must also match (this is what Moshi's reply path does).
	s := m.NewStream()
	var out []*tensor.Tensor
	for f := range frames {
		c, err := refCodes.Narrow(2, f, 1)
		if err != nil {
			t.Fatal(err)
		}
		if c, err = c.Contiguous(); err != nil {
			t.Fatal(err)
		}
		a, err := s.DecodeStep(c)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, a)
	}
	allAudio, err := ops.Concat(out, 2)
	if err != nil {
		t.Fatal(err)
	}
	if d := maxAbsDiff(hostF32(t, allAudio), refAudioHost); d > 2e-4 {
		t.Errorf("frame-wise decode differs from full-sequence by %.2e", d)
	}
}

func hostI64(t *testing.T, x *tensor.Tensor) []int64 {
	t.Helper()
	c, err := x.Contiguous()
	if err != nil {
		t.Fatal(err)
	}
	if c, err = c.To(device.CPU()); err != nil {
		t.Fatal(err)
	}
	return tensor.MustSlice[int64](c)
}

func hostF32(t *testing.T, x *tensor.Tensor) []float32 {
	t.Helper()
	c, err := x.Contiguous()
	if err != nil {
		t.Fatal(err)
	}
	if c, err = c.To(device.CPU()); err != nil {
		t.Fatal(err)
	}
	return tensor.MustSlice[float32](c)
}
