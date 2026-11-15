package audio

import (
	"bytes"
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

// R3: a constant keeps unit amplitude through downsampling and
// upsampling (the downsampler used to scale it by (to/from)²: 1/9 at
// 48k → 16k).
func TestResampleConstantUnitGain(t *testing.T) {
	for _, c := range []struct{ from, to int }{{48000, 16000}, {16000, 48000}, {44100, 16000}, {16000, 24000}, {24000, 16000}} {
		in := make([]float32, c.from/10) // 100 ms
		for i := range in {
			in[i] = 1
		}
		out, err := Resample(in, c.from, c.to)
		if err != nil {
			t.Fatal(err)
		}
		if want := int(math.Ceil(float64(len(in)) * float64(c.to) / float64(c.from))); len(out) != want {
			t.Fatalf("%d→%d: %d samples, want %d", c.from, c.to, len(out), want)
		}
		for i, v := range out {
			if math.Abs(float64(v)-1) > 1e-5 {
				t.Fatalf("%d→%d: out[%d] = %v, want 1", c.from, c.to, i, v)
			}
		}
	}
}

// A passband tone keeps its amplitude and frequency; a tone above the
// output Nyquist is suppressed (anti-aliasing).
func TestResampleTones(t *testing.T) {
	tone := func(f float64, rate, n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(0.5 * math.Sin(2*math.Pi*f*float64(i)/float64(rate)))
		}
		return s
	}
	// rms over the interior, skipping the kernel's edge transients.
	rms := func(x []float32) float64 {
		var s float64
		x = x[len(x)/8 : len(x)-len(x)/8]
		for _, v := range x {
			s += float64(v) * float64(v)
		}
		return math.Sqrt(s / float64(len(x)))
	}
	for _, c := range []struct{ from, to int }{{48000, 16000}, {16000, 48000}} {
		in := tone(1000, c.from, c.from/5)
		out, err := Resample(in, c.from, c.to)
		if err != nil {
			t.Fatal(err)
		}
		want := tone(1000, c.to, len(out))
		var maxErr float64
		for i := len(out) / 8; i < len(out)-len(out)/8; i++ {
			maxErr = max(maxErr, math.Abs(float64(out[i]-want[i])))
		}
		if maxErr > 5e-3 {
			t.Errorf("%d→%d 1 kHz tone: max error %.4g vs the ideal tone", c.from, c.to, maxErr)
		}
	}
	// 12 kHz at 48 kHz is above 16 kHz's 8 kHz Nyquist.
	out, err := Resample(tone(12000, 48000, 48000/5), 48000, 16000)
	if err != nil {
		t.Fatal(err)
	}
	if r := rms(out); r > 0.01 {
		t.Errorf("12 kHz tone 48k→16k: rms %.4g, want it filtered out (input rms 0.354)", r)
	}
}

func TestResampleRejectsBadRates(t *testing.T) {
	for _, c := range []struct{ from, to int }{{0, 16000}, {16000, 0}, {-8000, 16000}, {0, 0}} {
		if _, err := Resample([]float32{1, 2, 3}, c.from, c.to); err == nil {
			t.Errorf("Resample %d→%d: no error", c.from, c.to)
		}
	}
}

// wav assembles a RIFF/WAVE file from a fmt chunk body and data.
func wav(fmtBody, data []byte) []byte {
	var b bytes.Buffer
	b.WriteString("RIFF")
	_ = binary.Write(&b, binary.LittleEndian, uint32(4+8+len(fmtBody)+8+len(data)))
	b.WriteString("WAVEfmt ")
	_ = binary.Write(&b, binary.LittleEndian, uint32(len(fmtBody)))
	b.Write(fmtBody)
	b.WriteString("data")
	_ = binary.Write(&b, binary.LittleEndian, uint32(len(data)))
	b.Write(data)
	return b.Bytes()
}

func fmtChunk(format, channels, rate, align, bits int) []byte {
	b := make([]byte, 16)
	binary.LittleEndian.PutUint16(b[0:], uint16(format))
	binary.LittleEndian.PutUint16(b[2:], uint16(channels))
	binary.LittleEndian.PutUint32(b[4:], uint32(rate))
	binary.LittleEndian.PutUint32(b[8:], uint32(rate*align))
	binary.LittleEndian.PutUint16(b[12:], uint16(align))
	binary.LittleEndian.PutUint16(b[14:], uint16(bits))
	return b
}

func TestDecodeWAVRoundTrip(t *testing.T) {
	in := []float32{0, 0.5, -0.5, 1, -1}
	var b bytes.Buffer
	if err := WriteWAV16(&b, in, 16000); err != nil {
		t.Fatal(err)
	}
	out, rate, err := DecodeWAV(b.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if rate != 16000 || len(out) != len(in) {
		t.Fatalf("rate %d, %d samples", rate, len(out))
	}
	for i := range in {
		if math.Abs(float64(out[i]-in[i])) > 1e-4 {
			t.Fatalf("sample %d: %v, want %v", i, out[i], in[i])
		}
	}
	// Stereo float32 averages the channels.
	data := make([]byte, 8)
	binary.LittleEndian.PutUint32(data[0:], math.Float32bits(0.25))
	binary.LittleEndian.PutUint32(data[4:], math.Float32bits(0.75))
	out, _, err = DecodeWAV(wav(fmtChunk(3, 2, 8000, 8, 32), data))
	if err != nil || len(out) != 1 || out[0] != 0.5 {
		t.Fatalf("stereo float: %v %v", out, err)
	}
}

// R17: malformed or unsupported files are errors, not garbage.
func TestDecodeWAVRejects(t *testing.T) {
	pcm := make([]byte, 8) // four 16-bit mono frames
	good := wav(fmtChunk(1, 1, 16000, 2, 16), pcm)
	if _, _, err := DecodeWAV(good); err != nil {
		t.Fatalf("baseline: %v", err)
	}
	for _, c := range []struct {
		name, want string
		raw        []byte
	}{
		{"not RIFF", "RIFF", []byte("hello world, not a wav")},
		{"ADPCM codec", "unsupported format", wav(fmtChunk(2, 1, 16000, 2, 16), pcm)},
		{"mu-law codec", "unsupported format", wav(fmtChunk(7, 1, 16000, 1, 8), pcm)},
		{"float 16-bit", "unsupported format", wav(fmtChunk(3, 1, 16000, 2, 16), pcm)},
		{"PCM 12-bit", "unsupported format", wav(fmtChunk(1, 1, 16000, 2, 12), pcm)},
		{"zero rate", "sample rate", wav(fmtChunk(1, 1, 0, 2, 16), pcm)},
		{"zero channels", "channels", wav(fmtChunk(1, 0, 16000, 0, 16), pcm)},
		{"bad block align", "block align", wav(fmtChunk(1, 1, 16000, 4, 16), pcm)},
		{"partial frame", "whole number", wav(fmtChunk(1, 1, 16000, 2, 16), pcm[:7])},
		{"truncated data chunk", "truncated", good[:len(good)-3]},
		{"missing data", "missing", good[:12+8+16]},
		{"short fmt", "short fmt", wav([]byte{1, 0, 1, 0}, pcm)},
	} {
		_, _, err := DecodeWAV(c.raw)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err %v, want one mentioning %q", c.name, err, c.want)
		}
	}
}

// R28: nil, empty and ragged filterbanks are errors, not panics.
func TestLogMelFramesFilterValidation(t *testing.T) {
	samples := make([]float32, 4*WhisperHop)
	for i := range samples {
		samples[i] = float32(math.Sin(float64(i) / 7))
	}
	bins := WhisperNFFT/2 + 1
	row := func(n int) []float32 {
		r := make([]float32, n)
		for i := range r {
			r[i] = 1
		}
		return r
	}
	for name, f := range map[string][][]float32{
		"nil":         nil,
		"empty":       {},
		"empty row":   {{}},
		"short first": {row(bins - 1), row(bins)},
		"long later":  {row(bins), row(bins + 5)},
		"short later": {row(bins), row(bins), row(3)},
	} {
		if _, _, err := LogMelFrames(samples, f); err == nil {
			t.Errorf("%s filters: no error", name)
		}
		if _, err := LogMel(samples, f); err == nil {
			t.Errorf("%s filters: LogMel no error", name)
		}
	}
	mel, frames, err := LogMelFrames(samples, [][]float32{row(bins), row(bins)})
	if err != nil || frames != 4 || len(mel) != 2*4 {
		t.Fatalf("valid filters: %d frames, %d values, %v", frames, len(mel), err)
	}
}
