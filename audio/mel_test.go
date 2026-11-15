package audio

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// logMelFramesRef is the original LogMelFrames (tables rebuilt per call, a
// direct 400-point real DFT per frame), the reference for the fast path.
func logMelFramesRef(samples []float32, filters [][]float32) ([]float32, int, error) {
	nMels := len(filters)
	bins := WhisperNFFT/2 + 1
	if nMels == 0 {
		return nil, 0, fmt.Errorf("audio.LogMel: filters must be (nMels × %d), got no rows", bins)
	}
	for m, row := range filters {
		if len(row) != bins {
			return nil, 0, fmt.Errorf("audio.LogMel: filters must be (nMels × %d), row %d has %d bins", bins, m, len(row))
		}
	}
	n := len(samples)
	frames := n / WhisperHop
	if frames == 0 {
		return nil, 0, fmt.Errorf("audio.LogMel: %d samples is shorter than one hop", n)
	}
	x := make([]float64, n+WhisperNFFT) // reflect-padded by NFFT/2 each side
	half := WhisperNFFT / 2
	src := samples
	for i, s := range src {
		x[half+i] = float64(s)
	}
	for i := 1; i <= half && i < n; i++ { // reflect padding (no edge repeat), as torch.stft
		x[half-i] = x[half+i]
		x[half+n-1+i] = x[half+n-1-i]
	}
	window := make([]float64, WhisperNFFT)
	for i := range window {
		window[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/WhisperNFFT)
	}
	// Twiddles for a direct real DFT; frames of silence are skipped.
	cosT := make([]float64, bins*WhisperNFFT)
	sinT := make([]float64, bins*WhisperNFFT)
	for k := range bins {
		for t := range WhisperNFFT {
			a := 2 * math.Pi * float64(k) * float64(t) / WhisperNFFT
			cosT[k*WhisperNFFT+t] = math.Cos(a)
			sinT[k*WhisperNFFT+t] = math.Sin(a)
		}
	}
	lastFrame := 0
	for i := len(src) - 1; i >= 0; i-- {
		if src[i] != 0 {
			lastFrame = min(frames-1, (i+WhisperNFFT)/WhisperHop)
			break
		}
	}
	power := make([]float64, bins)
	frame := make([]float64, WhisperNFFT)
	mel := make([]float64, nMels*frames)
	for f := 0; f <= lastFrame; f++ {
		allZero := true
		for t := range WhisperNFFT {
			v := x[f*WhisperHop+t]
			frame[t] = v * window[t]
			if v != 0 {
				allZero = false
			}
		}
		if allZero {
			continue
		}
		for k := range bins {
			var re, im float64
			ct := cosT[k*WhisperNFFT : (k+1)*WhisperNFFT]
			st := sinT[k*WhisperNFFT : (k+1)*WhisperNFFT]
			for t, v := range frame {
				re += v * ct[t]
				im -= v * st[t]
			}
			power[k] = re*re + im*im
		}
		for m := range nMels {
			var acc float64
			for k, w := range filters[m] {
				if w != 0 {
					acc += float64(w) * power[k]
				}
			}
			mel[m*frames+f] = acc
		}
	}
	out := make([]float32, len(mel))
	maxLog := math.Inf(-1)
	for i, v := range mel {
		l := math.Log10(math.Max(v, 1e-10))
		mel[i] = l
		if l > maxLog {
			maxLog = l
		}
	}
	floor := maxLog - 8
	for i, l := range mel {
		out[i] = float32((math.Max(l, floor) + 4) / 4)
	}
	return out, frames, nil
}

func randomAudio(n int, seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, 11))
	out := make([]float32, n)
	for i := range out {
		// speech-ish: two tones plus noise, with a silent stretch.
		t := float64(i) / WhisperSampleRate
		v := 0.3*math.Sin(2*math.Pi*220*t) + 0.2*math.Sin(2*math.Pi*1375.5*t) + 0.1*r.NormFloat64()
		if i > n/3 && i < n/3+4000 {
			v = 0
		}
		out[i] = float32(v)
	}
	return out
}

// TestLogMelFramesMatchesDirectDFT: the fast path reproduces the direct-DFT
// spectrogram on random audio of several lengths (shorter than the reflect
// pad, inner and trailing silence).
func TestLogMelFramesMatchesDirectDFT(t *testing.T) {
	for _, nMels := range []int{80, 128} {
		filters := MelFilterBank(nMels, WhisperNFFT, WhisperSampleRate, 0, 8000)
		for _, n := range []int{160, 199, 480, 4321, 16000*3 + 77, 16000 * 5} {
			x := randomAudio(n, uint64(n))
			if n == 16000*5 {
				for i := n / 2; i < n; i++ {
					x[i] = 0 // trailing silence: skipped frames
				}
			}
			want, wf, err := logMelFramesRef(x, filters)
			if err != nil {
				t.Fatal(err)
			}
			got, gf, err := LogMelFrames(x, filters)
			if err != nil {
				t.Fatal(err)
			}
			if gf != wf || len(got) != len(want) {
				t.Fatalf("n=%d: %d frames / %d values, want %d / %d", n, gf, len(got), wf, len(want))
			}
			var worst float64
			for i := range want {
				d := math.Abs(float64(got[i]-want[i])) / math.Max(math.Abs(float64(want[i])), 1e-3)
				worst = math.Max(worst, d)
			}
			t.Logf("nMels=%d n=%d: worst relative diff %.3g", nMels, n, worst)
			if worst > 1e-5 {
				t.Fatalf("nMels=%d n=%d: worst relative diff %.3g", nMels, n, worst)
			}
		}
	}
}

func benchMel(b *testing.B, f func([]float32, [][]float32) ([]float32, int, error)) {
	filters := MelFilterBank(128, WhisperNFFT, WhisperSampleRate, 0, 8000)
	x := randomAudio(WhisperSampleRate*WhisperChunkSec, 1)
	b.ResetTimer()
	for range b.N {
		if _, _, err := f(x, filters); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLogMelFrames30s(b *testing.B) {
	b.Run("new", func(b *testing.B) { benchMel(b, LogMelFrames) })
	b.Run("ref", func(b *testing.B) { benchMel(b, logMelFramesRef) })
}
