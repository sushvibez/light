package audio

import (
	"fmt"
	"math"
)

// Whisper front-end constants (openai/whisper audio.py).
const (
	WhisperSampleRate = 16000
	WhisperNFFT       = 400
	WhisperHop        = 160
	WhisperChunkSec   = 30
	WhisperFrames     = WhisperSampleRate * WhisperChunkSec / WhisperHop // 3000
)

// LogMel computes Whisper's log-mel spectrogram for one 30-second chunk.
// samples are 16 kHz mono; they are zero-padded (or truncated) to 30 s.
// filters is (nMels × (NFFT/2+1)) as shipped in preprocessor_config.json
// ("mel_filters", slaney-normalised). The result is (nMels × 3000)
// row-major, already scaled to (log10 + 4) / 4 with the 8-dB floor.
//
// STFT: periodic Hann window, hop 160, centre=true with reflect padding,
// the trailing frame dropped — identical to WhisperFeatureExtractor.
func LogMel(samples []float32, filters [][]float32) ([]float32, error) {
	n := WhisperSampleRate * WhisperChunkSec
	src := make([]float32, n)
	copy(src, samples)
	mel, _, err := LogMelFrames(src, filters)
	return mel, err
}

// LogMelFrames is LogMel for an arbitrary-length signal: no padding or
// truncation to 30 s, len(samples)/160 frames (torch.stft centre=true with
// the trailing frame dropped, as the Whisper and Qwen3-ASR feature
// extractors do). Returns the (nMels × frames) row-major spectrogram and
// the frame count.
func LogMelFrames(samples []float32, filters [][]float32) ([]float32, int, error) {
	nMels := len(filters)
	bins := WhisperNFFT/2 + 1
	if nMels == 0 || len(filters[0]) != bins {
		return nil, 0, fmt.Errorf("audio.LogMel: filters must be (nMels × %d), got %d × %d", bins, nMels, len(filters[0]))
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

// MelFilterBank builds the slaney-scaled, slaney-normalised triangular
// filterbank of transformers' `mel_filter_bank` (what Whisper ships in
// preprocessor_config.json and what Qwen3-ASR computes at load), as
// (nMels × (nfft/2+1)) rows for LogMel.
func MelFilterBank(nMels, nfft, sampleRate int, fMin, fMax float64) [][]float32 {
	bins := nfft/2 + 1
	hzToMel := func(f float64) float64 {
		if f >= 1000 {
			return 15 + math.Log(f/1000)*27/math.Log(6.4)
		}
		return 3 * f / 200
	}
	melToHz := func(m float64) float64 {
		if m >= 15 {
			return 1000 * math.Exp(math.Log(6.4)/27*(m-15))
		}
		return 200 * m / 3
	}
	melLo, melHi := hzToMel(fMin), hzToMel(fMax)
	edges := make([]float64, nMels+2)
	for i := range edges {
		edges[i] = melToHz(melLo + (melHi-melLo)*float64(i)/float64(nMels+1))
	}
	out := make([][]float32, nMels)
	for m := range nMels {
		row := make([]float32, bins)
		lo, mid, hi := edges[m], edges[m+1], edges[m+2]
		enorm := 2 / (hi - lo)
		for k := range bins {
			f := float64(k) * float64(sampleRate) / 2 / float64(bins-1)
			down := (f - lo) / (mid - lo)
			up := (hi - f) / (hi - mid)
			v := math.Max(0, math.Min(down, up))
			row[k] = float32(v * enorm)
		}
		out[m] = row
	}
	return out
}
