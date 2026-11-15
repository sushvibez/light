package audio

import (
	"fmt"
	"math"
	"runtime"
	"sync"
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
	win, tw := stftTables()
	lastFrame := 0
	for i := len(src) - 1; i >= 0; i-- {
		if src[i] != 0 {
			lastFrame = min(frames-1, (i+WhisperNFFT)/WhisperHop)
			break
		}
	}
	// Each filter row's nonzero span, so the per-frame projection skips the
	// zero tails (same terms, same order).
	spanLo, spanHi := make([]int, nMels), make([]int, nMels)
	for m, row := range filters {
		lo, hi := 0, 0
		for k, w := range row {
			if w != 0 {
				if hi == 0 {
					lo = k
				}
				hi = k + 1
			}
		}
		spanLo[m], spanHi[m] = lo, hi
	}
	mel := make([]float64, nMels*frames)

	// Frames are independent: split them into contiguous ranges, one per
	// worker. Within a range two non-silent frames share one complex FFT
	// (a in the real part, b in the imaginary part); frames of silence are
	// skipped and keep mel 0, as before.
	nf := lastFrame + 1
	workers := min(runtime.GOMAXPROCS(0), (nf+melFramesPerWorker-1)/melFramesPerWorker)
	run := func(start, end int) {
		z := make([]complex128, WhisperNFFT)
		buf := make([]complex128, WhisperNFFT)
		pa := make([]float64, bins)
		pb := make([]float64, bins)
		load := func(f int, imag bool) (zero bool) {
			zero = true
			off := f * WhisperHop
			for t := range WhisperNFFT {
				v := x[off+t]
				if v != 0 {
					zero = false
				}
				if imag {
					z[t] = complex(real(z[t]), v*win[t])
				} else {
					z[t] = complex(v*win[t], 0)
				}
			}
			return zero
		}
		project := func(f int, power []float64) {
			for m := range nMels {
				row := filters[m]
				var acc float64
				for k := spanLo[m]; k < spanHi[m]; k++ {
					if w := row[k]; w != 0 {
						acc += float64(w) * power[k]
					}
				}
				mel[m*frames+f] = acc
			}
		}
		for f := start; f < end; {
			if load(f, false) {
				f++
				continue
			}
			pair := f+1 < end && !load(f+1, true)
			fft400(buf, z, tw)
			if pair {
				for k := range bins {
					zk, zn := buf[k], buf[(WhisperNFFT-k)%WhisperNFFT]
					cn := complex(real(zn), -imag(zn))
					a, b := zk+cn, zk-cn // 2·A[k], 2i·B[k]
					pa[k] = (real(a)*real(a) + imag(a)*imag(a)) / 4
					pb[k] = (real(b)*real(b) + imag(b)*imag(b)) / 4
				}
				project(f, pa)
				project(f+1, pb)
				f += 2
				continue
			}
			for k := range bins {
				c := buf[k]
				pa[k] = real(c)*real(c) + imag(c)*imag(c)
			}
			project(f, pa)
			f++ // f+1 (if in range) is silent or a fresh start
		}
	}
	if workers <= 1 {
		run(0, nf)
	} else {
		var wg sync.WaitGroup
		for w := range workers {
			wg.Go(func() { run(w*nf/workers, (w+1)*nf/workers) })
		}
		wg.Wait()
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

// melFramesPerWorker is the smallest frame range worth its own goroutine.
const melFramesPerWorker = 128

var (
	stftOnce sync.Once
	stftWin  []float64    // periodic Hann window, length NFFT
	stftTw   []complex128 // exp(-2πik/NFFT), k < NFFT
)

// stftTables returns the immutable window and FFT twiddles, built once.
func stftTables() ([]float64, []complex128) {
	stftOnce.Do(func() {
		stftWin = make([]float64, WhisperNFFT)
		stftTw = make([]complex128, WhisperNFFT)
		for i := range WhisperNFFT {
			stftWin[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/WhisperNFFT)
			a := -2 * math.Pi * float64(i) / WhisperNFFT
			stftTw[i] = complex(math.Cos(a), math.Sin(a))
		}
	})
	return stftWin, stftTw
}

// fft400Factors is WhisperNFFT = 4·4·5·5 for the mixed-radix FFT.
var fft400Factors = []int{4, 4, 5, 5}

// fft400 writes the forward DFT of src (length WhisperNFFT) into dst.
func fft400(dst, src []complex128, tw []complex128) {
	fftMixed(dst, src, WhisperNFFT, 1, fft400Factors, tw)
}

// fftMixed is a recursive decimation-in-time mixed-radix FFT: dst (length
// n) receives the DFT of src[0], src[stride], …, src[(n-1)·stride]. The
// input splits into p = factors[0] interleaved length-m subsequences whose
// DFTs land in dst[j·m:(j+1)·m]; then for every k the p values
// dst[j·m+k] are twiddled by W_n^{jk} and combined by a size-p DFT in
// place. tw holds W_N^e for the full size N, so W_n^e = tw[e·N/n mod N].
func fftMixed(dst, src []complex128, n, stride int, factors []int, tw []complex128) {
	if n == 1 {
		dst[0] = src[0]
		return
	}
	p := factors[0]
	m := n / p
	for j := range p {
		fftMixed(dst[j*m:(j+1)*m], src[j*stride:], m, stride*p, factors[1:], tw)
	}
	N := len(tw)
	step := N / n
	var t [5]complex128 // p ≤ 5
	for k := range m {
		for j := range p {
			t[j] = dst[j*m+k] * tw[(j*k*step)%N]
		}
		for q := range p {
			s := t[0]
			for j := 1; j < p; j++ {
				s += t[j] * tw[(j*q*m*step)%N] // W_n^{jqm} = W_p^{jq}
			}
			dst[q*m+k] = s
		}
	}
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
