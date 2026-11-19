// Package audio holds the host-side audio front end for speech models:
// WAV decoding, resampling and the Whisper log-mel spectrogram.
package audio

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

// ReadWAV decodes a RIFF/WAVE file (PCM 8/16/24/32-bit or IEEE float32)
// into mono float32 samples in [-1, 1], averaging channels, and returns
// the file's sample rate.
func ReadWAV(path string) ([]float32, int, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	return DecodeWAV(raw)
}

// DecodeWAV is ReadWAV on an in-memory file. It accepts integer PCM
// (format 1, 8/16/24/32-bit) and IEEE float (format 3, 32/64-bit), plain or
// WAVE_FORMAT_EXTENSIBLE, and rejects anything else — compressed codecs, a
// zero sample rate or channel count, a block alignment that does not match
// the sample layout, truncated chunks, a partial trailing frame — rather
// than decoding garbage.
func DecodeWAV(raw []byte) ([]float32, int, error) {
	if len(raw) < 12 || string(raw[0:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		return nil, 0, errors.New("audio: not a RIFF/WAVE file")
	}
	var (
		format, channels, bits, align int
		rate                          int
		haveFmt                       bool
		data                          []byte
	)
	pos := 12
	for pos+8 <= len(raw) {
		id := string(raw[pos : pos+4])
		size := int(binary.LittleEndian.Uint32(raw[pos+4 : pos+8]))
		if avail := len(raw) - (pos + 8); size > avail {
			return nil, 0, fmt.Errorf("audio: truncated %q chunk: declares %d bytes, %d present", id, size, avail)
		}
		body := raw[pos+8 : pos+8+size]
		switch id {
		case "fmt ":
			if len(body) < 16 {
				return nil, 0, errors.New("audio: short fmt chunk")
			}
			format = int(binary.LittleEndian.Uint16(body[0:2]))
			channels = int(binary.LittleEndian.Uint16(body[2:4]))
			rate = int(binary.LittleEndian.Uint32(body[4:8]))
			align = int(binary.LittleEndian.Uint16(body[12:14]))
			bits = int(binary.LittleEndian.Uint16(body[14:16]))
			if format == 0xFFFE { // WAVE_FORMAT_EXTENSIBLE: sub-format GUID
				if len(body) < 26 {
					return nil, 0, errors.New("audio: short WAVE_FORMAT_EXTENSIBLE fmt chunk")
				}
				format = int(binary.LittleEndian.Uint16(body[24:26]))
			}
			haveFmt = true
		case "data":
			data = body
		}
		pos += 8 + size + size%2
	}
	if !haveFmt || data == nil {
		return nil, 0, errors.New("audio: missing fmt or data chunk")
	}
	switch {
	case format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32):
	case format == 3 && (bits == 32 || bits == 64):
	default:
		return nil, 0, fmt.Errorf("audio: unsupported format %d / %d-bit (PCM 8/16/24/32-bit or float 32/64-bit only)", format, bits)
	}
	if channels <= 0 {
		return nil, 0, errors.New("audio: zero channels")
	}
	if rate <= 0 {
		return nil, 0, fmt.Errorf("audio: invalid sample rate %d", rate)
	}
	bytesPer := bits / 8
	if align != bytesPer*channels {
		return nil, 0, fmt.Errorf("audio: block align %d, want %d (%d channels × %d bytes)", align, bytesPer*channels, channels, bytesPer)
	}
	if len(data)%align != 0 {
		return nil, 0, fmt.Errorf("audio: data chunk of %d bytes is not a whole number of %d-byte frames", len(data), align)
	}
	frames := len(data) / align
	out := make([]float32, frames)
	for i := range frames {
		var sum float32
		for c := range channels {
			off := (i*channels + c) * bytesPer
			var v float32
			switch {
			case format == 3 && bits == 32:
				v = math.Float32frombits(binary.LittleEndian.Uint32(data[off:]))
			case format == 3 && bits == 64:
				v = float32(math.Float64frombits(binary.LittleEndian.Uint64(data[off:])))
			case bits == 8:
				v = (float32(data[off]) - 128) / 128
			case bits == 16:
				v = float32(int16(binary.LittleEndian.Uint16(data[off:]))) / 32768
			case bits == 24:
				s := int32(data[off]) | int32(data[off+1])<<8 | int32(data[off+2])<<16
				if s&0x800000 != 0 {
					s |= ^0xFFFFFF
				}
				v = float32(s) / 8388608
			default: // PCM 32-bit
				v = float32(int32(binary.LittleEndian.Uint32(data[off:]))) / 2147483648
			}
			sum += v
		}
		out[i] = sum / float32(channels)
	}
	return out, rate, nil
}

// WriteWAV16 writes mono 16-bit PCM.
func WriteWAV16(w io.Writer, samples []float32, rate int) error {
	n := len(samples) * 2
	hdr := make([]byte, 44)
	copy(hdr[0:], "RIFF")
	binary.LittleEndian.PutUint32(hdr[4:], uint32(36+n))
	copy(hdr[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(hdr[16:], 16)
	binary.LittleEndian.PutUint16(hdr[20:], 1)
	binary.LittleEndian.PutUint16(hdr[22:], 1)
	binary.LittleEndian.PutUint32(hdr[24:], uint32(rate))
	binary.LittleEndian.PutUint32(hdr[28:], uint32(rate*2))
	binary.LittleEndian.PutUint16(hdr[32:], 2)
	binary.LittleEndian.PutUint16(hdr[34:], 16)
	copy(hdr[36:], "data")
	binary.LittleEndian.PutUint32(hdr[40:], uint32(n))
	if _, err := w.Write(hdr); err != nil {
		return err
	}
	buf := make([]byte, n)
	for i, s := range samples {
		v := math.Round(float64(max(-1, min(1, s))) * 32767)
		binary.LittleEndian.PutUint16(buf[2*i:], uint16(int16(v)))
	}
	_, err := w.Write(buf)
	return err
}

// Resample converts samples from rate `from` to rate `to` with a windowed
// sinc (Blackman window, 32 input taps per side) low-passed at the lower
// Nyquist. Each output is normalised by the sum of the kernel weights it
// used, so the DC gain is exactly 1 in both directions (a constant signal
// keeps its amplitude, as scipy.signal.resample_poly / librosa do). Good
// enough for speech; use an external resampler for hi-fi work. Both rates
// must be positive.
func Resample(samples []float32, from, to int) ([]float32, error) {
	if from <= 0 || to <= 0 {
		return nil, fmt.Errorf("audio.Resample: sample rates must be positive, got %d → %d", from, to)
	}
	if from == to || len(samples) == 0 {
		return samples, nil
	}
	ratio := float64(to) / float64(from)
	outLen := int(math.Ceil(float64(len(samples)) * ratio))
	out := make([]float32, outLen)
	cutoff := math.Min(1, ratio) // relative to the input Nyquist
	const taps = 32
	for i := range out {
		center := float64(i) / ratio
		lo := int(math.Floor(center)) - taps + 1
		hi := int(math.Floor(center)) + taps
		var acc, wsum float64
		for j := max(lo, 0); j <= min(hi, len(samples)-1); j++ {
			// Blackman window over [-taps, taps].
			wpos := (float64(j) - center) / float64(taps)
			if wpos < -1 || wpos > 1 {
				continue
			}
			x := (float64(j) - center) * cutoff
			s := 1.0
			if x != 0 {
				s = math.Sin(math.Pi*x) / (math.Pi * x)
			}
			w := 0.42 + 0.5*math.Cos(math.Pi*wpos) + 0.08*math.Cos(2*math.Pi*wpos)
			acc += float64(samples[j]) * s * w
			wsum += s * w
		}
		// The ideal low-pass kernel is cutoff·sinc(cutoff·x); dividing by
		// the summed weights both applies that scale and corrects for the
		// window and the edges, so no separate cutoff factor is needed.
		if wsum != 0 {
			acc /= wsum
		}
		out[i] = float32(acc)
	}
	return out, nil
}
