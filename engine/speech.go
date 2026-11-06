package engine

import (
	"errors"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"light/audio"
	"light/device"
	"light/hub"
	"light/models"
	"light/tensor"
	"light/tokenizer"
)

// MimiRepo is the codec Moshi speaks through.
const MimiRepo = "kyutai/mimi"

// Speech is a loaded speech-to-speech stack: the Moshi LM, the Mimi codec
// and the SentencePiece text tokenizer. It needs the whole card (8.2 GB
// of Q8 weights), so the runtime releases the chat model when it loads.
type Speech struct {
	Spec     Spec
	Moshi    *models.Moshi
	Mimi     *models.Mimi
	Tok      *tokenizer.SentencePieceModel
	Dev      device.Device
	LoadTime time.Duration
}

// ConverseOpts controls one turn.
type ConverseOpts struct {
	ExtraSec  float64 // silence appended after the user's audio so Moshi can finish (default 2 s)
	Greedy    bool    // argmax instead of Moshi's default top-k sampling
	Seed      int64   // sampling seed (0: time-based)
	TextTemp  float64 // defaults 0.7 / 25 (kyutai)
	TextTopK  int
	AudioTemp float64 // defaults 0.8 / 250
	AudioTopK int
}

// ConverseResult is one turn: Moshi's reply as text and 24 kHz audio.
type ConverseResult struct {
	Text       string    `json:"text"`
	Tokens     []int64   `json:"tokens"`
	InputSec   float64   `json:"input_sec"`
	AudioSec   float64   `json:"audio_sec"`
	Frames     int       `json:"frames"`
	StepMS     float64   `json:"step_ms"` // wall time per 80 ms frame
	RealTimeX  float64   `json:"real_time_x"`
	SampleRate int       `json:"sample_rate"`
	Audio      []float32 `json:"-"`
}

// LoadSpeech loads a Moshi catalog entry (Spec.GGUF, or the Q8 GGUF of
// Spec.Repo) together with the Mimi codec and the text tokenizer.
func LoadSpeech(spec Spec) (*Speech, error) {
	t0 := time.Now()
	dev, err := pickDevice(spec.Device)
	if err != nil {
		return nil, err
	}
	if dev.Kind() != device.KindCUDA {
		return nil, errors.New("engine: Moshi needs CUDA (7.7 B parameters, fused Q8 matmul)")
	}
	repo := spec.Repo
	if repo == "" {
		repo = models.MoshikoRepo
	}
	path := spec.GGUF
	if path == "" {
		if path, err = hub.Download(repo, "", models.MoshikoGGUF); err != nil {
			return nil, err
		}
	}
	tokPath, err := hub.Download(repo, "", models.MoshikoTokenizer)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.LoadSentencePieceModel(tokPath)
	if err != nil {
		return nil, err
	}
	mimi, err := models.LoadMimiFromHF(MimiRepo, dev)
	if err != nil {
		return nil, fmt.Errorf("engine: Mimi codec: %w", err)
	}
	m, err := models.LoadMoshiGGUF(path, dev)
	if err != nil {
		return nil, err
	}
	s := &Speech{Spec: spec, Moshi: m, Mimi: mimi, Tok: tok, Dev: dev}
	s.Spec.Family = "moshi"
	s.LoadTime = time.Since(t0)
	return s, nil
}

// Converse runs one full-duplex turn over mono samples at any rate and
// returns Moshi's spoken reply (24 kHz) with its text.
func (s *Speech) Converse(samples []float32, rate int, opts ConverseOpts) (*ConverseResult, error) {
	start := time.Now()
	if opts.ExtraSec <= 0 {
		opts.ExtraSec = 2
	}
	if opts.TextTemp == 0 {
		opts.TextTemp = 0.7
	}
	if opts.TextTopK == 0 {
		opts.TextTopK = 25
	}
	if opts.AudioTemp == 0 {
		opts.AudioTemp = 0.8
	}
	if opts.AudioTopK == 0 {
		opts.AudioTopK = 250
	}
	sr := s.Mimi.Cfg.SampleRate
	pcm := audio.Resample(samples, rate, sr)
	if len(pcm) == 0 {
		return nil, errors.New("engine: empty audio")
	}
	x, err := tensor.New(pcm, []int{1, 1, len(pcm)}, s.Dev)
	if err != nil {
		return nil, err
	}
	var textSampler, audioSampler func([]float32) int
	if !opts.Greedy {
		seed := opts.Seed
		if seed == 0 {
			seed = time.Now().UnixNano()
		}
		rng := rand.New(rand.NewSource(seed))
		textSampler = models.TopKSampler(opts.TextTopK, opts.TextTemp, rng)
		audioSampler = models.TopKSampler(opts.AudioTopK, opts.AudioTemp, rng)
	}
	hop := s.Mimi.Cfg.HopLength()
	extra := int(opts.ExtraSec*float64(sr)/float64(hop) + 0.5)
	turn, err := s.Moshi.Converse(s.Mimi, x, extra, textSampler, audioSampler)
	if err != nil {
		return nil, err
	}
	res := &ConverseResult{Tokens: turn.TextTokens, Frames: turn.Frames, SampleRate: sr,
		InputSec: float64(len(pcm)) / float64(sr), Text: strings.TrimSpace(s.Tok.Decode(turn.Words()))}
	if turn.Audio != nil {
		host, err := turn.Audio.To(device.CPU())
		if err != nil {
			return nil, err
		}
		res.Audio = append([]float32(nil), tensor.MustSlice[float32](host)...)
		res.AudioSec = float64(len(res.Audio)) / float64(sr)
	}
	wall := time.Since(start)
	if turn.Frames > 0 {
		res.StepMS = ms(wall) / float64(turn.Frames)
	}
	if w := wall.Seconds(); w > 0 {
		res.RealTimeX = float64(turn.Frames*hop) / float64(sr) / w
	}
	return res, nil
}

// Describe is a one-line summary.
func (s *Speech) Describe() string {
	return fmt.Sprintf("%s (%.1fB params + Mimi)", s.Spec.ID, float64(s.Moshi.ParamCount())/1e9)
}
