package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"

	"github.com/wailsapp/wails/v3/pkg/application"

	"light/audio"
	"light/engine"
	"light/models"
)

// Token is streamed to the frontend as the "token" event.
type Token struct {
	Request int64  `json:"request"`
	Text    string `json:"text"`
}

// Done closes a request as the "done" event.
type Done struct {
	Request int64        `json:"request"`
	Error   string       `json:"error,omitempty"`
	Stats   engine.Stats `json:"stats"`
}

// StudioService is the Go API the React frontend calls through Wails
// bindings (frontend/bindings/light/studio).
type StudioService struct {
	rt      *engine.Runtime
	app     *application.App
	apiAddr string
	seq     atomic.Int64
}

func NewStudioService(rt *engine.Runtime) *StudioService { return &StudioService{rt: rt} }

// Models lists everything the catalog found on disk.
func (s *StudioService) Models() ([]engine.Entry, error) { return engine.Catalog() }

// Status reports load state and memory use.
func (s *StudioService) Status() engine.Status { return s.rt.Status() }

// APIAddr is where the OpenAI-compatible server listens.
func (s *StudioService) APIAddr() string { return "http://" + s.apiAddr + "/v1" }

// Load replaces the current model. Blocks until weights are on the device.
func (s *StudioService) Load(spec engine.Spec) (engine.Status, error) {
	if _, err := s.rt.Load(spec); err != nil {
		return s.rt.Status(), err
	}
	return s.rt.Status(), nil
}

// Unload frees the current model.
func (s *StudioService) Unload() error { return s.rt.Unload() }

// Chat starts a streaming generation and returns its request id; tokens
// arrive as "token" events and the end as a "done" event.
func (s *StudioService) Chat(msgs []engine.Message, opts engine.GenOpts) (int64, error) {
	if s.rt.Model() == nil {
		return 0, errors.New("no model loaded")
	}
	if s.rt.Model().Busy() {
		return 0, engine.ErrBusy
	}
	id := s.seq.Add(1)
	go func() {
		_, st, err := s.rt.Generate(context.Background(), msgs, opts, func(t string) {
			s.app.Event.Emit("token", Token{Request: id, Text: t})
		})
		d := Done{Request: id, Stats: st}
		if err != nil {
			d.Error = err.Error()
		}
		s.app.Event.Emit("done", d)
	}()
	return id, nil
}

// Stop cancels the running generation.
func (s *StudioService) Stop() { s.rt.Stop() }

// LoadSTT loads a Whisper catalog entry into the speech-to-text slot.
func (s *StudioService) LoadSTT(spec engine.Spec) (engine.Status, error) {
	if _, err := s.rt.LoadSTT(spec); err != nil {
		return s.rt.Status(), err
	}
	return s.rt.Status(), nil
}

// UnloadSTT frees the speech-to-text model.
func (s *StudioService) UnloadSTT() { s.rt.UnloadSTT() }

// LoadSpeech loads a Moshi catalog entry into the speech-to-speech slot
// (the chat model is released: Moshi needs the whole card).
func (s *StudioService) LoadSpeech(spec engine.Spec) (engine.Status, error) {
	if _, err := s.rt.LoadSpeech(spec); err != nil {
		return s.rt.Status(), err
	}
	return s.rt.Status(), nil
}

// UnloadSpeech frees the speech-to-speech stack.
func (s *StudioService) UnloadSpeech() { s.rt.UnloadSpeech() }

// ConverseResult is a Moshi turn plus where the reply WAV was written.
type ConverseResult struct {
	engine.ConverseResult
	ReplyPath string `json:"reply_path"`
}

// ConverseFile opens a file picker, sends the chosen WAV through Moshi
// (loading the first cached speech model if none is) and writes the spoken
// reply next to it as <name>.moshi.wav. extraSec is the silence appended
// so the reply can finish. Returns an empty Text when cancelled.
func (s *StudioService) ConverseFile(extraSec float64) (*ConverseResult, error) {
	path, err := s.app.Dialog.OpenFile().
		SetTitle("Choose your spoken turn").
		AddFilter("WAV audio", "*.wav").
		PromptForSingleSelection()
	if err != nil {
		return nil, err
	}
	if path == "" {
		return &ConverseResult{}, nil
	}
	sp := s.rt.Speech()
	if sp == nil {
		entries, _ := engine.Catalog()
		for _, e := range entries {
			if e.Kind == "speech" {
				if sp, err = s.rt.LoadSpeech(e.Spec); err != nil {
					return nil, err
				}
				break
			}
		}
		if sp == nil {
			return nil, errors.New("no Moshi checkpoint on disk (download kyutai/moshiko-candle-q8)")
		}
	}
	samples, rate, err := audio.ReadWAV(path)
	if err != nil {
		return nil, err
	}
	res, err := sp.Converse(samples, rate, engine.ConverseOpts{ExtraSec: extraSec})
	if err != nil {
		return nil, err
	}
	out := strings.TrimSuffix(path, filepath.Ext(path)) + ".moshi.wav"
	f, err := os.Create(out)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if err := audio.WriteWAV16(f, res.Audio, res.SampleRate); err != nil {
		return nil, err
	}
	return &ConverseResult{ConverseResult: *res, ReplyPath: out}, nil
}

// LoadEmbedder loads an embedding catalog entry (served on /v1/embeddings).
func (s *StudioService) LoadEmbedder(spec engine.Spec) (engine.Status, error) {
	if _, err := s.rt.LoadEmbedder(spec); err != nil {
		return s.rt.Status(), err
	}
	return s.rt.Status(), nil
}

// UnloadEmbedder frees the embedding model.
func (s *StudioService) UnloadEmbedder() { s.rt.UnloadEmbedder() }

// TranscribeFile opens a file picker and transcribes the chosen WAV with
// the loaded Whisper model (loading the first cached one if none is).
// language "" lets Whisper detect it. Returns an empty Text when cancelled.
func (s *StudioService) TranscribeFile(language string) (*engine.TranscribeResult, error) {
	path, err := s.app.Dialog.OpenFile().
		SetTitle("Choose audio to transcribe").
		AddFilter("WAV audio", "*.wav").
		PromptForSingleSelection()
	if err != nil {
		return nil, err
	}
	if path == "" {
		return &engine.TranscribeResult{}, nil
	}
	tr := s.rt.STT()
	if tr == nil {
		entries, _ := engine.Catalog()
		for _, e := range entries {
			if e.Kind == "stt" {
				if tr, err = s.rt.LoadSTT(e.Spec); err != nil {
					return nil, err
				}
				break
			}
		}
		if tr == nil {
			return nil, errors.New("no Whisper checkpoint on disk (download openai/whisper-tiny or larger)")
		}
	}
	return tr.TranscribeWAV(path, models.TranscribeOpts{Language: language})
}
