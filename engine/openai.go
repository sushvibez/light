package engine

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"

	"light/audio"
	"light/models"
)

// Limits bounds what one HTTP request may ask of the server. Zero fields
// take DefaultLimits' values.
type Limits struct {
	JSONBytes       int64   // chat / embeddings request body (images ride inline as data URLs)
	UploadBytes     int64   // whole multipart audio request body
	AudioSeconds    float64 // decoded audio for /v1/audio/transcriptions
	ConverseSeconds float64 // decoded audio for /v1/audio/conversations
	ExtraSeconds    float64 // extra_seconds of /v1/audio/conversations
	EmbedInputs     int     // strings per embeddings request
	EmbedTokens     int     // padded tokens (inputs × longest input) per embeddings request
	MaxTokens       int     // max_tokens of a chat completion
}

// DefaultLimits are the OpenAIHandler limits.
var DefaultLimits = Limits{
	JSONBytes:       32 << 20,
	UploadBytes:     64 << 20,
	AudioSeconds:    30 * 60,
	ConverseSeconds: 5 * 60,
	ExtraSeconds:    30,
	EmbedInputs:     256,
	EmbedTokens:     64 << 10,
	MaxTokens:       1 << 17,
}

func (l Limits) withDefaults() Limits {
	d := DefaultLimits
	if l.JSONBytes <= 0 {
		l.JSONBytes = d.JSONBytes
	}
	if l.UploadBytes <= 0 {
		l.UploadBytes = d.UploadBytes
	}
	if l.AudioSeconds <= 0 {
		l.AudioSeconds = d.AudioSeconds
	}
	if l.ConverseSeconds <= 0 {
		l.ConverseSeconds = d.ConverseSeconds
	}
	if l.ExtraSeconds <= 0 {
		l.ExtraSeconds = d.ExtraSeconds
	}
	if l.EmbedInputs <= 0 {
		l.EmbedInputs = d.EmbedInputs
	}
	if l.EmbedTokens <= 0 {
		l.EmbedTokens = d.EmbedTokens
	}
	if l.MaxTokens <= 0 {
		l.MaxTokens = d.MaxTokens
	}
	return l
}

// OpenAIHandler serves the subset of the OpenAI REST API that chat clients
// use: GET /v1/models and POST /v1/chat/completions (streaming and not).
// "model" in a request may be a catalog id; if it differs from the loaded
// model, the runtime loads it first. Requests are bounded by
// DefaultLimits.
func OpenAIHandler(rt *Runtime) http.Handler { return OpenAIHandlerLimits(rt, DefaultLimits) }

// OpenAIHandlerLimits is OpenAIHandler with explicit request limits.
func OpenAIHandlerLimits(rt *Runtime, lim Limits) http.Handler {
	lim = lim.withDefaults()
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, _ *http.Request) {
		entries, _ := Catalog()
		type mdl struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			OwnedBy string `json:"owned_by"`
			Created int64  `json:"created"`
		}
		data := make([]mdl, 0, len(entries))
		for _, e := range entries {
			data = append(data, mdl{ID: e.Spec.ID, Object: "model", OwnedBy: "light", Created: 0})
		}
		writeJSON(w, map[string]any{"object": "list", "data": data})
	})
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		if !decodeJSON(w, r, lim.JSONBytes, &req) {
			return
		}
		if len(req.Messages) == 0 {
			httpErr(w, 400, "messages required")
			return
		}
		opts := GenOpts{MaxTokens: req.MaxTokens, Temperature: 0.7, TopP: 0.9}
		if req.Temperature != nil {
			opts.Temperature = float32(*req.Temperature)
		}
		if req.TopP != nil {
			opts.TopP = float32(*req.TopP)
		}
		if req.Seed != nil {
			opts.Seed = uint64(*req.Seed)
		}
		if msg := validateGenOpts(req, lim); msg != "" {
			httpErr(w, 400, msg)
			return
		}
		// The lease pins the selected model (no other request can replace
		// it or generate on it) until the response is written.
		lease, err := acquireModel(rt, req.Model)
		if err != nil {
			httpErr(w, 503, err.Error())
			return
		}
		defer lease.Release()
		m := lease.Model()
		id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
		created := time.Now().Unix()
		if !req.Stream {
			text, st, err := lease.Generate(r.Context(), req.Messages, opts, nil)
			if err != nil {
				httpErr(w, 500, err.Error())
				return
			}
			writeJSON(w, map[string]any{
				"id": id, "object": "chat.completion", "created": created, "model": m.Spec.ID,
				"choices": []any{map[string]any{
					"index": 0, "message": Message{Role: "assistant", Content: text}, "finish_reason": finish(st),
				}},
				"usage": map[string]int{"prompt_tokens": st.PromptTokens, "completion_tokens": st.Tokens, "total_tokens": st.PromptTokens + st.Tokens},
			})
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		fl, _ := w.(http.Flusher)
		send := func(v any) {
			b, _ := json.Marshal(v)
			fmt.Fprintf(w, "data: %s\n\n", b)
			if fl != nil {
				fl.Flush()
			}
		}
		chunk := func(delta map[string]any, fin any) map[string]any {
			return map[string]any{
				"id": id, "object": "chat.completion.chunk", "created": created, "model": m.Spec.ID,
				"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": fin}},
			}
		}
		send(chunk(map[string]any{"role": "assistant", "content": ""}, nil))
		_, st, err := lease.Generate(r.Context(), req.Messages, opts, func(s string) {
			send(chunk(map[string]any{"content": s}, nil))
		})
		if err != nil && !errors.Is(err, context.Canceled) {
			send(map[string]any{"error": map[string]string{"message": err.Error()}})
		} else {
			send(chunk(map[string]any{}, finish(st)))
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		if fl != nil {
			fl.Flush()
		}
	})
	// OpenAI audio API: multipart form with `file` (WAV), optional `model`
	// (catalog id of a Whisper repo), `language`, `response_format`.
	mux.HandleFunc("POST /v1/audio/transcriptions", func(w http.ResponseWriter, r *http.Request) {
		samples, rate, ok := readWAVUpload(w, r, lim.UploadBytes, lim.AudioSeconds)
		if !ok {
			return
		}
		tr, err := ensureSTT(rt, r.FormValue("model"))
		if err != nil {
			httpErr(w, 503, err.Error())
			return
		}
		res, err := tr.Transcribe(samples, rate, models.TranscribeOpts{Language: r.FormValue("language"), Prompt: r.FormValue("prompt")})
		if err != nil {
			httpErr(w, 500, err.Error())
			return
		}
		if r.FormValue("response_format") == "text" {
			w.Header().Set("Content-Type", "text/plain; charset=utf-8")
			fmt.Fprintln(w, res.Text)
			return
		}
		writeJSON(w, res)
	})
	// OpenAI embeddings API: {"model", "input": string | [string]} →
	// L2-normalised float vectors. Task prefixes ("task: search result |
	// query: …", "title: none | text: …") are the caller's to add.
	mux.HandleFunc("POST /v1/embeddings", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model          string          `json:"model"`
			Input          json.RawMessage `json:"input"`
			EncodingFormat string          `json:"encoding_format"`
		}
		if !decodeJSON(w, r, lim.JSONBytes, &req) {
			return
		}
		if req.EncodingFormat != "" && req.EncodingFormat != "float" {
			httpErr(w, 400, "only encoding_format \"float\" is supported")
			return
		}
		var texts []string
		var one string
		if json.Unmarshal(req.Input, &one) == nil {
			texts = []string{one}
		} else if json.Unmarshal(req.Input, &texts) != nil || len(texts) == 0 {
			httpErr(w, 400, "input must be a string or a non-empty array of strings")
			return
		}
		if len(texts) > lim.EmbedInputs {
			httpErr(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("%d inputs exceeds the limit of %d per request", len(texts), lim.EmbedInputs))
			return
		}
		e, err := ensureEmbedder(rt, req.Model)
		if err != nil {
			httpErr(w, 503, err.Error())
			return
		}
		vecs, tokens, err := e.EmbedLimit(texts, lim.EmbedTokens)
		if errors.Is(err, ErrEmbedTooLarge) {
			httpErr(w, http.StatusRequestEntityTooLarge, err.Error())
			return
		}
		if err != nil {
			httpErr(w, 500, err.Error())
			return
		}
		type item struct {
			Object    string    `json:"object"`
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}
		data := make([]item, len(vecs))
		for i, v := range vecs {
			data[i] = item{Object: "embedding", Index: i, Embedding: v}
		}
		writeJSON(w, map[string]any{
			"object": "list", "data": data, "model": e.Spec.ID,
			"usage": map[string]int{"prompt_tokens": tokens, "total_tokens": tokens},
		})
	})
	// Speech-to-speech (Moshi): multipart form with `file` (WAV of the
	// user's turn), optional `model`, `extra_seconds` (silence appended so
	// the reply can finish, default 2), `greedy` ("1" for argmax), `seed`,
	// and `response_format`: "json" (default: text, timings and the reply
	// as base64 WAV in audio_wav_base64) or "wav" (audio/wav body, text in
	// the X-Light-Text header).
	mux.HandleFunc("POST /v1/audio/conversations", func(w http.ResponseWriter, r *http.Request) {
		samples, rate, ok := readWAVUpload(w, r, lim.UploadBytes, lim.ConverseSeconds)
		if !ok {
			return
		}
		opts := ConverseOpts{Greedy: r.FormValue("greedy") == "1"}
		if s := r.FormValue("extra_seconds"); s != "" {
			v, err := strconv.ParseFloat(s, 64)
			if err != nil || math.IsNaN(v) || v < 0 || v > lim.ExtraSeconds {
				httpErr(w, 400, fmt.Sprintf("extra_seconds must be a number in [0, %g]", lim.ExtraSeconds))
				return
			}
			opts.ExtraSec = v
		}
		if s := r.FormValue("seed"); s != "" {
			v, err := strconv.ParseInt(s, 10, 64)
			if err != nil {
				httpErr(w, 400, "seed must be an integer")
				return
			}
			opts.Seed = v
		}
		sp, err := ensureSpeech(rt, r.FormValue("model"))
		if err != nil {
			httpErr(w, 503, err.Error())
			return
		}
		res, err := sp.Converse(samples, rate, opts)
		if err != nil {
			httpErr(w, 500, err.Error())
			return
		}
		var wav bytes.Buffer
		if err := audio.WriteWAV16(&wav, res.Audio, res.SampleRate); err != nil {
			httpErr(w, 500, err.Error())
			return
		}
		if r.FormValue("response_format") == "wav" {
			w.Header().Set("Content-Type", "audio/wav")
			w.Header().Set("X-Light-Text", strconv.Quote(res.Text))
			_, _ = w.Write(wav.Bytes())
			return
		}
		writeJSON(w, struct {
			*ConverseResult
			AudioWAV string `json:"audio_wav_base64"`
		}{res, base64.StdEncoding.EncodeToString(wav.Bytes())})
	})
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, rt.Status())
	})
	return cors(mux)
}

// tooLarge reports whether err comes from an http.MaxBytesReader limit
// (multipart wraps it without %w, so the message is checked too).
func tooLarge(err error) bool {
	var mbe *http.MaxBytesError
	return errors.As(err, &mbe) || (err != nil && strings.Contains(err.Error(), "request body too large"))
}

// decodeJSON decodes a body of at most limit bytes into v, writing a 413
// or 400 and returning false on failure.
func decodeJSON(w http.ResponseWriter, r *http.Request, limit int64, v any) bool {
	r.Body = http.MaxBytesReader(w, r.Body, limit)
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		if tooLarge(err) {
			httpErr(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("request body exceeds %d bytes", limit))
		} else {
			httpErr(w, 400, "invalid JSON: "+err.Error())
		}
		return false
	}
	return true
}

// readWAVUpload parses a multipart body of at most limit bytes and decodes
// its `file` field as WAV of at most maxSec seconds, writing the error
// response and returning ok=false on failure.
func readWAVUpload(w http.ResponseWriter, r *http.Request, limit int64, maxSec float64) (samples []float32, rate int, ok bool) {
	r.Body = http.MaxBytesReader(w, r.Body, limit)
	if err := r.ParseMultipartForm(8 << 20); err != nil {
		if tooLarge(err) {
			httpErr(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("request body exceeds %d bytes", limit))
		} else {
			httpErr(w, 400, "multipart form required: "+err.Error())
		}
		return nil, 0, false
	}
	f, _, err := r.FormFile("file")
	if err != nil {
		httpErr(w, 400, "missing file field")
		return nil, 0, false
	}
	defer f.Close()
	raw, err := io.ReadAll(f) // bounded by limit
	if err != nil {
		httpErr(w, 400, err.Error())
		return nil, 0, false
	}
	samples, rate, err = audio.DecodeWAV(raw)
	if err != nil {
		httpErr(w, 400, "only WAV input is supported: "+err.Error())
		return nil, 0, false
	}
	if sec := float64(len(samples)) / float64(rate); sec > maxSec {
		httpErr(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("audio is %.0f s, the limit is %.0f s", sec, maxSec))
		return nil, 0, false
	}
	return samples, rate, true
}

// validateGenOpts checks a chat request's numeric options; "" when valid.
func validateGenOpts(req chatRequest, lim Limits) string {
	if req.MaxTokens < 0 || req.MaxTokens > lim.MaxTokens {
		return fmt.Sprintf("max_tokens must be in [0, %d]", lim.MaxTokens)
	}
	if t := req.Temperature; t != nil && (*t < 0 || *t > 100) {
		return "temperature must be in [0, 100]"
	}
	if p := req.TopP; p != nil && (*p < 0 || *p > 1) {
		return "top_p must be in [0, 1]"
	}
	return ""
}

type chatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Stream      bool      `json:"stream"`
	MaxTokens   int       `json:"max_tokens"`
	Temperature *float64  `json:"temperature"`
	TopP        *float64  `json:"top_p"`
	Seed        *int64    `json:"seed"`
}

// acquireModel leases the chat model a request names (a catalog id, or
// ""/"default" for whatever is loaded), loading it first when another
// model is loaded. Selection and lease are atomic: the returned lease's
// model is the requested one and stays loaded until Release.
func acquireModel(rt *Runtime, id string) (*Lease, error) {
	anyModel := id == "" || strings.EqualFold(id, "default")
	l, err := rt.Acquire(func(m *Model) bool { return anyModel || id == m.Spec.ID })
	if !errors.Is(err, ErrNoModel) {
		return l, err
	}
	if anyModel {
		return nil, errors.New("no model loaded; pass a catalog id as \"model\"")
	}
	entries, _ := Catalog()
	for _, e := range entries {
		if e.Spec.ID == id {
			return rt.LoadAndAcquire(e.Spec)
		}
	}
	if m := rt.Model(); m != nil {
		return nil, fmt.Errorf("model %q not in catalog (loaded: %s)", id, m.Spec.ID)
	}
	return nil, fmt.Errorf("model %q not in catalog", id)
}

// ensureEmbedder returns the loaded embedder, loading the requested (or the
// first cached) embedding catalog entry when needed.
func ensureEmbedder(rt *Runtime, id string) (*Embedder, error) {
	if e := rt.Embedder(); e != nil && (id == "" || e.Spec.ID == id) {
		return e, nil
	}
	entries, _ := Catalog()
	for _, e := range entries {
		if e.Kind == "embedding" && (id == "" || e.Spec.ID == id) {
			return rt.LoadEmbedder(e.Spec)
		}
	}
	if id == "" {
		return nil, errors.New("no embedding model in the catalog")
	}
	return nil, fmt.Errorf("embedding model %q not in catalog", id)
}

// ensureSTT returns the loaded transcriber, loading the requested (or the
// first cached) Whisper catalog entry when needed.
func ensureSTT(rt *Runtime, id string) (*Transcriber, error) {
	if t := rt.STT(); t != nil && (id == "" || id == t.Spec.ID) {
		return t, nil
	}
	entries, _ := Catalog()
	for _, e := range entries {
		if e.Kind == "stt" && (id == "" || e.Spec.ID == id) {
			return rt.LoadSTT(e.Spec)
		}
	}
	if id != "" {
		return nil, fmt.Errorf("speech model %q not in catalog", id)
	}
	return nil, errors.New("no Whisper checkpoint in the HF cache")
}

// ensureSpeech returns the loaded Moshi stack, loading the requested (or
// the first cached) speech catalog entry when needed.
func ensureSpeech(rt *Runtime, id string) (*Speech, error) {
	if s := rt.Speech(); s != nil && (id == "" || id == s.Spec.ID) {
		return s, nil
	}
	entries, _ := Catalog()
	for _, e := range entries {
		if e.Kind == "speech" && (id == "" || e.Spec.ID == id) {
			return rt.LoadSpeech(e.Spec)
		}
	}
	if id != "" {
		return nil, fmt.Errorf("speech model %q not in catalog", id)
	}
	return nil, errors.New("no Moshi checkpoint in the HF cache (kyutai/moshiko-candle-q8)")
}

func finish(st Stats) string {
	if st.Stop == "length" {
		return "length"
	}
	return "stop"
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func httpErr(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"message": msg, "type": "server_error"}})
}

func cors(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(204)
			return
		}
		h.ServeHTTP(w, r)
	})
}
