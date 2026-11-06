package engine

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"light/audio"
	"light/models"
)

// OpenAIHandler serves the subset of the OpenAI REST API that chat clients
// use: GET /v1/models and POST /v1/chat/completions (streaming and not).
// "model" in a request may be a catalog id; if it differs from the loaded
// model, the runtime loads it first.
func OpenAIHandler(rt *Runtime) http.Handler {
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
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			httpErr(w, 400, "invalid JSON: "+err.Error())
			return
		}
		if len(req.Messages) == 0 {
			httpErr(w, 400, "messages required")
			return
		}
		m, err := ensureModel(rt, req.Model)
		if err != nil {
			httpErr(w, 503, err.Error())
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
		id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
		created := time.Now().Unix()
		if !req.Stream {
			text, st, err := rt.Generate(r.Context(), req.Messages, opts, nil)
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
		_, st, err := rt.Generate(r.Context(), req.Messages, opts, func(s string) {
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
		if err := r.ParseMultipartForm(64 << 20); err != nil {
			httpErr(w, 400, "multipart form required: "+err.Error())
			return
		}
		f, _, err := r.FormFile("file")
		if err != nil {
			httpErr(w, 400, "missing file field")
			return
		}
		defer f.Close()
		raw, err := io.ReadAll(f)
		if err != nil {
			httpErr(w, 400, err.Error())
			return
		}
		tr, err := ensureSTT(rt, r.FormValue("model"))
		if err != nil {
			httpErr(w, 503, err.Error())
			return
		}
		samples, rate, err := audio.DecodeWAV(raw)
		if err != nil {
			httpErr(w, 400, "only WAV input is supported: "+err.Error())
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
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			httpErr(w, 400, "invalid JSON: "+err.Error())
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
		e, err := ensureEmbedder(rt, req.Model)
		if err != nil {
			httpErr(w, 503, err.Error())
			return
		}
		vecs, tokens, err := e.Embed(texts)
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
		if err := r.ParseMultipartForm(64 << 20); err != nil {
			httpErr(w, 400, "multipart form required: "+err.Error())
			return
		}
		f, _, err := r.FormFile("file")
		if err != nil {
			httpErr(w, 400, "missing file field")
			return
		}
		defer f.Close()
		raw, err := io.ReadAll(f)
		if err != nil {
			httpErr(w, 400, err.Error())
			return
		}
		samples, rate, err := audio.DecodeWAV(raw)
		if err != nil {
			httpErr(w, 400, "only WAV input is supported: "+err.Error())
			return
		}
		sp, err := ensureSpeech(rt, r.FormValue("model"))
		if err != nil {
			httpErr(w, 503, err.Error())
			return
		}
		opts := ConverseOpts{Greedy: r.FormValue("greedy") == "1"}
		opts.ExtraSec, _ = strconv.ParseFloat(r.FormValue("extra_seconds"), 64)
		opts.Seed, _ = strconv.ParseInt(r.FormValue("seed"), 10, 64)
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

type chatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Stream      bool      `json:"stream"`
	MaxTokens   int       `json:"max_tokens"`
	Temperature *float64  `json:"temperature"`
	TopP        *float64  `json:"top_p"`
	Seed        *int64    `json:"seed"`
}

func ensureModel(rt *Runtime, id string) (*Model, error) {
	m := rt.Model()
	if m != nil && (id == "" || id == m.Spec.ID || strings.EqualFold(id, "default")) {
		return m, nil
	}
	if id == "" {
		return nil, errors.New("no model loaded; pass a catalog id as \"model\"")
	}
	entries, _ := Catalog()
	for _, e := range entries {
		if e.Spec.ID == id {
			return rt.Load(e.Spec)
		}
	}
	if m != nil {
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
