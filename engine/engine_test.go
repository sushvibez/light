package engine

import (
	"bufio"
	"bytes"
	"encoding/json"
	"math"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"light/models"
)

func TestTemplates(t *testing.T) {
	msgs := []Message{{Role: "system", Content: "Be brief."}, {Role: "user", Content: "Hi"}, {Role: "assistant", Content: "Hello"}, {Role: "user", Content: "Bye"}}
	want := map[string]string{
		"chatml":  "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n",
		"llama3":  "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nBye<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
		"phi3":    "<|system|>\nBe brief.<|end|>\n<|user|>\nHi<|end|>\n<|assistant|>\nHello<|end|>\n<|user|>\nBye<|end|>\n<|assistant|>\n",
		"phi4":    "<|system|>Be brief.<|end|><|user|>Hi<|end|><|assistant|>Hello<|end|><|user|>Bye<|end|><|assistant|>",
		"mistral": "<s>[INST] Be brief.\n\nHi [/INST] Hello</s>[INST] Bye [/INST]",
		"gemma":   "<bos><start_of_turn>user\nBe brief.\n\nHi<end_of_turn>\n<start_of_turn>model\nHello<end_of_turn>\n<start_of_turn>user\nBye<end_of_turn>\n<start_of_turn>model\n",
	}
	for fam, w := range want {
		if got := templates[fam].Render(msgs); got != w {
			t.Errorf("%s:\n got %q\nwant %q", fam, got, w)
		}
	}
}

func TestParamsFromName(t *testing.T) {
	for name, want := range map[string]string{
		"Qwen/Qwen2-0.5B-Instruct": "0.5B", "qwen2-0.5b-instruct-q4_k_m.gguf": "0.5B",
		"meta-llama/Llama-3.2-1B-Instruct": "1B", "google/embeddinggemma-300m": "300M",
		"microsoft/Phi-3.5-mini-instruct": "",
	} {
		if got := paramsFromName(name); got != want {
			t.Errorf("%s: got %q want %q", name, got, want)
		}
	}
}

func qwenGGUF(t *testing.T) Spec {
	t.Helper()
	entries, err := Catalog()
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if filepath.Base(e.Spec.GGUF) == "qwen2-0.5b-instruct-q4_k_m.gguf" {
			if e.Spec.Repo == "" {
				t.Fatal("catalog did not match a tokenizer repo for the Qwen2 GGUF")
			}
			return e.Spec
		}
	}
	t.Skip("qwen2-0.5b-instruct-q4_k_m.gguf not in ~/.cache/light/gguf")
	return Spec{}
}

// End-to-end: catalog → load GGUF → OpenAI handler, non-streaming and SSE.
func TestOpenAIServerQwenGGUF(t *testing.T) {
	if os.Getenv("HF_HUB_OFFLINE") == "" {
		os.Setenv("HF_HUB_OFFLINE", "1")
	}
	spec := qwenGGUF(t)
	rt := NewRuntime()
	if _, err := rt.Load(spec); err != nil {
		t.Fatal(err)
	}
	if rt.Model().Spec.Family != "chatml" {
		t.Fatalf("template = %s, want chatml", rt.Model().Spec.Family)
	}
	srv := httptest.NewServer(OpenAIHandler(rt))
	defer srv.Close()

	body := `{"messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"temperature":0,"max_tokens":16}`
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	var out struct {
		Choices []struct {
			Message      Message `json:"message"`
			FinishReason string  `json:"finish_reason"`
		}
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		}
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if len(out.Choices) != 1 || !strings.Contains(out.Choices[0].Message.Content, "4") {
		t.Fatalf("unexpected completion: %+v", out)
	}
	t.Logf("reply %q (%s, %d tok)", out.Choices[0].Message.Content, out.Choices[0].FinishReason, out.Usage.CompletionTokens)

	body = `{"messages":[{"role":"user","content":"Name three colours."}],"temperature":0,"max_tokens":24,"stream":true}`
	resp, err = http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content-type %q", ct)
	}
	var text strings.Builder
	chunks, done := 0, false
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			done = true
			break
		}
		var ch struct {
			Choices []struct {
				Delta        map[string]string `json:"delta"`
				FinishReason *string           `json:"finish_reason"`
			}
		}
		if err := json.Unmarshal([]byte(payload), &ch); err != nil {
			t.Fatalf("bad chunk %s: %v", payload, err)
		}
		chunks++
		text.WriteString(ch.Choices[0].Delta["content"])
	}
	if !done || chunks < 3 || text.Len() == 0 {
		t.Fatalf("stream: done=%v chunks=%d text=%q", done, chunks, text.String())
	}
	t.Logf("streamed %d chunks: %q", chunks, text.String())

	// /v1/models lists the catalog.
	resp, err = http.Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	var ml struct{ Data []struct{ ID string } }
	_ = json.NewDecoder(resp.Body).Decode(&ml)
	resp.Body.Close()
	if len(ml.Data) == 0 {
		t.Fatal("/v1/models empty")
	}
}

// Llama-3.2 GGUF: tokenizer borrowed from the cached HF repo, llama3 template,
// BOS written by the template, EOS from generation_config.json.
func TestGenerateLlama3GGUF(t *testing.T) {
	os.Setenv("HF_HUB_OFFLINE", "1")
	entries, _ := Catalog()
	var spec *Spec
	for _, e := range entries {
		if strings.HasPrefix(filepath.Base(e.Spec.GGUF), "llama-3.2-1b") {
			s := e.Spec
			spec = &s
		}
	}
	if spec == nil {
		t.Skip("no llama-3.2 GGUF in catalog")
	}
	m, err := Load(*spec)
	if err != nil {
		t.Fatal(err)
	}
	if m.Spec.Family != "llama3" || !strings.HasPrefix(m.Spec.Repo, "meta-llama/") {
		t.Fatalf("family %s repo %s", m.Spec.Family, m.Spec.Repo)
	}
	text, st, err := m.Generate(t.Context(), []Message{{Role: "user", Content: "What is the capital of France? One word."}}, GenOpts{Temperature: 0, MaxTokens: 12}, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%q stop=%s %.1f tok/s", text, st.Stop, st.TokPerSec)
	if !strings.Contains(text, "Paris") || st.Stop != "eos" {
		t.Fatalf("reply %q stop %s", text, st.Stop)
	}
}

// Gemma 3 1B through the engine at BF16: gemma template, <bos> from the
// template, EOS 1/106. Greedy reply must match transformers ("2 + 2 = 4").
func TestGenerateGemma3BF16(t *testing.T) {
	requireHostMemGB(t, 8)
	releaseModelMemory(t)
	os.Setenv("HF_HUB_OFFLINE", "1")
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--unsloth--gemma-3-1b-it/snapshots/*/model.safetensors")); len(m) == 0 {
		t.Skip("unsloth/gemma-3-1b-it not in HF cache")
	}
	m, err := Load(Spec{ID: "unsloth/gemma-3-1b-it", Repo: "unsloth/gemma-3-1b-it", DType: "bf16", Device: ""})
	if err != nil {
		t.Fatal(err)
	}
	if m.Spec.Family != "gemma" {
		t.Fatalf("family %s", m.Spec.Family)
	}
	text, st, err := m.Generate(t.Context(), []Message{{Role: "user", Content: "What is 2+2?"}}, GenOpts{Temperature: 0, MaxTokens: 16}, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%q stop=%s %.1f tok/s prefill %.0f ms", text, st.Stop, st.TokPerSec, st.PrefillMS)
	if strings.TrimSpace(text) != "2 + 2 = 4" || st.Stop != "eos" {
		t.Fatalf("reply %q stop %s", text, st.Stop)
	}
}

// Whisper through the catalog, runtime and the OpenAI audio endpoint.
func TestOpenAITranscription(t *testing.T) {
	os.Setenv("HF_HUB_OFFLINE", "1")
	entries, _ := Catalog()
	var spec *Spec
	for _, e := range entries {
		if e.Kind == "stt" && e.Spec.Repo == "openai/whisper-tiny" {
			s := e.Spec
			spec = &s
		}
	}
	if spec == nil {
		t.Skip("openai/whisper-tiny not in catalog")
	}
	rt := NewRuntime()
	srv := httptest.NewServer(OpenAIHandler(rt))
	defer srv.Close()
	wav, err := os.ReadFile("../testdata/whisper/front_center_16k.wav")
	if err != nil {
		t.Fatal(err)
	}
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	fw, _ := mw.CreateFormFile("file", "a.wav")
	fw.Write(wav)
	mw.WriteField("model", spec.ID)
	mw.WriteField("language", "en")
	mw.Close()
	resp, err := http.Post(srv.URL+"/v1/audio/transcriptions", mw.FormDataContentType(), &body)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var res TranscribeResult
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		t.Fatal(err)
	}
	if res.Text != "Front Center." {
		t.Fatalf("text %q (status %d)", res.Text, resp.StatusCode)
	}
	t.Logf("%q in %.0f ms mel + %.0f ms encode + %.0f ms decode (%.1fx real time)", res.Text, res.MelMS, res.EncodeMS, res.DecodeMS, res.RealTimeX)
	if rt.Status().STT == nil {
		t.Fatal("status does not report the STT model")
	}
}

// TestOpenAITranscriptionQwen3ASR drives the same endpoint with the
// Qwen3-ASR transcriber: auto-detected language comes back as a code and
// the forced-language path yields the same text.
func TestOpenAITranscriptionQwen3ASR(t *testing.T) {
	os.Setenv("HF_HUB_OFFLINE", "1")
	entries, _ := Catalog()
	var spec *Spec
	for _, e := range entries {
		if e.Kind == "stt" && e.ModelType == "qwen3_asr" {
			s := e.Spec
			spec = &s
			if e.Params != "1.7B" {
				t.Errorf("params %q", e.Params)
			}
		}
	}
	if spec == nil {
		t.Skip("no Qwen3-ASR checkpoint in catalog")
	}
	rt := NewRuntime()
	srv := httptest.NewServer(OpenAIHandler(rt))
	defer srv.Close()
	wav, err := os.ReadFile("../testdata/whisper/front_center_16k.wav")
	if err != nil {
		t.Fatal(err)
	}
	for _, lang := range []string{"", "en"} {
		var body bytes.Buffer
		mw := multipart.NewWriter(&body)
		fw, _ := mw.CreateFormFile("file", "a.wav")
		fw.Write(wav)
		mw.WriteField("model", spec.ID)
		mw.WriteField("language", lang)
		mw.Close()
		resp, err := http.Post(srv.URL+"/v1/audio/transcriptions", mw.FormDataContentType(), &body)
		if err != nil {
			t.Fatal(err)
		}
		var res TranscribeResult
		err = json.NewDecoder(resp.Body).Decode(&res)
		resp.Body.Close()
		if err != nil {
			t.Fatal(err)
		}
		if res.Text != "Front, center." || res.Language != "en" {
			t.Fatalf("lang %q: text %q language %q (status %d)", lang, res.Text, res.Language, resp.StatusCode)
		}
		t.Logf("lang %q: %q [%s] in %.0f ms mel + %.0f ms encode + %.0f ms decode (%.1fx real time)", lang, res.Text, res.Language, res.MelMS, res.EncodeMS, res.DecodeMS, res.RealTimeX)
	}
	rt.UnloadSTT()
}

// GGUF with neither a sidecar tokenizer.json nor a cached HF repo: the
// tokenizer comes from the file's tokenizer.ggml.* metadata (chatml
// template detected from <|im_start|>, EOS from the metadata).
func TestGenerateQwenGGUFEmbeddedTokenizer(t *testing.T) {
	spec := qwenGGUF(t)
	spec.Repo, spec.TokenizerPath = "", ""
	m, err := Load(spec)
	if err != nil {
		t.Fatal(err)
	}
	if m.Spec.Family != "chatml" || !m.eos[151645] {
		t.Fatalf("family %s eos %v", m.Spec.Family, m.eos)
	}
	text, st, err := m.Generate(t.Context(), []Message{{Role: "user", Content: "What is the capital of France? One word."}}, GenOpts{Temperature: 0, MaxTokens: 12}, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%q stop=%s %.1f tok/s", text, st.Stop, st.TokPerSec)
	if !strings.Contains(text, "Paris") || st.Stop != "eos" {
		t.Fatalf("reply %q stop %s", text, st.Stop)
	}
}

// Gemma 3 4B (multimodal checkpoint, text-only) through the engine at
// BF16; gated on LIGHT_HEAVY_TESTS (8.6 GB). Greedy must match transformers.
func TestGenerateGemma3_4BBF16(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1")
	}
	requireHostMemGB(t, 14)
	releaseModelMemory(t)
	os.Setenv("HF_HUB_OFFLINE", "1")
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--unsloth--gemma-3-4b-it/snapshots/*/model.safetensors.index.json")); len(m) == 0 {
		t.Skip("unsloth/gemma-3-4b-it not in HF cache")
	}
	m, err := Load(Spec{ID: "unsloth/gemma-3-4b-it", Repo: "unsloth/gemma-3-4b-it", DType: "bf16"})
	if err != nil {
		t.Fatal(err)
	}
	if m.Spec.Family != "gemma" {
		t.Fatalf("family %s", m.Spec.Family)
	}
	text, st, err := m.Generate(t.Context(), []Message{{Role: "user", Content: "What is 2+2?"}}, GenOpts{Temperature: 0, MaxTokens: 16}, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%q stop=%s %.1f tok/s", text, st.Stop, st.TokPerSec)
	if !strings.Contains(text, "2 + 2 = 4") || st.Stop != "eos" {
		t.Fatalf("reply %q stop %s", text, st.Stop)
	}
}

// Gemma 2 2B through the engine at BF16 (softcaps, no QK-norm): gemma
// template, EOS 1/107. Greedy must reproduce transformers' "2 + 2 = 4".
func TestGenerateGemma2BF16(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (2.6B params)")
	}
	requireHostMemGB(t, 12)
	releaseModelMemory(t)
	os.Setenv("HF_HUB_OFFLINE", "1")
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--unsloth--gemma-2-2b-it/snapshots/*/model*.safetensors")); len(m) == 0 {
		t.Skip("unsloth/gemma-2-2b-it not in HF cache")
	}
	m, err := Load(Spec{ID: "unsloth/gemma-2-2b-it", Repo: "unsloth/gemma-2-2b-it", DType: "bf16"})
	if err != nil {
		t.Fatal(err)
	}
	if m.Spec.Family != "gemma" {
		t.Fatalf("family %s", m.Spec.Family)
	}
	text, st, err := m.Generate(t.Context(), []Message{{Role: "user", Content: "What is 2+2?"}}, GenOpts{Temperature: 0, MaxTokens: 16}, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%q stop=%s %.1f tok/s", text, st.Stop, st.TokPerSec)
	if !strings.Contains(text, "2 + 2 = 4") || st.Stop != "eos" {
		t.Fatalf("reply %q stop %s", text, st.Stop)
	}
}

// Gemma 4 E2B through the engine at BF16: gemma4 template (<|turn>…<turn|>),
// EOS 1/106/50 from generation_config. Gated (10 GB checkpoint).
func TestGenerateGemma4E2BBF16(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1")
	}
	requireHostMemGB(t, 10)
	releaseModelMemory(t)
	os.Setenv("HF_HUB_OFFLINE", "1")
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--unsloth--gemma-4-E2B-it/snapshots/*/model*.safetensors")); len(m) == 0 {
		t.Skip("unsloth/gemma-4-E2B-it not in HF cache")
	}
	m, err := Load(Spec{ID: "unsloth/gemma-4-E2B-it", Repo: "unsloth/gemma-4-E2B-it", DType: "bf16"})
	if err != nil {
		t.Fatal(err)
	}
	if m.Spec.Family != "gemma4" || !m.eos[106] {
		t.Fatalf("family %s eos %v", m.Spec.Family, m.eos)
	}
	text, st, err := m.Generate(t.Context(), []Message{{Role: "user", Content: "What is 2+2?"}}, GenOpts{Temperature: 0, MaxTokens: 16}, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%q stop=%s %.1f tok/s", text, st.Stop, st.TokPerSec)
	if !strings.Contains(text, "2 + 2 = **4**") || st.Stop != "eos" {
		t.Fatalf("reply %q stop %s", text, st.Stop)
	}
}

// EmbeddingGemma 2 through the OpenAI embeddings endpoint: query/document
// cosine must beat the off-topic pair and match the transformers reference
// (models/embeddinggemma2_test.go) to BF16 tolerance.
func TestOpenAIEmbeddings(t *testing.T) {
	os.Setenv("HF_HUB_OFFLINE", "1")
	entries, _ := Catalog()
	var spec *Spec
	for _, e := range entries {
		if e.Kind == "embedding" && e.Spec.Repo == "unsloth/embeddinggemma-2" {
			s := e.Spec
			spec = &s
		}
	}
	if spec == nil {
		t.Skip("unsloth/embeddinggemma-2 not in catalog")
	}
	rt := NewRuntime()
	srv := httptest.NewServer(OpenAIHandler(rt))
	defer srv.Close()
	body := `{"model":"unsloth/embeddinggemma-2","input":[` +
		`"task: search result | query: What causes the northern lights?",` +
		`"title: none | text: The northern lights are caused by charged particles from the sun colliding with gases in the upper atmosphere.",` +
		`"title: none | text: A sourdough starter needs regular feeding with flour and water to stay active."]}`
	resp, err := http.Post(srv.URL+"/v1/embeddings", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil || resp.StatusCode != 200 {
		t.Fatalf("status %d err %v", resp.StatusCode, err)
	}
	if len(out.Data) != 3 || len(out.Data[0].Embedding) != 768 || out.Usage.PromptTokens != 15+27+22 {
		t.Fatalf("got %d vectors, dim %d, tokens %d", len(out.Data), len(out.Data[0].Embedding), out.Usage.PromptTokens)
	}
	cos := func(a, b []float32) float64 {
		var s float64
		for i := range a {
			s += float64(a[i] * b[i])
		}
		return s
	}
	qd, qo := cos(out.Data[0].Embedding, out.Data[1].Embedding), cos(out.Data[0].Embedding, out.Data[2].Embedding)
	t.Logf("cos(query, doc) = %.4f, cos(query, other) = %.4f", qd, qo)
	if math.Abs(qd-0.8545) > 1.5e-2 || math.Abs(qo-0.5935) > 1.5e-2 {
		t.Fatalf("cosines off the reference 0.8545 / 0.5935")
	}
	if rt.Status().Embedder == nil {
		t.Fatal("status has no embedder")
	}
	rt.UnloadEmbedder()
}

const phi4PromptTokens = 21 // ref_phi4.py prompt ids

// Phi-4-mini through the engine at BF16: phi4 template (no newlines), EOS
// 200020/199999 from generation_config, GPT-4o tokenizer (Split regex with
// behavior Removed+invert). Gated (7.7 GB checkpoint).
func TestGeneratePhi4MiniBF16(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1")
	}
	requireHostMemGB(t, 12)
	releaseModelMemory(t)
	os.Setenv("HF_HUB_OFFLINE", "1")
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--microsoft--Phi-4-mini-instruct/snapshots/*/model-*.safetensors")); len(m) < 2 {
		t.Skip("microsoft/Phi-4-mini-instruct not in HF cache")
	}
	m, err := Load(Spec{ID: "microsoft/Phi-4-mini-instruct", Repo: "microsoft/Phi-4-mini-instruct", DType: "bf16"})
	if err != nil {
		t.Fatal(err)
	}
	if m.Spec.Family != "phi4" || !m.eos[200020] || !m.eos[199999] {
		t.Fatalf("family %s eos %v", m.Spec.Family, m.eos)
	}
	msgs := []Message{{Role: "system", Content: "You are a concise assistant."}, {Role: "user", Content: "Explain in one sentence why the sky is blue."}}
	if ids := m.Tok.Encode(m.Prompt(msgs)); len(ids) != phi4PromptTokens {
		t.Fatalf("prompt tokenised to %d ids: %v", len(ids), ids)
	}
	text, st, err := m.Generate(t.Context(), msgs, GenOpts{Temperature: 0, MaxTokens: 24}, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%q stop=%s %.1f tok/s", text, st.Stop, st.TokPerSec)
	if !strings.Contains(text, "Rayleigh scattering") {
		t.Fatalf("reply %q", text)
	}
}

// falcon-rw-1b (base model, ALiBi) through the engine at BF16 from the
// catalog: plain template, pytorch_model.bin weights, raw completion
// " Rome" after the capitals prompt.
func TestFalconRW1BEngineBF16(t *testing.T) {
	os.Setenv("HF_HUB_OFFLINE", "1")
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--tiiuae--falcon-rw-1b/snapshots/*/pytorch_model.bin")); len(m) == 0 {
		t.Skip("tiiuae/falcon-rw-1b not in HF cache")
	}
	requireHostMemGB(t, 8)
	releaseModelMemory(t)
	entries, _ := Catalog()
	var spec *Spec
	for _, e := range entries {
		if e.Spec.Repo == "tiiuae/falcon-rw-1b" {
			s := e.Spec
			spec = &s
		}
	}
	if spec == nil {
		t.Fatal("tiiuae/falcon-rw-1b not in catalog")
	}
	m, err := Load(*spec)
	if err != nil {
		t.Fatal(err)
	}
	if m.Spec.Family != "plain" || m.Spec.DType != "bf16" {
		t.Fatalf("family %s dtype %s", m.Spec.Family, m.Spec.DType)
	}
	ids := toI64(m.Tok.Encode("The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"))
	if len(ids) != 19 {
		t.Fatalf("prompt ids %v", ids)
	}
	row, err := m.step(ids, models.NewKVCache(m.LM.NumLayers()))
	if err != nil {
		t.Fatal(err)
	}
	best := 0
	for i, v := range row {
		if v > row[best] {
			best = i
		}
	}
	if best != 10598 { // " Rome"
		t.Fatalf("top-1 %d (%q), want 10598", best, m.Tok.Decode([]int32{int32(best)}))
	}
	text, st, err := m.Generate(t.Context(), []Message{{Role: "user", Content: "Name three colours."}}, GenOpts{Temperature: 0, MaxTokens: 12}, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%q stop=%s %.1f tok/s", text, st.Stop, st.TokPerSec)
}

// TestOpenAISpeechToSpeech drives /v1/audio/conversations with the Front
// Center clip through Moshi + Mimi (8.2 GB Q8 on CUDA; gated). Greedy
// decoding gives the "Hello, what's up?" opener seen in models.
func TestOpenAISpeechToSpeech(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (8.2 GB Q8 checkpoint)")
	}
	os.Setenv("HF_HUB_OFFLINE", "1")
	entries, _ := Catalog()
	var spec *Spec
	for _, e := range entries {
		if e.Kind == "speech" && e.Spec.Repo == models.MoshikoRepo {
			s := e.Spec
			spec = &s
		}
	}
	if spec == nil {
		t.Skip(models.MoshikoRepo + " not in catalog")
	}
	requireHostMemGB(t, 6)
	rt := NewRuntime()
	srv := httptest.NewServer(OpenAIHandler(rt))
	defer srv.Close()
	wav, err := os.ReadFile("../testdata/mimi/front_center_24k.wav")
	if err != nil {
		t.Fatal(err)
	}
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	fw, _ := mw.CreateFormFile("file", "a.wav")
	fw.Write(wav)
	mw.WriteField("model", spec.ID)
	mw.WriteField("greedy", "1")
	mw.WriteField("extra_seconds", "2")
	mw.Close()
	resp, err := http.Post(srv.URL+"/v1/audio/conversations", mw.FormDataContentType(), &body)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var res struct {
		ConverseResult
		AudioWAV string `json:"audio_wav_base64"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil || resp.StatusCode != 200 {
		t.Fatalf("status %d err %v", resp.StatusCode, err)
	}
	t.Logf("%q: %.1f s in → %.1f s reply, %d frames, %.0f ms/frame (%.1fx real time), wav %d bytes", res.Text, res.InputSec, res.AudioSec, res.Frames, res.StepMS, res.RealTimeX, len(res.AudioWAV))
	if !strings.HasPrefix(res.Text, "Hello") || res.Frames != 18+25 || res.AudioSec < 3 || len(res.AudioWAV) == 0 {
		t.Fatalf("unexpected reply: %+v", res.ConverseResult)
	}
	st := rt.Status()
	if st.Speech == nil || st.Loaded {
		t.Fatalf("status %+v should report Moshi and no chat model", st)
	}
	if s := rt.Speech(); s == nil || s.Spec.Family != "moshi" {
		t.Fatal("speech slot not filled")
	}
}
