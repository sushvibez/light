package engine

import (
	"encoding/base64"
	"encoding/json"
	"image"
	"os"
	"strings"
	"testing"
)

// An image + question through the engine (chat template with an image
// part, placeholder expansion, vision tower, MRoPE prefill and decode) on
// Qwen3.5-0.8B: the expanded prompt ids equal transformers' processor
// output and the greedy reply equals transformers' 16 tokens
// (models/testdata/qwen35-vision/vl.json).
func TestQwen35ImageChat(t *testing.T) {
	const dir = "../models/testdata/qwen35-vision/"
	raw, err := os.ReadFile(dir + "vl.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		IDs      []int64 `json:"ids"`
		Greedy   []int64 `json:"greedy"`
		GreedyTx string  `json:"greedy_text"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	m, err := Load(Spec{ID: "Qwen/Qwen3.5-0.8B", Repo: "Qwen/Qwen3.5-0.8B", DType: "f32", Device: "cpu"})
	if err != nil {
		t.Skip(err)
	}
	img, err := LoadImage(dir + "image.png")
	if err != nil {
		t.Fatal(err)
	}
	msgs := []Message{{Role: "user", Content: "Describe this image in one sentence.", Images: []image.Image{img}}}
	text, st, err := m.Generate(t.Context(), msgs, GenOpts{Temperature: 0, MaxTokens: len(ref.Greedy)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%d prompt tokens; reply %q", st.PromptTokens, text)
	if st.PromptTokens != len(ref.IDs) {
		t.Errorf("prompt has %d tokens, transformers %d", st.PromptTokens, len(ref.IDs))
	}
	if strings.TrimSpace(text) != strings.TrimSpace(ref.GreedyTx) {
		t.Errorf("reply %q\n want %q", text, ref.GreedyTx)
	}
}

// OpenAI content parts decode into text and images.
func TestMessageContentParts(t *testing.T) {
	png, err := os.ReadFile("../models/testdata/qwen35-vision/image.png")
	if err != nil {
		t.Fatal(err)
	}
	body := `{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,` +
		base64.StdEncoding.EncodeToString(png) + `"}},{"type":"text","text":"What is this?"}]}`
	var m Message
	if err := json.Unmarshal([]byte(body), &m); err != nil {
		t.Fatal(err)
	}
	if m.Role != "user" || m.Content != "What is this?" || len(m.Images) != 1 || m.Images[0].Bounds().Dx() != 320 {
		t.Fatalf("got %+v", m)
	}
	if err := json.Unmarshal([]byte(`{"role":"user","content":"hi"}`), &m); err != nil || m.Content != "hi" || m.Images != nil {
		t.Fatalf("string content: %+v %v", m, err)
	}
	if err := json.Unmarshal([]byte(`{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}`), &m); err == nil {
		t.Fatal("remote image URL accepted")
	}
}
