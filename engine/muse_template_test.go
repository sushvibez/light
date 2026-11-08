package engine

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"light/chattemplate"
	"light/tokenizer"
)

// Muse Glimmer's own chat template and tokenizer through the engine's
// loaders, without weights: the validation-bar conversation (system +
// user, generation prompt) must render and tokenize to the ids transformers'
// apply_chat_template gave in the 30B reference run
// (`ref_muse_glimmer.py meta-models/Muse-Glimmer-30B bf16 --chat`).
func TestMuseGlimmerTemplateMatchesTransformers(t *testing.T) {
	const repo = "meta-models/Muse-Glimmer-30B"
	home, _ := os.UserHomeDir()
	files, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B/snapshots/*/tokenizer.json"))
	if len(files) == 0 {
		t.Skip(repo + " tokenizer not in HF cache")
	}
	os.Setenv("HF_HUB_OFFLINE", "1")
	tok, err := tokenizer.Load(files[0])
	if err != nil {
		t.Fatal(err)
	}
	ct, err := hfChatTemplate(repo)
	if err != nil {
		t.Fatal(err)
	}
	msgs := []chattemplate.Message{
		{Role: "system", Content: "You are a concise assistant."},
		{Role: "user", Content: "Explain in one sentence why the sky is blue."},
	}
	text, err := ct.Render(msgs, chattemplate.Options{AddGenerationPrompt: true})
	if err != nil {
		t.Fatal(err)
	}
	want := []int32{200000, 200022, 15651, 200023, 4662, 583, 262, 81080, 43910, 1574, 34956, 300, 9762, 38, 2244, 1574,
		15, 14757, 73965, 38, 392, 2540, 706, 392, 1556, 4205, 200008, 200022, 1556, 200023, 51015, 310, 1085, 17705, 5959,
		290, 21752, 373, 10950, 26, 200008, 200022, 140680}
	got := tok.Encode(text)
	if !slices.Equal(got, want) {
		t.Fatalf("prompt ids %v\nwant       %v\nrendered %q", got, want, text)
	}
}
