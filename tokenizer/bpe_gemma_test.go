package tokenizer

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// Gemma 3 tokenizer.json: Replace " "→"▁" normalizer + a no-op Split
// pre_tokenizer, byte fallback, no Strip in the decoder (leading spaces
// survive). Expected ids from HF `tokenizers` 0.22 (add_special_tokens=False).
func TestGemma3SentencePiece(t *testing.T) {
	home, _ := os.UserHomeDir()
	matches, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--unsloth--gemma-3-1b-it/snapshots/*/tokenizer.json"))
	if len(matches) == 0 {
		t.Skip("unsloth/gemma-3-1b-it not in HF cache")
	}
	f, err := os.Open(matches[0])
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	bpe, err := LoadBPE(f)
	if err != nil {
		t.Fatal(err)
	}
	if !bpe.IsSentencePiece() {
		t.Fatal("Gemma tokenizer not detected as SentencePiece")
	}
	cases := []struct {
		text string
		ids  []int32
	}{
		{"Hello world", []int32{9259, 1902}},
		{" leading space", []int32{5830, 2557}},
		{"What is 2+2?", []int32{3689, 563, 236743, 236778, 236862, 236778, 236881}},
		{"<bos><start_of_turn>user\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\n", []int32{2, 105, 2364, 107, 3689, 563, 236743, 236778, 236862, 236778, 236881, 106, 107, 105, 4368, 107}},
		{"héllo wörld 日本語 🙂", []int32{236754, 7869, 844, 515, 4010, 671, 33375, 238582, 57235}},
		{"  double  spaces", []int32{138, 7902, 138, 35220}},
		{"line\n newline\ttab", []int32{1257, 107, 100837, 255968, 4823}},
		{"The capital of France is", []int32{818, 5279, 529, 7001, 563}},
	}
	for _, c := range cases {
		got := bpe.Encode(c.text)
		if !slices.Equal(got, c.ids) {
			t.Errorf("%q: got %v want %v", c.text, got, c.ids)
		}
		if back := bpe.Decode(c.ids); back != c.text {
			t.Errorf("decode %q: got %q", c.text, back)
		}
	}
}
