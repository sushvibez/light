package tokenizer_test

import (
	"strings"
	"testing"

	"light/tokenizer"
)

// buildByteVocab constructs a tiny GPT-2-like vocab where every printable
// ASCII byte (via GPT-2 byte-encoder) is its own token plus a few known
// merges: "h","e","l","o", then "he", "hel", "hell", "hello".
func buildByteVocab(t *testing.T) *tokenizer.BPE {
	t.Helper()
	// Vocab ids start from 0 for each single-char token.
	vocab := map[string]int32{
		"h": 0, "e": 1, "l": 2, "o": 3, " ": 4,
		"w": 5, "r": 6, "d": 7,
		"he": 8, "hel": 9, "hell": 10, "hello": 11, " w": 12, " wo": 13, " wor": 14, " worl": 15, " world": 16,
	}
	// Byte-encoder for ASCII space is codepoint 256 ("Ġ" in GPT-2). We rely on
	// our BPE.Encode to map " " → "Ġ" before lookup. So vocab entries for
	// whitespace-prefixed tokens use the Ġ-char. Remap here:
	vocabEncoded := make(map[string]int32, len(vocab))
	for k, v := range vocab {
		vocabEncoded[asciiToGPT2(k)] = v
	}
	merges := []string{
		"h e",
		"he l",
		"hel l",
		"hell o",
		"Ġ w",
		"Ġw o",
		"Ġwo r",
		"Ġwor l",
		"Ġworl d",
	}
	bpe, err := tokenizer.NewBPE(vocabEncoded, merges)
	if err != nil {
		t.Fatal(err)
	}
	return bpe
}

// asciiToGPT2 converts every ASCII byte in s to its GPT-2 byte-encoder char.
// Only space needs remapping for our test vocab.
func asciiToGPT2(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == ' ' {
			b.WriteRune('\u0120') // Ġ
		} else {
			b.WriteByte(c)
		}
	}
	return b.String()
}

func TestBPEEncodeDecodeRoundTrip(t *testing.T) {
	bpe := buildByteVocab(t)
	ids := bpe.Encode("hello world")
	if len(ids) == 0 {
		t.Fatal("encode produced no tokens")
	}
	// We expect two tokens: "hello" and " world" (the full merges should collapse each).
	// It's OK if the encoder chose a slightly different chunking due to
	// missing merges; we just verify round-trip.
	decoded := bpe.Decode(ids)
	if decoded != "hello world" {
		t.Errorf("round-trip got %q, want %q (ids=%v)", decoded, "hello world", ids)
	}
}

func TestBPEEncodeSingleWord(t *testing.T) {
	bpe := buildByteVocab(t)
	ids := bpe.Encode("hello")
	if len(ids) != 1 || bpe.IDToToken(ids[0]) != "hello" {
		t.Errorf("expected single 'hello' token; got ids=%v", ids)
	}
}

func TestBPEMerges(t *testing.T) {
	bpe := buildByteVocab(t)
	// "hel" should merge to "hel" via the chain h+e=he, he+l=hel.
	ids := bpe.Encode("hel")
	if len(ids) != 1 || bpe.IDToToken(ids[0]) != "hel" {
		t.Errorf("expected 'hel' merge; got %v", ids)
	}
}

func TestVocabSize(t *testing.T) {
	bpe := buildByteVocab(t)
	if bpe.VocabSize() != 17 {
		t.Errorf("vocab size = %d, want 17", bpe.VocabSize())
	}
}

func TestTokenIDLookup(t *testing.T) {
	bpe := buildByteVocab(t)
	if bpe.TokenToID("h") != 0 {
		t.Errorf("TokenToID: %d", bpe.TokenToID("h"))
	}
	if bpe.TokenToID("__nonexistent__") != -1 {
		t.Error("expected -1 for missing token")
	}
}

func TestBPEEmptyInput(t *testing.T) {
	bpe := buildByteVocab(t)
	if len(bpe.Encode("")) != 0 {
		t.Error("empty input should produce empty ids")
	}
	if bpe.Decode(nil) != "" {
		t.Error("empty ids should decode to empty")
	}
}

func TestBPEInvalidMerge(t *testing.T) {
	vocab := map[string]int32{"a": 0, "b": 1}
	if _, err := tokenizer.NewBPE(vocab, []string{"invalid-no-space"}); err == nil {
		t.Error("expected error on malformed merge line")
	}
}

func TestBPEEmptyVocab(t *testing.T) {
	if _, err := tokenizer.NewBPE(map[string]int32{}, nil); err == nil {
		t.Error("expected error on empty vocab")
	}
}

func TestDecodeDropsUnknownIDs(t *testing.T) {
	bpe := buildByteVocab(t)
	ids := bpe.Encode("hello")
	ids = append(ids, 99999) // unknown ID
	got := bpe.Decode(ids)
	if got != "hello" {
		t.Errorf("got %q, want 'hello'", got)
	}
}

func TestLoadBPEFromJSON(t *testing.T) {
	jsonBlob := `{
		"version": "1.0",
		"model": {
			"type": "BPE",
			"vocab": {"h": 0, "i": 1, "hi": 2},
			"merges": ["h i"]
		},
		"added_tokens": [{"id": 10, "content": "<s>", "special": true}]
	}`
	bpe, err := tokenizer.LoadBPE(strings.NewReader(jsonBlob))
	if err != nil {
		t.Fatal(err)
	}
	if bpe.TokenToID("hi") != 2 {
		t.Errorf("'hi' → %d", bpe.TokenToID("hi"))
	}
	if bpe.TokenToID("<s>") != 10 {
		t.Errorf("'<s>' → %d", bpe.TokenToID("<s>"))
	}
	ids := bpe.Encode("hi")
	if len(ids) != 1 || ids[0] != 2 {
		t.Errorf("encoded 'hi' = %v, want [2]", ids)
	}
}

func TestLoadBPEInvalidJSON(t *testing.T) {
	if _, err := tokenizer.LoadBPE(strings.NewReader("garbage")); err == nil {
		t.Error("expected error on bad JSON")
	}
	if _, err := tokenizer.LoadBPE(strings.NewReader(`{"model": {"type": "WordPiece"}}`)); err == nil {
		t.Error("expected error on unsupported model type")
	}
	if _, err := tokenizer.LoadBPE(strings.NewReader(`{"model": {"type": "BPE", "vocab": {}}}`)); err == nil {
		t.Error("expected error on empty vocab")
	}
}
