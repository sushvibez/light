package tokenizer_test

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"light/tokenizer"
)

// wordPieceJSON is a tiny synthetic BERT WordPiece tokenizer.json.
const wordPieceJSON = `{
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
      "play": 4, "##ing": 5, "##ed": 6, "love": 7,
      "the": 8, "cat": 9, ".": 10, "un": 11, "##aff": 12, "##able": 13,
      "cafe": 14, "中": 15, "a": 16
    }
  },
  "normalizer": { "type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": null, "lowercase": true },
  "pre_tokenizer": { "type": "BertPreTokenizer" },
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
    ],
    "pair": [
      {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
      {"Sequence": {"id": "B", "type_id": 1}},
      {"SpecialToken": {"id": "[SEP]", "type_id": 1}}
    ],
    "special_tokens": {
      "[CLS]": {"id": "[CLS]", "ids": [2], "tokens": ["[CLS]"]},
      "[SEP]": {"id": "[SEP]", "ids": [3], "tokens": ["[SEP]"]}
    }
  },
  "added_tokens": [
    {"id": 1, "content": "[UNK]", "special": true},
    {"id": 2, "content": "[CLS]", "special": true},
    {"id": 3, "content": "[SEP]", "special": true}
  ]
}`

func loadWordPiece(t *testing.T) *tokenizer.WordPiece {
	t.Helper()
	tok, err := tokenizer.Parse([]byte(wordPieceJSON))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	wp, ok := tok.(*tokenizer.WordPiece)
	if !ok {
		t.Fatalf("Parse returned %T, want *WordPiece", tok)
	}
	return wp
}

func pieces(tok tokenizer.Tokenizer, ids []int32) []string {
	out := make([]string, len(ids))
	for i, id := range ids {
		out[i] = tok.IDToToken(id)
	}
	return out
}

func TestWordPieceParse(t *testing.T) {
	wp := loadWordPiece(t)
	if wp.UnkID() != 1 {
		t.Errorf("UnkID=%d want 1", wp.UnkID())
	}
	if wp.VocabSize() != 17 {
		t.Errorf("VocabSize=%d want 17", wp.VocabSize())
	}
	if wp.TokenToID("##ing") != 5 || wp.TokenToID("nope") != -1 {
		t.Error("TokenToID lookup wrong")
	}
	if wp.IDToToken(9) != "cat" || wp.IDToToken(999) != "" {
		t.Error("IDToToken lookup wrong")
	}
}

func TestWordPieceGreedySubword(t *testing.T) {
	wp := loadWordPiece(t)
	got := pieces(wp, wp.Encode("playing played"))
	want := []string{"play", "##ing", "play", "##ed"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v want %v", got, want)
	}
}

func TestWordPiecePunctuationSplit(t *testing.T) {
	wp := loadWordPiece(t)
	got := pieces(wp, wp.Encode("the cat."))
	want := []string{"the", "cat", "."}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v want %v", got, want)
	}
}

func TestWordPieceNormalizer(t *testing.T) {
	wp := loadWordPiece(t)
	cases := []struct {
		in   string
		want []string
	}{
		{"THE CAT", []string{"the", "cat"}},
		{"Café", []string{"cafe"}},                             // strip_accents follows lowercase
		{"the\u0000\tcat​", []string{"the", "cat"}},            // clean_text drops NUL / format chars
		{"中a", []string{"中", "a"}},                             // handle_chinese_chars pads CJK
		{"a cat", []string{"a", "cat"}},                        // NBSP is whitespace
		{"[CLS] the [SEP]", []string{"[CLS]", "the", "[SEP]"}}, // specials survive lowercasing
	}
	for _, c := range cases {
		got := pieces(wp, wp.Encode(c.in))
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("Encode(%q) = %v want %v", c.in, got, c.want)
		}
	}
}

func TestWordPieceNoStripAccentsWhenDisabled(t *testing.T) {
	js := strings.Replace(wordPieceJSON, `"strip_accents": null`, `"strip_accents": false`, 1)
	tok, err := tokenizer.Parse([]byte(js))
	if err != nil {
		t.Fatal(err)
	}
	ids := tok.Encode("Café")
	if len(ids) != 1 || ids[0] != 1 {
		t.Errorf("with strip_accents=false 'Café' should be [UNK]; got %v", pieces(tok, ids))
	}
}

func TestWordPieceMultiSubword(t *testing.T) {
	wp := loadWordPiece(t)
	got := pieces(wp, wp.Encode("unaffable"))
	want := []string{"un", "##aff", "##able"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v want %v", got, want)
	}
}

func TestWordPieceUnknownWord(t *testing.T) {
	wp := loadWordPiece(t)
	ids := wp.Encode("xyzzy")
	if len(ids) != 1 || ids[0] != wp.UnkID() {
		t.Errorf("got %v want single unk(%d)", ids, wp.UnkID())
	}
	// A word that starts fine but has an unmatched tail is one unk, not partial.
	ids = wp.Encode("playzz")
	if len(ids) != 1 || ids[0] != wp.UnkID() {
		t.Errorf("got %v want single unk", pieces(wp, ids))
	}
}

func TestWordPieceMaxInputChars(t *testing.T) {
	js := strings.Replace(wordPieceJSON, `"max_input_chars_per_word": 100`, `"max_input_chars_per_word": 5`, 1)
	tok, err := tokenizer.Parse([]byte(js))
	if err != nil {
		t.Fatal(err)
	}
	ids := tok.Encode("playing")
	if len(ids) != 1 || ids[0] != 1 {
		t.Errorf("7-char word over the 5-char limit should be [UNK]; got %v", pieces(tok, ids))
	}
}

func TestWordPieceDecode(t *testing.T) {
	wp := loadWordPiece(t)
	if got := wp.Decode(wp.Encode("playing played")); got != "playing played" {
		t.Errorf("decode = %q", got)
	}
	if got := wp.Decode(wp.Encode("the cat.")); got != "the cat." {
		t.Errorf("decode = %q", got)
	}
	if got := wp.Decode(wp.EncodeWithSpecial("the cat")); got != "the cat" {
		t.Errorf("decode should skip specials; got %q", got)
	}
	if got := wp.Decode([]int32{8, 4242, 9}); got != "the cat" {
		t.Errorf("decode should skip unknown ids; got %q", got)
	}
}

func TestWordPiecePostProcessor(t *testing.T) {
	wp := loadWordPiece(t)
	got := wp.EncodeWithSpecial("the cat")
	want := []int32{2, 8, 9, 3}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("EncodeWithSpecial = %v want %v", got, want)
	}
	got = wp.EncodePair("the cat", "playing")
	want = []int32{2, 8, 9, 3, 4, 5, 3}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("EncodePair = %v want %v", got, want)
	}
	if got := wp.EncodeWithSpecial(""); !reflect.DeepEqual(got, []int32{2, 3}) {
		t.Errorf("empty EncodeWithSpecial = %v want [2 3]", got)
	}
	// No post-processor → EncodeWithSpecial degrades to Encode.
	wp.SetPostProcessor(nil)
	if got := wp.EncodeWithSpecial("the cat"); !reflect.DeepEqual(got, []int32{8, 9}) {
		t.Errorf("without post-processor got %v", got)
	}
}

func TestWordPieceBertProcessingBlock(t *testing.T) {
	js := strings.Replace(wordPieceJSON, `"type": "TemplateProcessing",`, `"type": "BertProcessing", "sep": ["[SEP]", 3], "cls": ["[CLS]", 2],`, 1)
	tok, err := tokenizer.Parse([]byte(js))
	if err != nil {
		t.Fatal(err)
	}
	got := tok.(tokenizer.TemplateEncoder).EncodeWithSpecial("the cat")
	if !reflect.DeepEqual(got, []int32{2, 8, 9, 3}) {
		t.Errorf("BertProcessing EncodeWithSpecial = %v", got)
	}
}

func TestNewWordPieceErrors(t *testing.T) {
	if _, err := tokenizer.NewWordPiece(map[string]int32{}, tokenizer.WordPieceConfig{}); err == nil {
		t.Error("expected error for empty vocab")
	}
	if _, err := tokenizer.NewWordPiece(map[string]int32{"a": 0}, tokenizer.WordPieceConfig{}); err == nil {
		t.Error("expected error when unk token is missing")
	}
}

// ---- real-file test against the cached bge-small-en-v1.5 tokenizer ----

func bgeTokenizerPath(t *testing.T) string {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home dir")
	}
	matches, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--unsloth--bge-small-en-v1.5/snapshots/*/tokenizer.json"))
	if len(matches) == 0 {
		t.Skip("bge-small-en-v1.5 tokenizer.json not cached")
	}
	return matches[0]
}

func TestWordPieceBGESmall(t *testing.T) {
	path := bgeTokenizerPath(t)
	tok, err := tokenizer.Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	wp, ok := tok.(*tokenizer.WordPiece)
	if !ok {
		t.Fatalf("Load returned %T, want *WordPiece", tok)
	}
	if wp.VocabSize() != 30522 {
		t.Errorf("VocabSize=%d want 30522", wp.VocabSize())
	}
	if wp.UnkID() != 100 {
		t.Errorf("UnkID=%d want 100", wp.UnkID())
	}

	// Expected ids/tokens were produced by HF `tokenizers` 0.22 on the same
	// file (encode with add_special_tokens=True and =False).
	cases := []struct {
		in     string
		tokens []string
		ids    []int32 // with specials
	}{
		{"Hello world!", []string{"hello", "world", "!"}, []int32{101, 7592, 2088, 999, 102}},
		{"The quick brown fox jumps over the lazy dog.",
			[]string{"the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."},
			[]int32{101, 1996, 4248, 2829, 4419, 14523, 2058, 1996, 13971, 3899, 1012, 102}},
		{"Tokenization is fun: unaffable embeddings, 3.14!",
			[]string{"token", "##ization", "is", "fun", ":", "una", "##ffa", "##ble", "em", "##bed", "##ding", "##s", ",", "3", ".", "14", "!"},
			[]int32{101, 19204, 3989, 2003, 4569, 1024, 14477, 20961, 3468, 7861, 8270, 4667, 2015, 1010, 1017, 1012, 2403, 999, 102}},
		{"Café résumé naïve Straße",
			[]string{"cafe", "resume", "naive", "st", "##raße"},
			[]int32{101, 7668, 13746, 15743, 2358, 27807, 102}},
		{"中国人 hello", []string{"中", "国", "人", "hello"}, []int32{101, 1746, 1799, 1756, 7592, 102}},
		{"New York-based\tcompany's\n  value",
			[]string{"new", "york", "-", "based", "company", "'", "s", "value"},
			[]int32{101, 2047, 2259, 1011, 2241, 2194, 1005, 1055, 3643, 102}},
		{"[CLS] already [SEP] here", []string{"[CLS]", "already", "[SEP]", "here"}, []int32{101, 101, 2525, 102, 2182, 102}},
		{"¿Qué? ÀÉÎÕÜ ñ", []string{"¿", "que", "?", "ae", "##io", "##u", "n"}, []int32{101, 1094, 10861, 1029, 29347, 3695, 2226, 1050, 102}},
		{"", nil, []int32{101, 102}},
		{"   ", nil, []int32{101, 102}},
	}
	for _, c := range cases {
		ids := wp.Encode(c.in)
		if got := pieces(wp, ids); !reflect.DeepEqual(got, c.tokens) && !(len(got) == 0 && len(c.tokens) == 0) {
			t.Errorf("Encode(%q) tokens = %v want %v", c.in, got, c.tokens)
		}
		if got := wp.EncodeWithSpecial(c.in); !reflect.DeepEqual(got, c.ids) {
			t.Errorf("EncodeWithSpecial(%q) = %v want %v", c.in, got, c.ids)
		}
	}

	pair := wp.EncodePair("Hello world!", "second one")
	wantPair := []int32{101, 7592, 2088, 999, 102, 2117, 2028, 102}
	if !reflect.DeepEqual(pair, wantPair) {
		t.Errorf("EncodePair = %v want %v", pair, wantPair)
	}

	// Decode matches HF's WordPiece decoder with cleanup=true.
	decodes := []struct{ in, want string }{
		{"Hello world!", "hello world!"},
		{"Tokenization is fun: unaffable embeddings, 3.14!", "tokenization is fun : unaffable embeddings, 3. 14!"},
		{"New York-based\tcompany's\n  value", "new york - based company ' s value"},
		{"中国人 hello", "中 国 人 hello"},
	}
	for _, d := range decodes {
		if got := wp.Decode(wp.EncodeWithSpecial(d.in)); got != d.want {
			t.Errorf("Decode(%q) = %q want %q", d.in, got, d.want)
		}
	}
}
