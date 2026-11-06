package tokenizer_test

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"light/tokenizer"
)

// unigramJSON builds a synthetic SentencePiece Unigram tokenizer.json. Scores
// are log-probs; the expected outputs below were produced by HF `tokenizers`
// 0.22 on the identical JSON. Byte pieces <0x00>..<0xFF> occupy ids 20..275.
func unigramJSON(byteFallback bool, scheme string, split bool) string {
	var b strings.Builder
	b.WriteString(`{"version":"1.0","model":{"type":"Unigram","unk_id":0,"byte_fallback":`)
	fmt.Fprintf(&b, "%v", byteFallback)
	b.WriteString(`,"vocab":[["<unk>",0.0],["<s>",0.0],["</s>",0.0],["▁",-3.0],["▁hello",-1.0],["▁wor",-2.0],["ld",-2.0],["▁he",-4.0],["llo",-4.0],["a",-5.0],["b",-5.0],["▁ab",-6.5],["▁world",-4.5],["l",-6.0],["d",-6.0],["h",-7.0],["e",-7.0],["o",-7.0],["r",-7.0],["w",-7.0]`)
	for i := 0; i < 256; i++ {
		fmt.Fprintf(&b, `,["<0x%02X>",-20.0]`, i)
	}
	b.WriteString(`]},"added_tokens":[{"id":1,"content":"<s>","special":true},{"id":2,"content":"</s>","special":true}],"normalizer":null,`)
	fmt.Fprintf(&b, `"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":%q,"split":%v},"post_processor":null}`, scheme, split)
	return b.String()
}

func loadUnigram(t *testing.T, js string) *tokenizer.Unigram {
	t.Helper()
	tok, err := tokenizer.Parse([]byte(js))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	u, ok := tok.(*tokenizer.Unigram)
	if !ok {
		t.Fatalf("Parse returned %T, want *Unigram", tok)
	}
	return u
}

func TestUnigramParse(t *testing.T) {
	u := loadUnigram(t, unigramJSON(false, "always", true))
	if u.UnkID() != 0 {
		t.Errorf("UnkID=%d want 0", u.UnkID())
	}
	if u.VocabSize() != 276 {
		t.Errorf("VocabSize=%d want 276", u.VocabSize())
	}
	if u.TokenToID("▁hello") != 4 || u.TokenToID("zzz") != -1 {
		t.Error("TokenToID wrong")
	}
	if u.IDToToken(6) != "ld" || u.IDToToken(9999) != "" {
		t.Error("IDToToken wrong")
	}
}

func TestUnigramViterbi(t *testing.T) {
	u := loadUnigram(t, unigramJSON(false, "always", true))
	cases := []struct {
		in   string
		want []int32
	}{
		// ▁hello(-1)+▁wor(-2)+ld(-2) = -5 beats ▁hello+▁world(-4.5) = -5.5
		// and ▁he(-4)+llo(-4)+... .
		{"hello world", []int32{4, 5, 6}},
		// ▁ab(-6.5) beats ▁(-3)+a(-5)+b(-5) = -13.
		{"ab", []int32{11}},
		{"ab hello", []int32{11, 4}},
		// Unknown run x,y,z fuses into one <unk>; leading ▁ is its own piece.
		{"xyz", []int32{3, 0}},
		{"hello xyz world", []int32{4, 3, 0, 5, 6}},
		{"xyzab", []int32{3, 0, 9, 10}},
		// é is unknown but h / llo are known.
		{"héllo", []int32{3, 15, 0, 8}},
		// Leading space is not doubled; a double space yields a bare ▁ piece.
		{" hello", []int32{4}},
		{"hello  world", []int32{4, 3, 5, 6}},
		// Added tokens match atomically.
		{"<s>hello world</s>", []int32{1, 4, 5, 6, 2}},
		{"", nil},
	}
	for _, c := range cases {
		got := u.Encode(c.in)
		if !reflect.DeepEqual(got, c.want) && !(len(got) == 0 && len(c.want) == 0) {
			t.Errorf("Encode(%q) = %v (%v) want %v", c.in, got, pieces(u, got), c.want)
		}
	}
}

func TestUnigramByteFallback(t *testing.T) {
	u := loadUnigram(t, unigramJSON(true, "always", true))
	cases := []struct {
		in   string
		want []int32
	}{
		{"xyz", []int32{3, 140, 141, 142}}, // <0x78> <0x79> <0x7A>
		{"hello xyz world", []int32{4, 3, 140, 141, 142, 5, 6}},
		{"héllo", []int32{3, 15, 215, 189, 8}}, // é = C3 A9
		{"xyzab", []int32{3, 140, 141, 142, 9, 10}},
	}
	for _, c := range cases {
		if got := u.Encode(c.in); !reflect.DeepEqual(got, c.want) {
			t.Errorf("Encode(%q) = %v (%v) want %v", c.in, got, pieces(u, got), c.want)
		}
	}
	// Byte pieces are reassembled into UTF-8 on decode.
	for _, s := range []string{"héllo", "hello xyz world"} {
		if got := u.Decode(u.Encode(s)); got != s {
			t.Errorf("round-trip %q = %q", s, got)
		}
	}
}

func TestUnigramPrependScheme(t *testing.T) {
	first := loadUnigram(t, unigramJSON(false, "first", true))
	// After an added token the segment is no longer first, so no ▁ prefix.
	if got := first.Encode("<s>hello world</s>"); !reflect.DeepEqual(got, []int32{1, 15, 16, 8, 5, 6, 2}) {
		t.Errorf("first-scheme after <s>: %v (%v)", got, pieces(first, got))
	}
	if got := first.Encode("hello world"); !reflect.DeepEqual(got, []int32{4, 5, 6}) {
		t.Errorf("first-scheme plain: %v", got)
	}

	never := loadUnigram(t, unigramJSON(false, "never", true))
	cases := []struct {
		in   string
		want []int32
	}{
		{"hello world", []int32{15, 16, 8, 5, 6}},
		{"ab", []int32{9, 10}},
		{"xyz", []int32{0}},
		{" hello", []int32{4}},
	}
	for _, c := range cases {
		if got := never.Encode(c.in); !reflect.DeepEqual(got, c.want) {
			t.Errorf("never-scheme Encode(%q) = %v want %v", c.in, got, c.want)
		}
	}
	// "never" keeps the leading space on decode; "always" strips it.
	if got := never.Decode(never.Encode(" hello")); got != " hello" {
		t.Errorf("never decode = %q", got)
	}
	always := loadUnigram(t, unigramJSON(false, "always", true))
	if got := always.Decode(always.Encode(" hello")); got != "hello" {
		t.Errorf("always decode = %q", got)
	}

	// Legacy add_prefix_space=false maps to "never".
	legacy := strings.Replace(unigramJSON(false, "always", true), `"prepend_scheme":"always"`, `"add_prefix_space":false`, 1)
	if got := loadUnigram(t, legacy).Encode("ab"); !reflect.DeepEqual(got, []int32{9, 10}) {
		t.Errorf("add_prefix_space=false Encode(ab) = %v", got)
	}
}

func TestUnigramNoSplit(t *testing.T) {
	// With split=false Viterbi runs over the whole string; here the vocab
	// has no piece spanning a ▁ boundary so results match split=true.
	u := loadUnigram(t, unigramJSON(false, "always", false))
	if got := u.Encode("hello  world"); !reflect.DeepEqual(got, []int32{4, 3, 5, 6}) {
		t.Errorf("Encode = %v", got)
	}
}

func TestUnigramDecodeRoundTrip(t *testing.T) {
	u := loadUnigram(t, unigramJSON(false, "always", true))
	for _, s := range []string{"hello world", "ab hello", "hello  world"} {
		if got := u.Decode(u.Encode(s)); got != s {
			t.Errorf("round-trip %q = %q", s, got)
		}
	}
	if got := u.Decode(u.Encode("<s>hello world</s>")); got != "hello world" {
		t.Errorf("decode should skip specials; got %q", got)
	}
	if got := u.Decode(u.Encode("xyz")); got != "<unk>" {
		t.Errorf("unk decode = %q", got)
	}
}

func TestUnigramNormalizerAndPostProcessor(t *testing.T) {
	// T5-style: collapse runs of spaces, append </s>.
	js := strings.Replace(unigramJSON(false, "always", true), `"normalizer":null,`,
		`"normalizer":{"type":"Sequence","normalizers":[{"type":"Replace","pattern":{"Regex":" {2,}"},"content":" "},{"type":"Lowercase"}]},`, 1)
	js = strings.Replace(js, `"post_processor":null`,
		`"post_processor":{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"</s>","type_id":0}}],"pair":[],"special_tokens":{"</s>":{"id":"</s>","ids":[2],"tokens":["</s>"]}}}`, 1)
	u := loadUnigram(t, js)
	if got := u.Encode("HELLO   world"); !reflect.DeepEqual(got, []int32{4, 5, 6}) {
		t.Errorf("normalized Encode = %v (%v)", got, pieces(u, got))
	}
	if got := u.EncodeWithSpecial("hello world"); !reflect.DeepEqual(got, []int32{4, 5, 6, 2}) {
		t.Errorf("EncodeWithSpecial = %v", got)
	}
}

func TestNewUnigramErrors(t *testing.T) {
	if _, err := tokenizer.NewUnigram(nil, tokenizer.DefaultUnigramConfig()); err == nil {
		t.Error("expected error for empty vocab")
	}
	cfg := tokenizer.DefaultUnigramConfig()
	cfg.PrependScheme = "sometimes"
	if _, err := tokenizer.NewUnigram([]tokenizer.UnigramPiece{{Piece: "a"}}, cfg); err == nil {
		t.Error("expected error for bad prepend scheme")
	}
	cfg = tokenizer.DefaultUnigramConfig()
	cfg.UnkID = 5
	if _, err := tokenizer.NewUnigram([]tokenizer.UnigramPiece{{Piece: "a"}}, cfg); err == nil {
		t.Error("expected error for out-of-range unk id")
	}
}
