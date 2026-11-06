package tokenizer

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// phi3TokenizerPath finds microsoft/Phi-3.5-mini-instruct's tokenizer.json in
// the local HF cache, or "" when it is not downloaded.
func phi3TokenizerPath() string {
	home, _ := os.UserHomeDir()
	matches, _ := filepath.Glob(filepath.Join(home,
		".cache/huggingface/hub/models--microsoft--Phi-3.5-mini-instruct/snapshots/*/tokenizer.json"))
	if len(matches) == 0 {
		return ""
	}
	return matches[0]
}

// Expected ids were produced by HF `tokenizers` (Tokenizer.from_file ...
// encode(s, add_special_tokens=False)) on the same file.
func TestBPESentencePiecePhi3(t *testing.T) {
	path := phi3TokenizerPath()
	if path == "" {
		t.Skip("Phi-3.5-mini-instruct not in HF cache")
	}
	tk, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	bpe, ok := tk.(*BPE)
	if !ok || !bpe.IsSentencePiece() {
		t.Fatalf("expected SentencePiece-mode BPE, got %T spm=%v", tk, ok && bpe.IsSentencePiece())
	}
	cases := []struct {
		text string
		ids  []int32
		dec  string
	}{
		{"Hello world", []int32{15043, 3186}, "Hello world"},
		{"The capital of France is", []int32{450, 7483, 310, 3444, 338}, "The capital of France is"},
		{" leading space", []int32{29871, 8236, 2913}, " leading space"},
		{"multi  space\nnewline", []int32{2473, 29871, 2913, 13, 1482, 1220}, "multi  space\nnewline"},
		{"<|user|>\nWhat is 2+2?<|end|>\n<|assistant|>\n",
			[]int32{32010, 1724, 338, 29871, 29906, 29974, 29906, 29973, 32007, 32001}, ""},
		{"héllo 日本語 🙂", []int32{298, 3610, 417, 29871, 30325, 30346, 30968, 29871, 243, 162, 156, 133}, "héllo 日本語 🙂"},
		{"x<|end|>y", []int32{921, 32007, 343}, ""},
		{"", nil, ""},
	}
	for _, c := range cases {
		got := tk.Encode(c.text)
		if !reflect.DeepEqual(got, c.ids) {
			t.Errorf("Encode(%q) = %v, want %v", c.text, got, c.ids)
		}
		if c.dec != "" {
			if d := tk.Decode(c.ids); d != c.dec {
				t.Errorf("Decode(%v) = %q, want %q", c.ids, d, c.dec)
			}
		}
	}
}

// A GPT-2 style file (ByteLevel pre_tokenizer) must not flip into
// SentencePiece mode even when it carries a normalizer block.
func TestBPEByteLevelNotSentencePiece(t *testing.T) {
	js := `{"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]},
	        "normalizer":{"type":"Lowercase"},
	        "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false}}`
	tk, err := Parse([]byte(js))
	if err != nil {
		t.Fatal(err)
	}
	if tk.(*BPE).IsSentencePiece() {
		t.Fatal("ByteLevel tokenizer wrongly detected as SentencePiece")
	}
}
