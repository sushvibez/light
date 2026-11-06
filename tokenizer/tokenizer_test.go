package tokenizer_test

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"light/tokenizer"
)

func TestParseDispatch(t *testing.T) {
	bpeJSON := `{"model":{"type":"BPE","vocab":{"h":0,"i":1,"hi":2},"merges":["h i"]},"added_tokens":[{"id":10,"content":"<s>","special":true}]}`
	tok, err := tokenizer.Parse([]byte(bpeJSON))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := tok.(*tokenizer.BPE); !ok {
		t.Errorf("BPE json gave %T", tok)
	}
	if got := tok.Encode("<s>hi"); !reflect.DeepEqual(got, []int32{10, 2}) {
		t.Errorf("BPE Encode = %v", got)
	}
	if tok, _ := tokenizer.Parse([]byte(wordPieceJSON)); tok == nil {
		t.Error("WordPiece json gave nil")
	} else if _, ok := tok.(*tokenizer.WordPiece); !ok {
		t.Errorf("WordPiece json gave %T", tok)
	}
	if tok, _ := tokenizer.Parse([]byte(unigramJSON(false, "always", true))); tok == nil {
		t.Error("Unigram json gave nil")
	} else if _, ok := tok.(*tokenizer.Unigram); !ok {
		t.Errorf("Unigram json gave %T", tok)
	}
	if _, err := tokenizer.Parse([]byte(`{"model":{"type":"Nope","vocab":{}}}`)); err == nil {
		t.Error("expected error for unsupported model type")
	}
	if _, err := tokenizer.Parse([]byte(`{bad`)); err == nil {
		t.Error("expected error for bad json")
	}
}

func TestBPEPairMerges(t *testing.T) {
	// tokenizers >= 0.20 writes merges as [["a","b"], ...].
	js := `{"model":{"type":"BPE","vocab":{"h":0,"i":1,"hi":2},"merges":[["h","i"]]}}`
	bpe, err := tokenizer.LoadBPE(strings.NewReader(js))
	if err != nil {
		t.Fatal(err)
	}
	if got := bpe.Encode("hi"); !reflect.DeepEqual(got, []int32{2}) {
		t.Errorf("Encode = %v", got)
	}
}

func TestLoadFromDisk(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.json")
	if err := os.WriteFile(path, []byte(wordPieceJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := tok.Encode("the cat"); !reflect.DeepEqual(got, []int32{8, 9}) {
		t.Errorf("Encode = %v", got)
	}
	if _, err := tokenizer.Load(filepath.Join(dir, "missing.json")); err == nil {
		t.Error("expected error for missing file")
	}
}
