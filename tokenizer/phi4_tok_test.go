package tokenizer

import (
	"os"
	"path/filepath"
	"testing"
)

// GPT-4o vocabulary (microsoft/Phi-4-mini-instruct tokenizer.json): Split
// regex written as behavior Removed + invert, with `(?i:…)` groups and
// `\s+(?!\S)`. Reference ids from transformers 5.21
// (AutoTokenizer, add_special_tokens=False).
var phi4TokRef = map[string][]int32{
	"<|system|>You are a concise assistant.<|end|><|user|>Explain in one sentence why the sky is blue.<|end|><|assistant|>": {200022, 3575, 553, 261, 82463, 29186, 13, 200020, 200021, 176289, 306, 1001, 21872, 4436, 290, 17307, 382, 9861, 13, 200020, 200019},
	"Hello world!  Multiple   spaces\n\nand newlines\r\n 12345 numbers, Ünïcödé, 日本語テキスト, it's don't I'LL":                  {13225, 2375, 0, 220, 39109, 256, 18608, 279, 427, 620, 10105, 370, 220, 7633, 2548, 8663, 11, 120241, 191375, 43369, 377, 11, 17428, 40909, 16056, 18368, 38236, 11, 4275, 4128, 3413, 7454},
	"def f(x):\n    return x**2  # comment\n": {1314, 285, 4061, 1883, 271, 622, 1215, 410, 17, 220, 1069, 5375, 198},
}

func TestPhi4MiniTokenizerMatchesHF(t *testing.T) {
	home, _ := os.UserHomeDir()
	paths, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--microsoft--Phi-4-mini-instruct/snapshots/*/tokenizer.json"))
	if len(paths) == 0 {
		t.Skip("microsoft/Phi-4-mini-instruct tokenizer.json not in HF cache")
	}
	want := phi4TokRef
	f, err := os.Open(paths[0])
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	tok, err := LoadBPE(f)
	if err != nil {
		t.Fatal(err)
	}
	for text, ids := range want {
		got := tok.Encode(text)
		if len(got) != len(ids) {
			t.Errorf("%q: got %v want %v", text, got, ids)
			continue
		}
		for i := range ids {
			if got[i] != ids[i] {
				t.Errorf("%q: got %v want %v", text, got, ids)
				break
			}
		}
		if d := tok.Decode(got); d != text {
			t.Errorf("decode %q != %q", d, text)
		}
	}
}
