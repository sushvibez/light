package tokenizer

import (
	"os"
	"path/filepath"
	"testing"
)

// Reference ids from transformers 5.21 (AutoTokenizer, add_special_tokens=
// False) on code-like text where the model's Split regex and the GPT-2
// pattern disagree: Qwen2 splits digits singly and glues "(" to "n", Llama 3
// groups digits in threes.
func TestHFSplitRegexMatchesTransformers(t *testing.T) {
	home, _ := os.UserHomeDir()
	texts := []string{
		"def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
		"Order #1234567890 costs $3.14159 (incl. VAT)!!  Really?\n\n\tYes.",
	}
	cases := []struct {
		repo string
		want [][]int32
	}{
		{"Qwen--Qwen2-0.5B-Instruct", [][]int32{
			{750, 15801, 1445, 982, 262, 470, 308, 421, 308, 366, 220, 17, 770, 15801, 1445, 12, 16, 8, 488, 15801, 1445, 12, 17, 340},
			{4431, 671, 16, 17, 18, 19, 20, 21, 22, 23, 24, 15, 7049, 400, 18, 13, 16, 19, 16, 20, 24, 320, 91157, 13, 47592, 8, 2928, 220, 28208, 1939, 197, 9454, 13},
		}},
		{"meta-llama--Llama-3.2-1B-Instruct", [][]int32{
			{755, 16178, 1471, 997, 262, 471, 308, 422, 308, 366, 220, 17, 775, 16178, 1471, 12, 16, 8, 489, 16178, 1471, 12, 17, 340},
			{4531, 674, 4513, 10961, 16474, 15, 7194, 400, 18, 13, 9335, 2946, 320, 92257, 13, 48692, 8, 3001, 220, 29308, 1980, 197, 9642, 13},
		}},
	}
	for _, c := range cases {
		m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--"+c.repo+"/snapshots/*/tokenizer.json"))
		if len(m) == 0 {
			t.Logf("%s not cached, skipping", c.repo)
			continue
		}
		tk, err := Load(m[0])
		if err != nil {
			t.Fatal(err)
		}
		bpe := tk.(*BPE)
		for i, text := range texts {
			if got := bpe.Encode(text); !equalIDs(got, c.want[i]) {
				t.Errorf("%s %q:\n got  %v\n want %v", c.repo, text, got, c.want[i])
			}
			if d := bpe.Decode(c.want[i]); d != text {
				t.Errorf("%s decode %q", c.repo, d)
			}
		}
	}
}
