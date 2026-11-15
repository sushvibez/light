package tokenizer

import (
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// mergeSymbolsRef is the original quadratic merge loop (rescan every pair,
// rebuild the slice per merge), kept as the reference for mergeSymbols.
func (b *BPE) mergeSymbolsRef(tokens []string) []string {
	if len(tokens) < 2 {
		return tokens
	}
	for {
		bestRank := int32(-1)
		bestIdx := -1
		for i := 0; i < len(tokens)-1; i++ {
			if rank, ok := b.merges[mergeKey{tokens[i], tokens[i+1]}]; ok {
				if bestIdx == -1 || rank < bestRank {
					bestRank = rank
					bestIdx = i
				}
			}
		}
		if bestIdx == -1 {
			break
		}
		merged := tokens[bestIdx] + tokens[bestIdx+1]
		next := make([]string, 0, len(tokens)-1)
		next = append(next, tokens[:bestIdx]...)
		next = append(next, merged)
		next = append(next, tokens[bestIdx+2:]...)
		tokens = next
	}
	return tokens
}

func runeSyms(s string) []string {
	var out []string
	for _, r := range s {
		out = append(out, string(r))
	}
	return out
}

// TestMergeSymbolsMatchesRef: a synthetic merge table with equal-rank
// overlaps ("a a" on runs of a), chains and competing pairs, checked
// against the reference loop on random strings.
func TestMergeSymbolsMatchesRef(t *testing.T) {
	merges := []string{"a a", "b c", "aa a", "c b", "aa aa", "a b", "bc a", "ab c", "aaa b", "cb a", "c c", "b b", "aaaa a", "bb b"}
	vocab := map[string]int32{"a": 0, "b": 1, "c": 2}
	b, err := NewBPE(vocab, merges)
	if err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewPCG(5, 5))
	for n := range 3000 {
		L := 1 + r.IntN(40)
		var sb strings.Builder
		for range L {
			sb.WriteByte("abc"[r.IntN(3)])
		}
		s := sb.String()
		if n%5 == 0 {
			s = strings.Repeat("a", L)
		}
		want := b.mergeSymbolsRef(runeSyms(s))
		got := b.mergeSymbols(runeSyms(s))
		if !slices.Equal(got, want) {
			t.Fatalf("%q: got %q want %q", s, got, want)
		}
	}
}

func loadQwenBPE(tb testing.TB) *BPE {
	home, _ := os.UserHomeDir()
	paths, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/tokenizer.json"))
	if len(paths) == 0 {
		tb.Skip("Qwen2.5-0.5B-Instruct tokenizer.json not in HF cache")
	}
	f, err := os.Open(paths[0])
	if err != nil {
		tb.Fatal(err)
	}
	defer f.Close()
	tok, err := LoadBPE(f)
	if err != nil {
		tb.Fatal(err)
	}
	return tok
}

const benchPrompt = `<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Explain, in two short paragraphs, why the sky is blue and why sunsets are red.
Then write a Go function that reverses a slice in place:

func Reverse[T any](s []T) {
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		s[i], s[j] = s[j], s[i]
	}
}
Ünïcödé, 日本語テキスト, numbers 1234567 and URLs https://example.com/a?b=c&d=e.<|im_end|>
<|im_start|>assistant
`

func longPunct() string {
	r := rand.New(rand.NewPCG(3, 3))
	const set = "!@#$%^&*()-_=+[]{};:,.<>/?|~`'\""
	var sb strings.Builder
	for range 10000 {
		sb.WriteByte(set[r.IntN(len(set))])
	}
	return sb.String()
}

// TestMergeSymbolsRealVocab: the real Qwen2.5 merge table on the prompt,
// a long punctuation run and long letter/digit runs encodes identically to
// the reference loop.
func TestMergeSymbolsRealVocab(t *testing.T) {
	b := loadQwenBPE(t)
	inputs := []string{benchPrompt, longPunct()[:3000], strings.Repeat("ab", 1500), strings.Repeat("=", 2000), strings.Repeat("hello", 400)}
	for _, in := range inputs {
		var syms []string
		for i := 0; i < len(in); i++ {
			syms = append(syms, string(b.byteEncoder[in[i]]))
		}
		want := b.mergeSymbolsRef(slices.Clone(syms))
		got := b.mergeSymbols(syms)
		if !slices.Equal(got, want) {
			t.Fatalf("%.40q...: %d vs %d pieces", in, len(got), len(want))
		}
	}
}

func BenchmarkBPEEncode(b *testing.B) {
	tok := loadQwenBPE(b)
	cases := map[string]string{
		"prompt":     benchPrompt,
		"punct10k":   longPunct(),
		"letters10k": strings.Repeat("abcdefghij", 1000),
	}
	for _, name := range []string{"prompt", "punct10k", "letters10k"} {
		text := cases[name]
		b.Run(name, func(b *testing.B) {
			b.SetBytes(int64(len(text)))
			for range b.N {
				tok.Encode(text)
			}
		})
	}
}
