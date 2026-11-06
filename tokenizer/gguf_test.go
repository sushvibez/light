package tokenizer

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// llama.cpp ships reference tokenizations for its vocab-only GGUF files
// (models/ggml-vocab-<name>.gguf{,.inp,.out}, tests/test-tokenizer-0.cpp:
// add_special=false, parse_special=false). Point LLAMA_CPP_DIR at a
// checkout; the test skips when it is absent.
func llamaCppModels(t *testing.T) string {
	t.Helper()
	dir := os.Getenv("LLAMA_CPP_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".unsloth", "llama.cpp")
	}
	if _, err := os.Stat(filepath.Join(dir, "models")); err != nil {
		t.Skip("llama.cpp checkout not found (set LLAMA_CPP_DIR)")
	}
	return filepath.Join(dir, "models")
}

func readVocabTests(t *testing.T, base string) (inputs []string, outputs [][]int32) {
	t.Helper()
	raw, err := os.ReadFile(base + ".inp")
	if err != nil {
		t.Skip(err)
	}
	outRaw, err := os.ReadFile(base + ".out")
	if err != nil {
		t.Skip(err)
	}
	const sep = "\n__ggml_vocab_test__\n"
	inputs = strings.Split(strings.TrimSuffix(string(raw), sep), sep)
	for _, line := range strings.Split(strings.TrimRight(string(outRaw), "\n"), "\n") {
		var ids []int32
		for _, f := range strings.Fields(line) {
			n, err := strconv.Atoi(f)
			if err != nil {
				t.Fatal(err)
			}
			ids = append(ids, int32(n))
		}
		outputs = append(outputs, ids)
	}
	if len(inputs) != len(outputs) {
		t.Fatalf("%s: %d inputs vs %d outputs", base, len(inputs), len(outputs))
	}
	return inputs, outputs
}

func TestGGUFVocabMatchesLlamaCpp(t *testing.T) {
	dir := llamaCppModels(t)
	for _, name := range []string{"llama-spm", "phi-3", "llama-bpe", "qwen2", "gpt-2", "deepseek-llm", "deepseek-coder", "falcon", "command-r", "mpt"} {
		t.Run(name, func(t *testing.T) {
			base := filepath.Join(dir, "ggml-vocab-"+name+".gguf")
			if _, err := os.Stat(base); err != nil {
				t.Skip(err)
			}
			bpe, info, err := LoadGGUF(base)
			if err != nil {
				t.Fatal(err)
			}
			inputs, outputs := readVocabTests(t, base)
			fails := 0
			for i, text := range inputs {
				got := bpe.Encode(text)
				if !equalIDs(got, outputs[i]) {
					fails++
					if fails <= 5 {
						t.Errorf("%q:\n got  %v\n want %v", text, got, outputs[i])
					}
				}
				// Round trip: llama.cpp's detokenizer output is not in the
				// fixtures, so only check that decoding our own ids gives
				// the text back for BPE vocabularies (SPM drops the prefix
				// space and NFKC-normalises some inputs).
				if info.Model == "gpt2" && len(got) > 0 && bpe.Decode(got) != text {
					t.Errorf("decode(%q) = %q", text, bpe.Decode(got))
				}
			}
			t.Logf("%s: model=%s pre=%q %d/%d cases match, bos=%d eos=%d", name, info.Model, info.Pre, len(inputs)-fails, len(inputs), info.BOS, info.EOS)
		})
	}
}

func equalIDs(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// A real model file rather than a vocab-only fixture: the Qwen2-0.5B GGUF's
// embedded tokenizer must agree with the HF tokenizer.json of the same
// model, specials included.
func TestGGUFQwen2MatchesHFTokenizer(t *testing.T) {
	home, _ := os.UserHomeDir()
	ggufPath := filepath.Join(home, ".cache/light/gguf/qwen2-0.5b-instruct-q4_k_m.gguf")
	hfMatches, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/*/tokenizer.json"))
	if _, err := os.Stat(ggufPath); err != nil || len(hfMatches) == 0 {
		t.Skip("Qwen2-0.5B GGUF or HF tokenizer not cached")
	}
	g, info, err := LoadGGUF(ggufPath)
	if err != nil {
		t.Fatal(err)
	}
	hfTok, err := Load(hfMatches[0])
	if err != nil {
		t.Fatal(err)
	}
	hf := hfTok.(*BPE)
	if info.Pre != "qwen2" || info.EOS != 151645 && info.EOS != 151643 {
		t.Fatalf("info %+v", info)
	}
	for _, text := range []string{
		"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2? Explain in one sentence.<|im_end|>\n<|im_start|>assistant\n",
		"The quick brown fox jumps over the lazy dog.   Multiple   spaces, tabs\t\tand\nnewlines.",
		"日本語のテキストと emoji 🦙🚀 mixed with 'contractions' I've, you'll, 3.14159 and 1234567890",
		"def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
	} {
		got, want := g.Encode(text), hf.Encode(text)
		if !equalIDs(got, want) {
			t.Errorf("%q:\n gguf %v\n hf   %v", text, got, want)
		}
		if d := g.Decode(got); d != text {
			t.Errorf("decode mismatch: %q", d)
		}
	}
}
