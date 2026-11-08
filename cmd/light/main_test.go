package main

import (
	"bufio"
	"io"
	"slices"
	"strings"
	"testing"

	"light/engine"
	"light/hub"
	"light/models"
)

func entry(id, kind string) engine.Entry { return engine.Entry{Spec: engine.Spec{ID: id}, Kind: kind} }

func TestResolve(t *testing.T) {
	es := []engine.Entry{
		entry("Qwen/Qwen2.5-0.5B-Instruct", "safetensors"),
		entry("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", "safetensors"),
		entry("gguf/qwen2-0.5b-instruct-q4_k_m.gguf", "gguf"),
		entry("openai/whisper-tiny", "stt"),
	}
	for _, c := range []struct {
		q, want string
		kinds   map[string]bool
	}{
		{"Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", nil}, // exact id beats the -GPTQ substring
		{"qwen2.5-0.5b-instruct", "Qwen/Qwen2.5-0.5B-Instruct", nil},      // exact repo name, any case
		{"gptq", "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", nil},             // unique substring
		{"qwen2-0.5b-instruct-q4_k_m", "gguf/qwen2-0.5b-instruct-q4_k_m.gguf", nil},
		{"q4_k_m.gguf", "gguf/qwen2-0.5b-instruct-q4_k_m.gguf", nil},
		{"whisper", "openai/whisper-tiny", nil},
	} {
		e, err := resolve(es, c.q, c.kinds)
		if err != nil || e.Spec.ID != c.want {
			t.Errorf("resolve(%q) = %q, %v; want %q", c.q, e.Spec.ID, err, c.want)
		}
	}
	if _, err := resolve(es, "0.5b", nil); err == nil || !strings.Contains(err.Error(), "matches 3 models") {
		t.Errorf("ambiguous query: err %v", err)
	}
	if _, err := resolve(es, "whisper", chatKinds); err == nil {
		t.Error("run must not resolve a speech-to-text model")
	}
	if _, err := resolve(es, "llama", nil); err == nil || !strings.Contains(err.Error(), "light pull") {
		t.Errorf("missing model: err %v", err)
	}
}

func names(fs []hub.RepoFile) []string {
	var out []string
	for _, f := range fs {
		out = append(out, f.Name)
	}
	slices.Sort(out)
	return out
}

func files(ns ...string) []hub.RepoFile {
	var out []hub.RepoFile
	for _, n := range ns {
		out = append(out, hub.RepoFile{Name: n, Size: 1})
	}
	return out
}

func TestPlanPull(t *testing.T) {
	// Safetensors repo: weights, configs, tokenizer; no onnx / original / README / .bin.
	p := planPull("Qwen/Qwen2.5-7B", files("config.json", "generation_config.json", "model-00001-of-00002.safetensors",
		"model-00002-of-00002.safetensors", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
		"merges.txt", "vocab.json", "README.md", "LICENSE", ".gitattributes", "onnx/model.onnx", "original/consolidated.00.pth",
		"pytorch_model.bin", "chat_template.jinja"))
	want := []string{"chat_template.jinja", "config.json", "generation_config.json", "merges.txt", "model-00001-of-00002.safetensors",
		"model-00002-of-00002.safetensors", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"}
	if got := names(p.cache); !slices.Equal(got, want) || p.ggufDir != nil || p.choices != nil {
		t.Errorf("safetensors plan %v / %v / %v", got, names(p.ggufDir), names(p.choices))
	}
	// One GGUF: into the GGUF directory.
	p = planPull("org/m-GGUF", files("m-Q4_K_M.gguf", "README.md", "config.json"))
	if got := names(p.ggufDir); !slices.Equal(got, []string{"m-Q4_K_M.gguf"}) || p.cache != nil {
		t.Errorf("single gguf plan %v / %v", got, names(p.cache))
	}
	// Several: choices, nothing downloaded.
	p = planPull("org/m-GGUF", files("m-Q4_K_M.gguf", "m-Q8_0.gguf"))
	if len(p.choices) != 2 || p.cache != nil || p.ggufDir != nil {
		t.Errorf("multi gguf plan %+v", p)
	}
	// PyTorch-only checkpoint.
	p = planPull("tiiuae/falcon-rw-1b", files("config.json", "pytorch_model.bin", "tokenizer.json", "README.md"))
	if got := names(p.cache); !slices.Equal(got, []string{"config.json", "pytorch_model.bin", "tokenizer.json"}) {
		t.Errorf("pytorch plan %v", got)
	}
	// Moshi keeps its GGUF and SentencePiece model in the HF cache.
	p = planPull(models.MoshikoRepo, files(models.MoshikoGGUF, models.MoshikoTokenizer, "README.md", "other.safetensors"))
	if got := names(p.cache); !slices.Equal(got, []string{models.MoshikoGGUF, models.MoshikoTokenizer}) || p.ggufDir != nil {
		t.Errorf("moshi plan %v / %v", got, names(p.ggufDir))
	}
}

func TestReadMessage(t *testing.T) {
	in := bufio.NewReader(strings.NewReader("hello\r\n\"\"\"\nline one\nline two\n\"\"\"\nlast"))
	for _, want := range []string{"hello", "line one\nline two", "last"} {
		got, err := readMessage(in)
		if err != nil || got != want {
			t.Fatalf("readMessage = %q, %v; want %q", got, err, want)
		}
	}
	if _, err := readMessage(in); err != io.EOF {
		t.Fatalf("at end: err %v, want EOF", err)
	}
}
