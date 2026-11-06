package engine

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"light/gguf"
	"light/tokenizer"
)

// fakeHFCache lays out models--<org>--<name>/refs/main + snapshots/<sha>/
// under a temporary HF_HOME with the tokenizer files of a chattemplate
// fixture, so hub.Download resolves them offline.
func fakeHFCache(t *testing.T, repos map[string]string) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HF_HOME", home)
	for repo, fixture := range repos {
		repoDir := filepath.Join(home, "hub", "models--"+strings.ReplaceAll(repo, "/", "--"))
		snap := filepath.Join(repoDir, "snapshots", "0123abcd")
		if err := os.MkdirAll(snap, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(filepath.Join(repoDir, "refs"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(repoDir, "refs", "main"), []byte("0123abcd"), 0o644); err != nil {
			t.Fatal(err)
		}
		src := filepath.Join("..", "chattemplate", "testdata", "hf", fixture)
		for _, f := range []string{"tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"} {
			raw, err := os.ReadFile(filepath.Join(src, f))
			if os.IsNotExist(err) {
				continue
			}
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(snap, f), raw, 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestJinjaTemplateFromHFRepo(t *testing.T) {
	fakeHFCache(t, map[string]string{
		"test/gemma2": "gemma2", "test/qwen3": "qwen3", "test/llama": "llama3.2",
	})
	user := []Message{{Role: "user", Content: "Hello! What is 2+2?"}}
	sysUser := []Message{{Role: "system", Content: "Be brief."}, {Role: "user", Content: "Hi"}}

	// Gemma 2: template from tokenizer_config.json, writes <bos> itself.
	ct := loadChatTemplate(Spec{Repo: "test/gemma2"}, nil)
	if ct == nil {
		t.Fatal("gemma2: no chat template loaded")
	}
	g := withJinja(templates["gemma"], ct)
	if g.Source != "jinja" || g.Family != "gemma" || g.Stop[0] != "<end_of_turn>" {
		t.Fatalf("gemma2 template: source %q family %q stop %v", g.Source, g.Family, g.Stop)
	}
	if got, want := g.Render(user), "<bos><start_of_turn>user\nHello! What is 2+2?<end_of_turn>\n<start_of_turn>model\n"; got != want {
		t.Errorf("gemma2 user:\n got %q\nwant %q", got, want)
	}
	// Its template raises on a system turn; Render falls back to the
	// built-in family template, which folds the system prompt in.
	if got, want := g.Render(sysUser), templates["gemma"].Render(sysUser); got != want {
		t.Errorf("gemma2 system fallback:\n got %q\nwant %q", got, want)
	}

	// Qwen3: tokenizer_config.json template (thinking-aware).
	ct = loadChatTemplate(Spec{Repo: "test/qwen3"}, nil)
	if ct == nil {
		t.Fatal("qwen3: no chat template loaded")
	}
	if got, want := withJinja(templates["chatml"], ct).Render(sysUser),
		"<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"; got != want {
		t.Errorf("qwen3:\n got %q\nwant %q", got, want)
	}

	// Llama 3.2: chat_template.jinja; the official template adds the
	// knowledge-cutoff / date header the hand-coded one lacks.
	ct = loadChatTemplate(Spec{Repo: "test/llama"}, nil)
	if ct == nil {
		t.Fatal("llama: no chat template loaded")
	}
	got := withJinja(templates["llama3"], ct).Render(sysUser)
	if !strings.HasPrefix(got, "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: ") ||
		!strings.HasSuffix(got, "Be brief.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n") {
		t.Errorf("llama3.2: %q", got)
	}

	// No repo / nothing cached: no template, builtin stays.
	if ct := loadChatTemplate(Spec{}, nil); ct != nil {
		t.Error("empty spec produced a template")
	}
}

func TestJinjaTemplateFromGGUF(t *testing.T) {
	w := gguf.NewWriter()
	w.AddString("general.architecture", "llama")
	w.AddString("tokenizer.ggml.model", "gpt2")
	w.AddString("tokenizer.ggml.pre", "qwen2")
	w.AddStringArray("tokenizer.ggml.tokens", []string{"a", "b", "ab", "<s>", "</s>"})
	w.AddStringArray("tokenizer.ggml.merges", []string{"a b"})
	w.AddU32("tokenizer.ggml.bos_token_id", 3)
	w.AddU32("tokenizer.ggml.eos_token_id", 4)
	w.AddString("tokenizer.chat_template", "{{ bos_token }}{% for m in messages %}[{{ m.role }}] {{ m.content }}{{ eos_token }}{% endfor %}{% if add_generation_prompt %}[assistant] {% endif %}")
	raw, err := w.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	tok, _, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	ct := loadChatTemplate(Spec{GGUF: path}, tok)
	if ct == nil {
		t.Fatal("no template from GGUF metadata")
	}
	got := withJinja(TemplateFor(tok), ct).Render([]Message{{Role: "user", Content: "ab"}})
	if want := "<s>[user] ab</s>[assistant] "; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestBuiltinTemplateSource(t *testing.T) {
	w := gguf.NewWriter()
	w.AddString("tokenizer.ggml.model", "gpt2")
	w.AddString("tokenizer.ggml.pre", "qwen2")
	w.AddStringArray("tokenizer.ggml.tokens", []string{"a", "<|im_start|>", "<|im_end|>"})
	w.AddStringArray("tokenizer.ggml.merges", []string{})
	raw, _ := w.Bytes()
	path := filepath.Join(t.TempDir(), "notpl.gguf")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	tok, _, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	if ct := loadChatTemplate(Spec{GGUF: path}, tok); ct != nil {
		t.Fatal("GGUF without tokenizer.chat_template produced a template")
	}
	if tp := TemplateFor(tok); tp.Source != "builtin" || tp.Family != "chatml" {
		t.Fatalf("source %q family %q", tp.Source, tp.Family)
	}
}
