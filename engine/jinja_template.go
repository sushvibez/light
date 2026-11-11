package engine

import (
	"os"
	"path/filepath"

	"light/chattemplate"
	"light/gguf"
	"light/hub"
	"light/tokenizer"
)

// probe is the conversation a model's chat template must render before we
// trust it over the hand-ported family template.
var probe = []chattemplate.Message{{Role: "user", Content: "Hello"}}

// hfChatTemplate loads the chat template an HF repo ships: its
// chat_template.jinja, or tokenizer_config.json's chat_template. Only
// small tokenizer files are fetched (normally already cached next to
// tokenizer.json).
func hfChatTemplate(repo string) (*chattemplate.Template, error) {
	cfgPath, err := hub.Download(repo, "main", "tokenizer_config.json")
	if err != nil {
		return nil, err
	}
	dir := filepath.Dir(cfgPath)
	if !chattemplate.HasTemplate(dir) {
		// Newer checkpoints keep the template only in chat_template.jinja,
		// which a weights-only download may not have fetched yet.
		p, err := hub.Download(repo, "main", "chat_template.jinja")
		if err != nil {
			return nil, chattemplate.ErrNoTemplate
		}
		if filepath.Dir(p) != dir {
			src, err := os.ReadFile(p)
			if err != nil {
				return nil, err
			}
			return chattemplate.New(string(src), chattemplate.SpecialTokens(dir))
		}
	}
	return chattemplate.LoadDir(dir)
}

// ggufChatTemplate compiles the GGUF's tokenizer.chat_template, with
// bos_token / eos_token resolved from its tokenizer.ggml.*_token_id through
// tok (llama.cpp passes the same two strings).
func ggufChatTemplate(path string, tok *tokenizer.BPE) (*chattemplate.Template, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	g, err := gguf.Open(f)
	if err != nil {
		return nil, err
	}
	src, _ := g.Meta["tokenizer.chat_template"].(string)
	if src == "" {
		return nil, chattemplate.ErrNoTemplate
	}
	special := map[string]string{}
	for key, name := range map[string]string{
		"tokenizer.ggml.bos_token_id": "bos_token",
		"tokenizer.ggml.eos_token_id": "eos_token",
	} {
		if id, ok := metaID(g.Meta[key]); ok {
			if s := tok.IDToToken(int32(id)); s != "" {
				special[name] = s
			}
		}
	}
	return chattemplate.New(src, special)
}

func metaID(v any) (int64, bool) {
	switch n := v.(type) {
	case uint8:
		return int64(n), true
	case int8:
		return int64(n), true
	case uint16:
		return int64(n), true
	case int16:
		return int64(n), true
	case uint32:
		return int64(n), true
	case int32:
		return int64(n), true
	case uint64:
		return int64(n), true
	case int64:
		return n, true
	}
	return 0, false
}

// loadChatTemplate finds the model's own template: a GGUF's embedded one
// first (it belongs to those exact weights), then the HF repo's. It
// returns nil when there is none or it does not render the probe.
func loadChatTemplate(spec Spec, tok *tokenizer.BPE) *chattemplate.Template {
	var cands []func() (*chattemplate.Template, error)
	if spec.GGUF != "" {
		cands = append(cands, func() (*chattemplate.Template, error) { return ggufChatTemplate(spec.GGUF, tok) })
	}
	if spec.Repo != "" {
		cands = append(cands, func() (*chattemplate.Template, error) { return hfChatTemplate(spec.Repo) })
	}
	for _, load := range cands {
		ct, err := load()
		if err != nil {
			continue
		}
		if _, err := ct.Render(probe, chattemplate.Options{AddGenerationPrompt: true}); err != nil {
			continue
		}
		return ct
	}
	return nil
}

// withJinja makes fam render through the model's chat template. Family and
// Stop stay those of the hand-ported family. A conversation the template
// rejects via raise_exception (Gemma 2 has no system role, Mistral wants
// strictly alternating turns) falls back to the family renderer, which
// folds such turns instead of failing the request.
func withJinja(fam Template, ct *chattemplate.Template) Template {
	t := fam
	t.Source = "jinja"
	t.Jinja = ct
	builtin := fam.Render
	t.Render = func(msgs []Message) string {
		cm := make([]chattemplate.Message, len(msgs))
		for i, m := range msgs {
			cm[i] = chattemplate.Message{Role: m.Role, Content: m.Content, Images: len(m.Images)}
		}
		out, err := ct.Render(cm, chattemplate.Options{AddGenerationPrompt: true})
		if err != nil {
			return builtin(msgs)
		}
		return out
	}
	return t
}
