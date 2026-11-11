// Package chattemplate renders chat prompts with a model's own Hugging Face
// `chat_template` (a Jinja2 template), byte-for-byte like transformers'
// `tokenizer.apply_chat_template(..., tokenize=False)`.
//
// The Jinja engine lives in light/internal/jinja; this package adds what
// transformers layers on top: the environment settings (trim_blocks and
// lstrip_blocks on, the `tojson` override, `raise_exception` and
// `strftime_now` globals, the {% generation %} tag), the variables
// apply_chat_template passes (messages, tools, documents,
// add_generation_prompt and the tokenizer's special tokens), and locating
// the template in a checkpoint directory (chat_template.jinja, or the
// `chat_template` field of tokenizer_config.json, string or named list).
package chattemplate

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"light/internal/jinja"
)

// Message is one chat turn with plain-text content.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	// Images, when > 0, renders content as a list: that many
	// {"type": "image"} parts, then {"type": "text", "text": Content}
	// (the form multimodal templates read).
	Images int `json:"-"`
}

// RaiseError is returned when the template calls raise_exception(msg)
// (e.g. "Conversation roles must alternate user/assistant/...").
type RaiseError struct{ Msg string }

func (e *RaiseError) Error() string { return "chat template: " + e.Msg }

// ErrNoTemplate means the checkpoint ships no chat template.
var ErrNoTemplate = errors.New("chattemplate: no chat_template found")

// Template is a compiled chat template plus the special tokens the
// tokenizer exposes to it. Safe for concurrent use.
type Template struct {
	// Source is the Jinja source of the default template.
	Source string
	// Special holds bos_token, eos_token, ... as transformers'
	// tokenizer.special_tokens_map passes them to the template.
	Special map[string]string

	def  *jinja.Template
	tool *jinja.Template // the "tool_use" named template, if any
}

// Options are apply_chat_template's keyword arguments.
type Options struct {
	AddGenerationPrompt bool
	// Tools is a JSON array of tool schemas (OpenAI function format) or nil.
	Tools json.RawMessage
	// Documents is a JSON array of {title, text} documents or nil.
	Documents json.RawMessage
	// Vars are extra template variables (enable_thinking, date_string,
	// ...). Values may be Go scalars, []any, map[string]any or
	// json.RawMessage (which keeps object key order). They override the
	// special tokens, like transformers' **kwargs.
	Vars map[string]any
	// Now is the clock behind strftime_now; nil means time.Now.
	Now func() time.Time
}

// env is the jinja2 environment transformers builds in
// _cached_compile_jinja_template.
var env = jinja.Options{TrimBlocks: true, LstripBlocks: true}

// New compiles a template. special may be nil.
func New(source string, special map[string]string) (*Template, error) {
	def, err := jinja.Parse(source, env)
	if err != nil {
		return nil, err
	}
	if special == nil {
		special = map[string]string{}
	}
	return &Template{Source: source, Special: special, def: def}, nil
}

// specialTokenAttrs are PreTrainedTokenizerBase.SPECIAL_TOKENS_ATTRIBUTES.
var specialTokenAttrs = []string{"bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"}

// tokenString accepts "tok" or an AddedToken object {"content": "tok", ...}.
func tokenString(raw json.RawMessage) (string, bool) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", false
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s, true
	}
	var obj struct {
		Content *string `json:"content"`
	}
	if json.Unmarshal(raw, &obj) == nil && obj.Content != nil {
		return *obj.Content, true
	}
	return "", false
}

// LoadDir reads the chat template of an HF checkpoint directory (an HF
// cache snapshot): chat_template.jinja takes priority over the
// tokenizer_config.json `chat_template` field, as in transformers. Special
// tokens come from tokenizer_config.json, then special_tokens_map.json for
// any it lacks. It returns ErrNoTemplate when neither file has a template.
func LoadDir(dir string) (*Template, error) {
	cfg, err := readConfig(dir)
	if err != nil {
		return nil, err
	}
	special := specialTokens(dir, cfg)
	if raw, err := os.ReadFile(filepath.Join(dir, "chat_template.jinja")); err == nil {
		return New(string(raw), special)
	}
	field, ok := cfg["chat_template"]
	if !ok || string(field) == "null" {
		return nil, ErrNoTemplate
	}
	return fromConfigField(field, special)
}

// HasTemplate reports whether dir holds a chat template (a
// chat_template.jinja file or a tokenizer_config.json chat_template).
func HasTemplate(dir string) bool {
	if _, err := os.Stat(filepath.Join(dir, "chat_template.jinja")); err == nil {
		return true
	}
	cfg, err := readConfig(dir)
	if err != nil {
		return false
	}
	field, ok := cfg["chat_template"]
	return ok && string(field) != "null"
}

// SpecialTokens returns the special tokens transformers would pass to the
// template for the tokenizer in dir (tokenizer_config.json, then
// special_tokens_map.json for the ones it lacks).
func SpecialTokens(dir string) map[string]string {
	cfg, _ := readConfig(dir)
	return specialTokens(dir, cfg)
}

func readConfig(dir string) (map[string]json.RawMessage, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "tokenizer_config.json"))
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var cfg map[string]json.RawMessage
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("chattemplate: tokenizer_config.json: %w", err)
	}
	return cfg, nil
}

func specialTokens(dir string, cfg map[string]json.RawMessage) map[string]string {
	special := map[string]string{}
	addSpecial := func(m map[string]json.RawMessage, onlyMissing bool) {
		keys := append([]string{}, specialTokenAttrs...)
		for _, field := range []string{"extra_special_tokens", "model_specific_special_tokens"} {
			var extra map[string]json.RawMessage
			if json.Unmarshal(m[field], &extra) == nil {
				for k, v := range extra {
					if _, ok := m[k]; !ok {
						m[k] = v
					}
					keys = append(keys, k)
				}
			}
		}
		for _, k := range keys {
			if _, have := special[k]; have && onlyMissing {
				continue
			}
			if s, ok := tokenString(m[k]); ok {
				special[k] = s
			}
		}
	}
	if cfg != nil {
		addSpecial(cfg, false)
	}
	if raw, err := os.ReadFile(filepath.Join(dir, "special_tokens_map.json")); err == nil {
		var stm map[string]json.RawMessage
		if json.Unmarshal(raw, &stm) == nil {
			addSpecial(stm, true)
		}
	}
	return special
}

// fromConfigField compiles tokenizer_config.json's chat_template: a string,
// or a list of {name, template} with "default" (and optionally "tool_use",
// picked when tools are passed).
func fromConfigField(field json.RawMessage, special map[string]string) (*Template, error) {
	var src string
	if json.Unmarshal(field, &src) == nil {
		return New(src, special)
	}
	var named []struct {
		Name     string `json:"name"`
		Template string `json:"template"`
	}
	if err := json.Unmarshal(field, &named); err != nil {
		return nil, fmt.Errorf("chattemplate: chat_template is neither a string nor a list of {name, template}")
	}
	var t *Template
	var toolSrc string
	for _, n := range named {
		switch n.Name {
		case "default":
			var err error
			if t, err = New(n.Template, special); err != nil {
				return nil, err
			}
		case "tool_use":
			toolSrc = n.Template
		}
	}
	if t == nil {
		return nil, fmt.Errorf("chattemplate: named chat templates without a \"default\" one")
	}
	if toolSrc != "" {
		var err error
		if t.tool, err = jinja.Parse(toolSrc, env); err != nil {
			return nil, err
		}
	}
	return t, nil
}

// Render formats msgs (plain-text turns) like apply_chat_template.
func (t *Template) Render(msgs []Message, opt Options) (string, error) {
	l := jinja.NewList()
	for _, m := range msgs {
		d := jinja.NewDict()
		d.Set("role", m.Role)
		if m.Images > 0 {
			parts := jinja.NewList()
			for range m.Images {
				p := jinja.NewDict()
				p.Set("type", "image")
				parts.Items = append(parts.Items, p)
			}
			p := jinja.NewDict()
			p.Set("type", "text")
			p.Set("text", m.Content)
			parts.Items = append(parts.Items, p)
			d.Set("content", parts)
		} else {
			d.Set("content", m.Content)
		}
		l.Items = append(l.Items, d)
	}
	return t.render(l, opt)
}

// RenderJSON formats a JSON array of OpenAI-style message objects (which
// may carry tool_calls, reasoning_content, list content, ...).
func (t *Template) RenderJSON(messages json.RawMessage, opt Options) (string, error) {
	v, err := jinja.FromJSON(messages)
	if err != nil {
		return "", fmt.Errorf("chattemplate: messages: %w", err)
	}
	l, ok := v.(*jinja.List)
	if !ok {
		return "", errors.New("chattemplate: messages must be a JSON array")
	}
	return t.render(aliasReasoning(l), opt)
}

// aliasReasoning is transformers' _alias_reasoning_fields: a message's
// reasoning is exposed as both `thinking` and `reasoning_content`.
func aliasReasoning(l *jinja.List) *jinja.List {
	out := jinja.NewList()
	for _, it := range l.Items {
		m, ok := it.(*jinja.Dict)
		if !ok {
			out.Items = append(out.Items, it)
			continue
		}
		var reasoning any
		found := false
		for _, k := range []string{"reasoning_content", "thinking", "reasoning"} {
			if v, ok := m.Get(k); ok && v != nil {
				reasoning, found = v, true
				break
			}
		}
		if !found {
			out.Items = append(out.Items, m)
			continue
		}
		d := jinja.NewDict()
		d.Set("thinking", reasoning)
		for _, k := range m.Keys() {
			v, _ := m.Get(k)
			d.Set(k, v)
		}
		d.Set("reasoning_content", reasoning)
		out.Items = append(out.Items, d)
	}
	return out
}

func (t *Template) render(messages *jinja.List, opt Options) (string, error) {
	now := opt.Now
	if now == nil {
		now = time.Now
	}
	vars := map[string]any{
		"raise_exception": jinja.Func(func(args []any, kw *jinja.Dict) (any, error) {
			msg := ""
			if len(args) > 0 {
				if s, ok := args[0].(string); ok {
					msg = s
				} else {
					msg = fmt.Sprint(args[0])
				}
			}
			return nil, &RaiseError{Msg: msg}
		}),
		"strftime_now": jinja.Func(func(args []any, kw *jinja.Dict) (any, error) {
			if len(args) != 1 {
				return nil, errors.New("strftime_now() takes exactly one argument")
			}
			f, ok := args[0].(string)
			if !ok {
				return nil, errors.New("strftime_now() argument must be str")
			}
			return Strftime(now(), f)
		}),
	}
	for k, v := range t.Special {
		vars[k] = v
	}
	for k, v := range opt.Vars {
		c, err := jinja.FromGo(v)
		if err != nil {
			return "", fmt.Errorf("chattemplate: var %s: %w", k, err)
		}
		vars[k] = c
	}
	tpl := t.def
	vars["tools"] = nil
	if len(opt.Tools) > 0 && string(opt.Tools) != "null" {
		tools, err := jinja.FromJSON(opt.Tools)
		if err != nil {
			return "", fmt.Errorf("chattemplate: tools: %w", err)
		}
		vars["tools"] = tools
		if t.tool != nil {
			tpl = t.tool
		}
	}
	vars["documents"] = nil
	if len(opt.Documents) > 0 && string(opt.Documents) != "null" {
		docs, err := jinja.FromJSON(opt.Documents)
		if err != nil {
			return "", fmt.Errorf("chattemplate: documents: %w", err)
		}
		vars["documents"] = docs
	}
	vars["messages"] = messages
	vars["add_generation_prompt"] = opt.AddGenerationPrompt
	out, err := tpl.Render(vars)
	if err != nil {
		var re *RaiseError
		if errors.As(err, &re) {
			return "", re
		}
		return "", err
	}
	return out, nil
}
