package engine

import (
	"strings"

	"light/chattemplate"
	"light/tokenizer"
)

// Message is one chat turn. Role is "system", "user" or "assistant".
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Template renders a conversation into the prompt string a model family
// was trained on, ending with the assistant prefix so generation starts at
// the reply. When the checkpoint ships a chat_template (HF
// tokenizer_config.json / chat_template.jinja, or GGUF
// tokenizer.chat_template) that renders, Load executes it through
// light/chattemplate (Source "jinja", byte-identical to transformers'
// apply_chat_template); otherwise the hand-ported family template below is
// used (Source "builtin").
type Template struct {
	Family string
	Render func(msgs []Message) string
	// Stop lists extra end-of-turn strings whose ids are added to EOS.
	Stop []string
	// Source is "jinja" or "builtin".
	Source string
	// Jinja is the model's compiled chat template when Source is "jinja".
	Jinja *chattemplate.Template
}

var templates = map[string]Template{
	"chatml": {
		Family: "chatml", Stop: []string{"<|im_end|>"},
		Render: func(msgs []Message) string {
			var b strings.Builder
			for _, m := range msgs {
				b.WriteString("<|im_start|>" + m.Role + "\n" + m.Content + "<|im_end|>\n")
			}
			b.WriteString("<|im_start|>assistant\n")
			return b.String()
		},
	},
	"llama3": {
		Family: "llama3", Stop: []string{"<|eot_id|>", "<|end_of_text|>"},
		Render: func(msgs []Message) string {
			var b strings.Builder
			b.WriteString("<|begin_of_text|>")
			for _, m := range msgs {
				b.WriteString("<|start_header_id|>" + m.Role + "<|end_header_id|>\n\n" + strings.TrimSpace(m.Content) + "<|eot_id|>")
			}
			b.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
			return b.String()
		},
	},
	"phi3": {
		Family: "phi3", Stop: []string{"<|end|>", "<|endoftext|>"},
		Render: func(msgs []Message) string {
			var b strings.Builder
			for _, m := range msgs {
				b.WriteString("<|" + m.Role + "|>\n" + m.Content + "<|end|>\n")
			}
			b.WriteString("<|assistant|>\n")
			return b.String()
		},
	},
	// Phi-4 (mini/reasoning): the Phi-3 markers without the newlines.
	"phi4": {
		Family: "phi4", Stop: []string{"<|end|>", "<|endoftext|>"},
		Render: func(msgs []Message) string {
			var b strings.Builder
			for _, m := range msgs {
				b.WriteString("<|" + m.Role + "|>" + m.Content + "<|end|>")
			}
			b.WriteString("<|assistant|>")
			return b.String()
		},
	},
	// Mistral / Llama-2 instruct: system folded into the first user turn.
	"mistral": {
		Family: "mistral", Stop: []string{"</s>"},
		Render: func(msgs []Message) string {
			var b strings.Builder
			b.WriteString("<s>")
			system := ""
			for _, m := range msgs {
				switch m.Role {
				case "system":
					system = m.Content
				case "user":
					c := m.Content
					if system != "" {
						c = system + "\n\n" + c
						system = ""
					}
					b.WriteString("[INST] " + c + " [/INST]")
				case "assistant":
					b.WriteString(" " + m.Content + "</s>")
				}
			}
			return b.String()
		},
	},
	// Gemma 4: <|turn>role\n…<turn|>\n, system is a turn of its own.
	"gemma4": {
		Family: "gemma4", Stop: []string{"<turn|>"},
		Render: func(msgs []Message) string {
			var b strings.Builder
			b.WriteString("<bos>")
			for _, m := range msgs {
				role := m.Role
				if role == "assistant" {
					role = "model"
				}
				b.WriteString("<|turn>" + role + "\n" + m.Content + "<turn|>\n")
			}
			b.WriteString("<|turn>model\n")
			return b.String()
		},
	},
	"gemma": {
		Family: "gemma", Stop: []string{"<end_of_turn>"},
		Render: func(msgs []Message) string {
			var b strings.Builder
			b.WriteString("<bos>")
			system := ""
			for _, m := range msgs {
				role := m.Role
				if role == "system" {
					system = m.Content
					continue
				}
				if role == "assistant" {
					role = "model"
				}
				c := m.Content
				if system != "" && role == "user" {
					c = system + "\n\n" + c
					system = ""
				}
				b.WriteString("<start_of_turn>" + role + "\n" + c + "<end_of_turn>\n")
			}
			b.WriteString("<start_of_turn>model\n")
			return b.String()
		},
	},
}

// TemplateFor picks the chat template by the special tokens present in the
// vocabulary; falls back to a plain "User:/Assistant:" transcript.
func TemplateFor(tok *tokenizer.BPE) Template {
	t := builtinFor(tok)
	t.Source = "builtin"
	return t
}

func builtinFor(tok *tokenizer.BPE) Template {
	has := func(s string) bool { return tok.TokenToID(s) >= 0 }
	switch {
	case has("<|im_start|>"):
		return templates["chatml"]
	case has("<|start_header_id|>"):
		return templates["llama3"]
	case has("<|user|>") && has("<|end|>") && has("<|tool|>"):
		return templates["phi4"] // Phi-4-mini adds the tool-call markers; Phi-3.5 has none
	case has("<|user|>") && has("<|end|>"):
		return templates["phi3"]
	case has("<|turn>") && has("<turn|>"):
		return templates["gemma4"]
	case has("<start_of_turn>"):
		return templates["gemma"]
	case has("[INST]") || has("<s>"):
		return templates["mistral"]
	}
	return Template{
		Family: "plain", Stop: []string{"\nUser:"},
		Render: func(msgs []Message) string {
			var b strings.Builder
			for _, m := range msgs {
				switch m.Role {
				case "system":
					b.WriteString(m.Content + "\n\n")
				case "user":
					b.WriteString("User: " + m.Content + "\n")
				case "assistant":
					b.WriteString("Assistant: " + m.Content + "\n")
				}
			}
			b.WriteString("Assistant:")
			return b.String()
		},
	}
}
