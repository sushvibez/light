package chattemplate

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Golden files come from scripts/verify/ref_chat_templates.py, which runs
// transformers' apply_chat_template(tokenize=False) on the same templates
// with strftime_now pinned to this instant.
var goldenNow = time.Date(2025, 3, 5, 14, 7, 9, 0, time.UTC)

type goldenCase struct {
	Name     string                     `json:"name"`
	Messages json.RawMessage            `json:"messages"`
	AGP      bool                       `json:"add_generation_prompt"`
	Kwargs   map[string]json.RawMessage `json:"kwargs"`
	Expected *string                    `json:"expected"`
	Error    string                     `json:"error"`
}

type golden struct {
	Repo     string            `json:"repo"`
	Revision string            `json:"revision"`
	Special  map[string]string `json:"special_tokens"`
	Cases    []goldenCase      `json:"cases"`
}

func TestHFGolden(t *testing.T) {
	dirs, err := filepath.Glob(filepath.Join("testdata", "hf", "*"))
	if err != nil || len(dirs) == 0 {
		t.Fatalf("no fixtures: %v", err)
	}
	for _, dir := range dirs {
		t.Run(filepath.Base(dir), func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join(dir, "golden.json"))
			if err != nil {
				t.Fatal(err)
			}
			var g golden
			if err := json.Unmarshal(raw, &g); err != nil {
				t.Fatal(err)
			}
			tpl, err := LoadDir(dir)
			if err != nil {
				t.Fatalf("LoadDir: %v", err)
			}
			// Every special token we expose must match transformers'; the
			// ones templates use (bos/eos) must be present iff it has them.
			for k, v := range tpl.Special {
				if want, ok := g.Special[k]; ok && want != v {
					t.Errorf("special %s = %q, transformers %q", k, v, want)
				}
			}
			for _, k := range []string{"bos_token", "eos_token"} {
				if _, ok := g.Special[k]; ok != (tpl.Special[k] != "") {
					t.Errorf("special %s: ours %q, transformers has it: %v", k, tpl.Special[k], ok)
				}
			}
			pass := 0
			for _, c := range g.Cases {
				opt := Options{AddGenerationPrompt: c.AGP, Now: func() time.Time { return goldenNow }, Vars: map[string]any{}}
				for k, v := range c.Kwargs {
					if k == "tools" {
						opt.Tools = v
					} else {
						opt.Vars[k] = v
					}
				}
				got, err := renderCase(tpl, c, opt)
				switch {
				case c.Expected == nil:
					if err == nil {
						t.Errorf("%s: want error %q, got output %q", c.Name, c.Error, got)
						continue
					}
					var re *RaiseError
					if errors.As(err, &re) && "TemplateError: "+re.Msg != c.Error {
						t.Errorf("%s: raise_exception(%q), transformers: %q", c.Name, re.Msg, c.Error)
						continue
					}
				case err != nil:
					t.Errorf("%s: %v", c.Name, err)
					continue
				case got != *c.Expected:
					t.Errorf("%s: mismatch\n got: %q\nwant: %q", c.Name, got, *c.Expected)
					continue
				}
				pass++
			}
			t.Logf("%s @ %.8s: %d/%d cases byte-identical", g.Repo, g.Revision, pass, len(g.Cases))
		})
	}
}

// renderCase uses the plain []Message API when the messages are plain
// {role, content} strings, so both entry points are covered.
func renderCase(tpl *Template, c goldenCase, opt Options) (string, error) {
	var plain []Message
	var generic []map[string]any
	if json.Unmarshal(c.Messages, &generic) == nil {
		simple := true
		for _, m := range generic {
			_, isStr := m["content"].(string)
			simple = simple && len(m) == 2 && isStr && m["role"] != nil
		}
		if simple && json.Unmarshal(c.Messages, &plain) == nil {
			return tpl.Render(plain, opt)
		}
	}
	return tpl.RenderJSON(c.Messages, opt)
}

func TestLoadDirNoTemplate(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{"bos_token": "<s>"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadDir(dir); !errors.Is(err, ErrNoTemplate) {
		t.Fatalf("got %v, want ErrNoTemplate", err)
	}
}

func TestLoadDirNamedTemplates(t *testing.T) {
	dir := t.TempDir()
	cfg := `{"eos_token": {"content": "</s>", "lstrip": false},
	  "chat_template": [{"name": "default", "template": "D{{ eos_token }}"},
	                    {"name": "tool_use", "template": "T{{ tools | length }}"}]}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	tpl, err := LoadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := tpl.Render(nil, Options{}); got != "D</s>" {
		t.Errorf("default template: %q", got)
	}
	if got, _ := tpl.Render(nil, Options{Tools: json.RawMessage(`[{"a":1},{"b":2}]`)}); got != "T2" {
		t.Errorf("tool_use template: %q", got)
	}
}

func TestReasoningAlias(t *testing.T) {
	tpl, err := New(`{% for m in messages %}{{ m | tojson }}{% endfor %}`, nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := tpl.RenderJSON(json.RawMessage(`[{"role":"assistant","content":"x","reasoning":"r"}]`), Options{})
	if err != nil {
		t.Fatal(err)
	}
	want := `{"thinking": "r", "role": "assistant", "content": "x", "reasoning": "r", "reasoning_content": "r"}`
	if got != want {
		t.Errorf("got %s\nwant %s", got, want)
	}
}

func TestStrftime(t *testing.T) {
	tm := time.Date(2025, 3, 5, 14, 7, 9, 0, time.UTC)
	for f, want := range map[string]string{
		"%d %b %Y":          "05 Mar 2025",
		"%d %B %Y":          "05 March 2025",
		"%Y-%m-%d %H:%M:%S": "2025-03-05 14:07:09",
		"%A %I%p %j %%":     "Wednesday 02PM 064 %",
		"%-d/%-m":           "5/3",
	} {
		got, err := Strftime(tm, f)
		if err != nil || got != want {
			t.Errorf("Strftime(%q) = %q, %v; want %q", f, got, err, want)
		}
	}
	if _, err := Strftime(tm, "%Q"); err == nil || !strings.Contains(err.Error(), "unsupported") {
		t.Errorf("want unsupported directive error, got %v", err)
	}
}
