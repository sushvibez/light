package jinja

import (
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
)

type jinjaCase struct {
	Name     string  `json:"name"`
	Template string  `json:"template"`
	Trim     bool    `json:"trim_blocks"`
	Lstrip   bool    `json:"lstrip_blocks"`
	Expected *string `json:"expected"`
	Error    string  `json:"error"`
}

type raiseErr struct{ msg string }

func (e *raiseErr) Error() string { return e.msg }

// TestJinja2Golden renders the cases of testdata/jinja_cases.json, whose
// expected output came from jinja2 itself (scripts/verify/ref_jinja_cases.py),
// and requires byte equality, or an error where jinja2 raised.
func TestJinja2Golden(t *testing.T) {
	raw, err := os.ReadFile("testdata/jinja_cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var file struct {
		Context json.RawMessage `json:"context"`
		Cases   []jinjaCase     `json:"cases"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		t.Fatal(err)
	}
	ctxV, err := FromJSON(file.Context)
	if err != nil {
		t.Fatal(err)
	}
	ctx := ctxV.(*Dict)
	pass := 0
	for _, c := range file.Cases {
		vars := map[string]any{
			"raise_exception": Func(func(args []any, kw *Dict) (any, error) {
				return nil, &raiseErr{toStr(args[0])}
			}),
		}
		for _, k := range ctx.Keys() {
			vars[k.(string)], _ = ctx.Get(k)
		}
		got, err := renderString(c.Template, Options{TrimBlocks: c.Trim, LstripBlocks: c.Lstrip}, vars)
		switch {
		case c.Expected == nil:
			if err == nil {
				t.Errorf("%s: jinja2 raised %q, we rendered %q", c.Name, c.Error, got)
				continue
			}
			var re *raiseErr
			if strings.HasPrefix(c.Error, "TemplateError: ") && (!errors.As(err, &re) || "TemplateError: "+re.msg != c.Error) {
				t.Errorf("%s: want %q, got %v", c.Name, c.Error, err)
				continue
			}
		case err != nil:
			t.Errorf("%s: %v\n template: %q", c.Name, err, c.Template)
			continue
		case got != *c.Expected:
			t.Errorf("%s:\n template: %q\n      got: %q\n     want: %q", c.Name, c.Template, got, *c.Expected)
			continue
		}
		pass++
	}
	t.Logf("%d/%d jinja2 cases identical", pass, len(file.Cases))
}

func renderString(src string, opt Options, vars map[string]any) (string, error) {
	tpl, err := Parse(src, opt)
	if err != nil {
		return "", err
	}
	return tpl.Render(vars)
}

func hf() Options { return Options{TrimBlocks: true, LstripBlocks: true} }

func TestWhitespaceControl(t *testing.T) {
	for _, c := range []struct{ src, want string }{
		{"  {% if true %}\n    x\n  {% endif %}\nend", "    x\nend"},
		{"a  {%- if true -%}  b  {%- endif -%}  c", "abc"},
		{"{{ 'a' }}\n{{ 'b' }}\n", "a\nb"},
		{"x\n  {%+ if true %}y{% endif +%}\nz", "x\n  y\nz"},
		{"a {# comment #}\n b", "a  b"},
		{"{% raw %}{{ x }}{% endraw %}", "{{ x }}"},
		{"{% generation %}gen{% endgeneration %}!", "gen!"},
	} {
		got, err := renderString(c.src, hf(), nil)
		if err != nil || got != c.want {
			t.Errorf("%q: got %q, %v; want %q", c.src, got, err, c.want)
		}
	}
}

func TestLoopsAndScoping(t *testing.T) {
	vars := map[string]any{"xs": NewList(int64(1), int64(2), int64(3))}
	for _, c := range []struct{ src, want string }{
		{"{% for x in xs %}{{ loop.index0 }}{{ loop.first }}{{ loop.last }}{{ loop.length }} {% endfor %}", "0TrueFalse3 1FalseFalse3 2FalseTrue3 "},
		{"{% for x in [] %}x{% else %}none{% endfor %}", "none"},
		{"{% set x = 0 %}{% for i in xs %}{% set x = x + i %}{% endfor %}{{ x }}", "0"},
		{"{% set ns = namespace(x=0) %}{% for i in xs %}{% set ns.x = ns.x + i %}{% endfor %}{{ ns.x }}", "6"},
		{"{% for i in xs %}{% if i == 2 %}{% break %}{% endif %}{{ i }}{% endfor %}", "1"},
	} {
		got, err := renderString(c.src, hf(), vars)
		if err != nil || got != c.want {
			t.Errorf("%q: got %q, %v; want %q", c.src, got, err, c.want)
		}
	}
}

func TestFilters(t *testing.T) {
	vars := map[string]any{}
	v, _ := FromJSON([]byte(`{"name": "f", "args": {"b": 1, "a": [true, null, 1.5]}}`))
	vars["tool"] = v
	for _, c := range []struct{ src, want string }{
		{"{{ tool | tojson }}", `{"name": "f", "args": {"b": 1, "a": [true, null, 1.5]}}`},
		{"{{ tool.args | tojson(indent=2) }}", "{\n  \"b\": 1,\n  \"a\": [\n    true,\n    null,\n    1.5\n  ]\n}"},
		{"{{ tool | string }}", `{'name': 'f', 'args': {'b': 1, 'a': [True, None, 1.5]}}`},
		{"{{ '  x ' | trim }}|{{ 'ab' | upper }}|{{ [1,2] | length }}", "x|AB|2"},
		{"{{ ['a','b'] | join(', ') }}|{{ missing | default('d') }}", "a, b|d"},
	} {
		got, err := renderString(c.src, hf(), vars)
		if err != nil || got != c.want {
			t.Errorf("%q: got %q, %v; want %q", c.src, got, err, c.want)
		}
	}
}

// TestUnsupportedIsAnError makes sure constructs the engine does not
// implement fail loudly at parse time instead of rendering differently.
func TestUnsupportedIsAnError(t *testing.T) {
	for _, src := range []string{
		"{% include 'x.jinja' %}",
		"{% extends 'base' %}",
		"{% block body %}{% endblock %}",
		"{% import 'm' as m %}",
		"{% call foo() %}{% endcall %}",
		"{% for x in xs recursive %}{% endfor %}",
		"{{ x | truncate(3) }}",
		"{{ x is nosuchtest }}",
		"{{ f(*args) }}",
		"{% if x %}",
	} {
		if _, err := Parse(src, hf()); err == nil {
			t.Errorf("%q: expected a parse error", src)
		}
	}
}

func TestRuntimeErrors(t *testing.T) {
	for _, src := range []string{
		"{{ x.y }}",
		"{{ 1 + 'a' }}",
		"{{ [1].append(2) }}",
		"{{ 'a'.format(x=1) }}{{ '{:>3}'.format(1) }}",
		"{{ {} | tojson if false else (x | tojson) }}",
	} {
		_, err := renderString(src, hf(), nil)
		var je *Error
		if err == nil || !errors.As(err, &je) {
			t.Errorf("%q: expected a *jinja.Error, got %v", src, err)
		}
	}
}

func TestPyFloatRepr(t *testing.T) {
	for f, want := range map[float64]string{
		1: "1.0", 0.1: "0.1", 1e16: "1e+16", 1e15: "1000000000000000.0", 1e-5: "1e-05",
		0.0001: "0.0001", 1.5e300: "1.5e+300", -2.5: "-2.5", 123456.789: "123456.789",
	} {
		if got := pyFloatRepr(f); got != want {
			t.Errorf("pyFloatRepr(%v) = %q, want %q", f, got, want)
		}
	}
}
