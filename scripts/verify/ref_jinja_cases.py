#!/usr/bin/env python3
"""Golden renders for internal/jinja/jinja_test.go, straight from jinja2.

    python scripts/verify/ref_jinja_cases.py [out.json]

Each case is rendered in the environment transformers builds for chat
templates (ImmutableSandboxedEnvironment, trim_blocks + lstrip_blocks,
loopcontrols, the tojson override, raise_exception), or with
trim/lstrip switched off where a case says so. The output records the
rendered string, or that jinja2 raised (Go must then return an error too).
"""
import json
import os
import sys

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment


def make_env(trim, lstrip):
    env = ImmutableSandboxedEnvironment(trim_blocks=trim, lstrip_blocks=lstrip,
                                        extensions=[jinja2.ext.loopcontrols])

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


CTX = {
    "messages": [
        {"role": "system", "content": "  Be brief.  "},
        {"role": "user", "content": "Hi\nthere"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Bye"},
    ],
    "d": {"b": 1, "a": [1, 2], "c": {"x": "y"}},
    "tool": {"name": "f", "description": "café — \"q\" 'a'", "params": {"n": 1.5, "ok": True, "none": None}},
    "words": ["banana", "Apple", "cherry", "apple"],
    "nums": [3, 1, 2, 5, 4],
    "text": "  Hello, World!  ",
    "emoji": "hi \U0001f642 <b>&</b>",
}

CASES = [
    # --- whitespace control -------------------------------------------------
    ("lstrip_trim", "  {% if true %}\n    x\n  {% endif %}\nend\n"),
    ("for_lines", "{% for i in [1,2,3] %}\n  {{ i }}\n{% endfor %}\n"),
    ("dash_both", "a  {%- if true -%}  b  {%- endif -%}  c"),
    ("dash_var", "a \n {{- 'b' -}} \n c"),
    ("plus_disables", "x\n  {%+ if true %}y{% endif +%}\nz"),
    ("comment", "a {# c #}\n b\n  {# c2 #}\nx {#- c3 -#}  y"),
    ("var_not_trimmed", "{{ 'a' }}\n{{ 'b' }}\n"),
    ("var_not_lstripped", "  {{ 'v' }}\n"),
    ("not_line_start", "a {% if true %}b{% endif %}\n"),
    ("crlf", "l1\r\n{% if true %}\r\nl2\r\n{% endif %}\r\n"),
    ("two_trailing_newlines", "x\n\n"),
    ("raw", "{% raw %}{{ x }} {% if %}{% endraw %}"),
    ("trim_off", "  {% if true %}\n    x\n  {% endif %}\n", {"trim": False, "lstrip": False}),
    ("trim_only", "  {% if true %}\n    x\n  {% endif %}\n", {"trim": True, "lstrip": False}),
    ("lstrip_only", "  {% if true %}\n    x\n  {% endif %}\n", {"trim": False, "lstrip": True}),
    # --- loops ----------------------------------------------------------------
    ("loop_vars", "{% for x in 'abc' %}{{ loop.index }}{{ loop.index0 }}{{ loop.first }}{{ loop.last }}{{ loop.length }}{{ loop.revindex }}{{ loop.revindex0 }}|{% endfor %}"),
    ("loop_prev_next", "{% for x in nums %}{{ loop.previtem }}<{{ x }}>{{ loop.nextitem }},{% endfor %}"),
    ("loop_cycle", "{% for x in nums %}{{ loop.cycle('a', 'b') }}{% endfor %}"),
    ("for_else", "{% for x in [] %}x{% else %}empty{% endfor %}"),
    ("for_if", "{% for x in range(10) if x is even %}{{ x }}/{{ loop.length }},{% endfor %}"),
    ("break_continue", "{% for x in range(10) %}{% if x == 2 %}{% continue %}{% endif %}{% if x == 5 %}{% break %}{% endif %}{{ x }}{% endfor %}"),
    ("nested_loop", "{% for a in [1,2] %}{% for b in [3,4] %}{{ loop.index }}{% endfor %}{{ loop.index }};{% endfor %}"),
    ("dict_iter", "{% for k in d %}{{ k }}{% endfor %}"),
    ("unpack_items", "{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}"),
    ("items_filter", "{% for k, v in d | items %}{{ k }}{% endfor %}"),
    ("reverse_slice_loop", "{% for m in messages[::-1] %}{{ m.role }},{% endfor %}"),
    # --- scoping --------------------------------------------------------------
    ("loop_scope", "{% set x = 1 %}{% for i in [1,2] %}{{ x }}{% set x = x + 1 %}{{ x }}{% endfor %}{{ x }}"),
    ("namespace", "{% set ns = namespace(n=0, s='') %}{% for i in range(4) %}{% set ns.n = ns.n + i %}{% set ns.s = ns.s ~ i %}{% endfor %}{{ ns.n }} {{ ns.s }}"),
    ("namespace_empty", "{% set ns = namespace() %}{% set ns.i = 3 %}{{ ns.i }}"),
    ("if_scope", "{% if true %}{% set y = 5 %}{% endif %}{{ y }}"),
    ("toplevel_reassign", "{% set messages = messages[1:] %}{{ messages | length }}{% for m in messages %}{{ m.role[0] }}{% endfor %}"),
    ("tuple_set", "{% set a, b = 1, 2 %}{{ a }}{{ b }}{% set t = 1, 2 %}{{ t }}"),
    ("with", "{% with a = 1, b = 2 %}{{ a + b }}{% endwith %}[{{ a }}]"),
    ("set_block", "{% set x %}  hi {{ 1 + 1 }} {% endset %}[{{ x }}]"),
    ("set_block_filter", "{% set x | upper | trim %}  hi  {% endset %}[{{ x }}]"),
    ("macro", "{% macro greet(name, greeting='Hi') %}{{ greeting }}, {{ name }}!{% endmacro %}{{ greet('Bob') }} {{ greet('Al', greeting='Yo') }} {{ greet() }}"),
    ("filter_block", "{% filter upper %}abc{{ 'd' }}{% endfilter %}"),
    # --- literals & printing ---------------------------------------------------
    ("print_values", "{{ none }}|{{ true }}|{{ False }}|{{ 1.0 }}|{{ 1e20 }}|{{ 1e16 }}|{{ 1e15 }}|{{ 0.1 + 0.2 }}|{{ 1/3 }}|{{ 1e-5 }}|{{ 0.0001 }}|{{ -0.0 }}|{{ 2.5e-7 }}"),
    ("print_containers", "{{ [1, 'a', none, true, 2.5] }}|{{ {'a': [1, {'b': 'c'}]} }}|{{ ('x',) }}|{{ (1, 2) }}|{{ [] }}|{{ {} }}"),
    ("print_quotes", "{{ [\"it's\", 'q\"', 'both\\'\"', 'nl\\n', 'tab\\t', 'bs\\\\'] }}"),
    ("repr_unicode", "{{ [tool.description, emoji, '\\x00\\x7f\\xa0\\u200b'] }}"),
    ("tool_string", "{{ tool | string }}"),
    ("string_escapes", "{{ 'a\\nb\\t\\u00e9\\x41\\101\\'\\\\ \\q' }}"),
    ("string_concat_literals", "{{ 'a' \"b\" 'c' }}"),
    ("dict_trailing_comma", "{{ {'a': 1,}['a'] }}{{ [1,2,][1] }}"),
    ("int_float_literals", "{{ 1_000 }} {{ 1.5e3 }} {{ 12.0 }}"),
    # --- operators --------------------------------------------------------------
    ("arith", "{{ 7 // 2 }} {{ -7 // 2 }} {{ 7 % 3 }} {{ -7 % 3 }} {{ 7 / 2 }} {{ 2 ** 10 }} {{ 1 + 2 * 3 }} {{ (1 + 2) * 3 }} {{ 'ab' * 3 }} {{ 4 / 2 }} {{ 7.5 // 2 }} {{ -2 ** 2 }} {{ true + 1 }}"),
    ("concat", "{{ 1 ~ 'a' ~ none ~ missing ~ [1] }}"),
    ("compare", "{{ 1 < 2 < 3 }} {{ 3 > 2 > 2 }} {{ 'a' in 'abc' }} {{ 'x' not in ['a'] }} {{ 1 == 1.0 }} {{ [1,2] == [1,2] }} {{ 'b' > 'a' }} {{ none == none }} {{ {'a':1} == {'a':1} }} {{ 1 == true }} {{ 'a' in d }}"),
    ("and_or", "{{ '' or 'def' }} {{ 'a' and 'b' }} {{ none or 0 }} {{ not none }} {{ 0 and 1 }} {{ not 1 == 2 }}"),
    ("not_precedence", "{{ not messages is string }} {{ not 'a' in 'b' }} {{ - 2 | abs }}"),
    ("condexpr", "{{ 'y' if 1 else 'n' }}{{ 'y' if 0 }}|{{ 'a' if false else 'b' if true else 'c' }}"),
    ("filter_precedence", "{{ 'x' + text | trim + 'y' }}|{{ nums | length - 1 }}"),
    ("neg_binds_before_filter", "{{ -nums | length }}"),
    ("percent_format", "{{ '%s-%d %%' % ('a', 3) }}|{{ '%s' % 'one' }}|{{ '%.2f' % 3.14159 }}|{{ 'n=%r' % 'x' }}"),
    # --- subscripts ------------------------------------------------------------------
    ("slices", "{{ nums[1:] }}{{ nums[::-1] }}{{ 'hello'[1:3] }}{{ nums[-1] }}{{ 'héllo'[1] }}{{ nums[-2:] }}{{ nums[:-2:2] }}{{ nums[10:] }}{{ nums[1:100] }}{{ 'abc'[::-1] }}{{ nums[-100:2] }}{{ nums[3:1:-1] }}"),
    ("index_attr", "{{ messages[0]['content'] }}|{{ messages[0].role }}|{{ d.c.x }}|{{ d['a'][1] }}|{{ nums.0 }}|{{ messages[-1].content }}"),
    ("missing", "[{{ messages[0].nope }}][{{ nums[10] }}][{{ d['zz'] }}][{{ missing }}]"),
    # --- undefined ----------------------------------------------------------------
    ("undefined_ops", "{{ x }}|{{ x | length }}|{{ x is defined }}|{% for i in x %}no{% endfor %}|{{ x == x }}|{{ x is none }}|{{ not x }}|{{ x | default('d') }}|{{ 'a' in x }}|{{ x is iterable }}"),
    ("undefined_attr_error", "{{ x.y }}"),
    ("undefined_add_error", "{{ x + 'a' }}"),
    ("undefined_call_error", "{{ x() }}"),
    # --- filters ------------------------------------------------------------------
    ("f_strings", "{{ text | trim }}|{{ 'xxhixx' | trim('x') }}|{{ 'aBc' | upper }}|{{ 'aBc' | lower }}|{{ 'hello wORLD-foo bar(baz' | title }}|{{ 'hELLO world' | capitalize }}|{{ 'aaa' | replace('a', 'b', 2) }}|{{ 'abc' | replace('b', '') }}"),
    ("f_length", "{{ 'héllo\U0001f642' | length }} {{ nums | count }} {{ d | length }}"),
    ("f_default", "{{ x | default('d') }}|{{ '' | default('d', true) }}|{{ none | default('d') }}|{{ x | d }}|{{ 0 | d(5, boolean=true) }}"),
    ("f_join", "{{ nums | join(', ') }}|{{ messages | join('/', attribute='role') }}|{{ 'abc' | join('-') }}|{{ [1, none, 'a'] | join }}"),
    ("f_first_last", "{{ nums | first }} {{ nums | last }} {{ 'abc' | first }} {{ [] | first }}|"),
    ("f_list", "{{ 'abc' | list }} {{ d | list }} {{ range(3) | list }} {{ range(1, 10, 3) | list }} {{ range(5, 0, -2) | list }}"),
    ("f_reverse", "{{ nums | reverse | list }} {{ 'abc' | reverse }}"),
    ("f_sort", "{{ nums | sort }} {{ nums | sort(reverse=true) }} {{ words | sort }} {{ words | sort(case_sensitive=true) }} {{ messages | sort(attribute='role') | map(attribute='content') | list }}"),
    ("f_unique", "{{ words | unique | list }} {{ [1, 2, 1, 3] | unique | list }}"),
    ("f_map", "{{ messages | map(attribute='role') | list }} {{ words | map('upper') | list }} {{ messages | map(attribute='nope', default='z') | list }} {{ ['1', '2'] | map('int') | sum }}"),
    ("f_select", "{{ messages | selectattr('role', 'equalto', 'user') | map(attribute='content') | list }}|{{ messages | rejectattr('role', 'eq', 'user') | list | length }}|{{ nums | select('odd') | list }}|{{ nums | reject('lt', 3) | list }}|{{ [0, 1, '', 'a'] | select | list }}|{{ messages | selectattr('content') | list | length }}"),
    ("f_numbers", "{{ nums | sum }} {{ nums | min }} {{ nums | max }} {{ words | max }} {{ '42' | int }} {{ '4.7' | int }} {{ 'x' | int(7) }} {{ '2.5' | float }} {{ 3 | float }} {{ -3 | abs }} {{ 2.567 | round(2) }} {{ 2.5 | round }} {{ 3.5 | round }} {{ 2.1 | round(method='ceil') }}"),
    ("f_misc", "{{ 5 | string }}{{ none | string }}|{{ 'a b  c' | wordcount }}|{{ '<a href=\"x\">&\\'</a>' | e }}|{{ '%s=%s' | format('a', 1) }}|{{ d | dictsort }}|{{ {'b': 2, 'a': 3} | dictsort(by='value', reverse=true) }}"),
    ("f_indent", "[{{ 'a\nb\n\nc' | indent(2) }}][{{ 'a\nb' | indent(first=true) }}][{{ 'a\n\nb' | indent('>', blank=true) }}]"),
    ("f_tojson", "{{ tool | tojson }}|{{ d | tojson(indent=2) }}|{{ messages[:1] | tojson(sort_keys=true) }}|{{ emoji | tojson }}|{{ emoji | tojson(ensure_ascii=true) }}|{{ [1.0, 2, true, none, 'a\\n\"b\\x01'] | tojson }}|{{ {} | tojson(indent=4) }}|{{ [[]] | tojson(indent=1) }}|{{ d | tojson(separators=(',', ':')) }}"),
    ("f_attr", "{{ messages[0] | attr('role') }}|"),
    # --- tests ----------------------------------------------------------------------
    ("tests_types", "{{ x is defined }}{{ x is undefined }}{{ none is none }}{{ 'a' is string }}{{ 1 is number }}{{ 1.5 is number }}{{ true is number }}{{ 1 is integer }}{{ true is integer }}{{ 1.0 is float }}{{ d is mapping }}{{ nums is mapping }}{{ nums is iterable }}{{ 'a' is iterable }}{{ 1 is iterable }}{{ nums is sequence }}{{ true is boolean }}{{ 1 is boolean }}"),
    ("tests_values", "{{ true is true }}{{ 1 is true }}{{ false is false }}{{ 0 is false }}{{ 4 is even }}{{ 3 is odd }}{{ 9 is divisibleby 3 }}{{ 9 is divisibleby(4) }}{{ 'a' is eq 'a' }}{{ 'a' is equalto('b') }}{{ none is sameas none }}{{ false is sameas false }}{{ 'a' is in 'abc' }}{{ 'abc' is lower }}{{ 'ABC' is upper }}{{ 1 is not none }}{{ 2 is gt 1 }}{{ 2 is le 1 }}{{ 'upper' is filter }}{{ 'odd' is test }}"),
    ("tests_and_else", "{{ 'y' if x is defined else 'n' }}{{ 'y' if nums is iterable and nums is not string else 'n' }}"),
    # --- methods ------------------------------------------------------------------
    ("str_methods", "[{{ text.strip() }}][{{ 'xxhix'.lstrip('x') }}][{{ text.rstrip() }}][{{ 'a,b,,c'.split(',') }}][{{ '  a  b \\n c '.split() }}][{{ ' a b c '.split(none, 1) }}][{{ 'a,b,c'.rsplit(',', 1) }}][{{ 'a b c'.rsplit(maxsplit=1) }}][{{ 'abc'.startswith(('x', 'a')) }}][{{ 'abc'.endswith('bc') }}][{{ 'aaa'.replace('a', 'b', 1) }}][{{ 'héllo'.find('l') }}][{{ 'abc'.find('z') }}][{{ 'banana'.count('an') }}][{{ ', '.join(['a', 'b']) }}][{{ 'a\\nb\\r\\nc'.splitlines() }}][{{ 'hello world 1st'.title() }}][{{ 'hELLO'.capitalize() }}][{{ '{} and {name}'.format(1, name='x') }}][{{ 'abc'.upper() }}{{ 'ABC'.lower() }}][{{ '12'.isdigit() }}{{ 'a1'.isalpha() }}{{ ' '.isspace() }}][{{ 'prefix-x'.removeprefix('prefix-') }}]"),
    ("dict_methods", "{{ d.get('b') }} {{ d.get('zz') }} {{ d.get('zz', 'dflt') }} {{ d.keys() | list }} {{ d.values() | list | length }} {{ 'b' in d.keys() }}"),
    ("list_methods", "{{ nums.index(5) }} {{ nums.count(1) }}"),
    ("qwen3_think_split", "{% set content = '<think>\\nreason\\n</think>\\n\\nanswer' %}{{ content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') }}|{{ content.split('</think>')[-1].lstrip('\\n') }}"),
    # --- globals / errors -----------------------------------------------------------
    ("raise", "{% if true %}{{ raise_exception('Roles must alternate') }}{% endif %}"),
    ("dict_global", "{{ dict(a=1, b='x') }} {{ namespace(a=1).a }}"),
    ("list_append_unsafe", "{{ nums.append(1) }}"),
    ("dict_update_unsafe", "{{ d.update({'a': 1}) }}"),
    ("type_error_add", "{{ 1 + 'a' }}"),
    ("type_error_iter", "{% for x in 5 %}{% endfor %}"),
    ("str_plus_int_error", "{{ 'a' + 1 }}"),
    ("syntax_unclosed", "{% if true %}x"),
    ("syntax_bad_expr", "{{ 1 + }}"),
    ("syntax_unknown_tag", "{% frobnicate %}"),
    ("syntax_endfor_mismatch", "{% for x in nums %}{% endif %}"),
    ("unknown_filter", "{{ 1 | nosuchfilter }}"),
    ("unknown_test", "{{ 1 is nosuchtest }}"),
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "internal", "jinja", "testdata", "jinja_cases.json")
    out = {"jinja2_version": jinja2.__version__, "context": CTX, "cases": []}
    for case in CASES:
        name, src = case[0], case[1]
        opts = case[2] if len(case) > 2 else {"trim": True, "lstrip": True}
        rec = {"name": name, "template": src, "trim_blocks": opts["trim"], "lstrip_blocks": opts["lstrip"]}
        try:
            rec["expected"] = make_env(opts["trim"], opts["lstrip"]).from_string(src).render(**CTX)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        out["cases"].append(rec)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    n_err = sum("error" in c for c in out["cases"])
    print(f"wrote {len(out['cases'])} cases ({n_err} raise) to {out_path}")


if __name__ == "__main__":
    main()
