#!/usr/bin/env python3
"""Golden chat-template renders for chattemplate/hf_test.go.

    python scripts/verify/ref_chat_templates.py [out_dir] [cache_dir]

For every repo below this fetches ONLY the tokenizer files (never weights),
renders a set of conversations with transformers' `apply_chat_template`
(tokenize=False) and writes, per model, under out_dir (default
chattemplate/testdata/hf/<name>/):

    tokenizer_config.json   the repo's config minus `added_tokens_decoder`
                            (large and irrelevant to the template)
    chat_template.jinja     copied when the repo ships one
    special_tokens_map.json copied when the repo ships one
    golden.json             {repo, revision, special_tokens, cases: [...]}

Each case holds the messages, add_generation_prompt, extra kwargs (tools,
enable_thinking, ...) and either the exact `expected` string or the
`error` transformers raised. `strftime_now` is pinned to 2025-03-05
14:07:09 so templates that print today's date (Llama 3.2, SmolLM3) are
reproducible; the Go test pins the same instant.
"""
import datetime as _dt
import json
import os
import shutil
import sys

import jinja2
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import transformers.utils.chat_template_utils as ctu

MODELS = [
    ("qwen2.5", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen3", "Qwen/Qwen3-0.6B"),
    ("llama3.2", "unsloth/Llama-3.2-1B-Instruct"),  # meta-llama/* is gated
    ("phi3.5", "microsoft/Phi-3.5-mini-instruct"),
    ("phi4-mini", "microsoft/Phi-4-mini-instruct"),
    ("gemma2", "unsloth/gemma-2-2b-it"),  # google/* is gated
    ("gemma3", "unsloth/gemma-3-1b-it"),
    ("mistral-v0.3", "unsloth/mistral-7b-instruct-v0.3"),  # mistralai/* is gated
    ("smollm2", "HuggingFaceTB/SmolLM2-360M-Instruct"),
    ("smollm3", "HuggingFaceTB/SmolLM3-3B"),
    ("deepseek-r1-distill", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
]

FILES = ["tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json", "tokenizer.json"]


class FixedNow(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2025, 3, 5, 14, 7, 9)


ctu.datetime = FixedNow  # strftime_now() reads the module global at call time

SYS = {"role": "system", "content": "You are a helpful assistant."}
U1 = {"role": "user", "content": "Hello! What is 2+2?"}
A1 = {"role": "assistant", "content": "2+2 is 4."}
U2 = {"role": "user", "content": "And times 3?"}
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Paris"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]

CONVS = [
    ("user", [U1], {}),
    ("system_user", [SYS, U1], {}),
    ("multi", [U1, A1, U2], {}),
    ("system_multi", [SYS, U1, A1, U2], {}),
    ("ends_assistant", [SYS, U1, A1], {}),
    ("whitespace_unicode", [{"role": "user", "content": "  Spaces, \"quotes\", 'apostrophes',\ttabs\nand unicode: héllo — 你好 \U0001f642 <b>&amp;</b>\n\n"}], {}),
    ("think_history", [U1, {"role": "assistant", "content": "<think>\nSimple sum.\n</think>\n\n2+2 is 4."}, U2], {}),
    ("no_thinking", [SYS, U1], {"enable_thinking": False}),
    ("tools", [SYS, U1], {"tools": TOOLS}),
    ("tool_call", [
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call12345", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}}]},
        {"role": "tool", "tool_call_id": "call12345", "name": "get_weather", "content": "{\"temperature\": 22, \"sky\": \"clear\"}"},
    ], {"tools": TOOLS}),
    ("alternation_error", [U1, U2], {}),
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "chattemplate", "testdata", "hf")
    cache = sys.argv[2] if len(sys.argv) > 2 else None
    for name, repo in MODELS:
        snap = snapshot_download(repo, allow_patterns=FILES, cache_dir=cache)
        revision = os.path.basename(snap)
        tok = AutoTokenizer.from_pretrained(snap)
        out = os.path.join(out_root, name)
        os.makedirs(out, exist_ok=True)

        with open(os.path.join(snap, "tokenizer_config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        cfg.pop("added_tokens_decoder", None)
        with open(os.path.join(out, "tokenizer_config.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
            f.write("\n")
        for extra in ("chat_template.jinja", "special_tokens_map.json"):
            src = os.path.join(snap, extra)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(out, extra))

        cases = []
        for conv_name, msgs, kwargs in CONVS:
            for agp in (False, True):
                case = {"name": f"{conv_name}/agp={int(agp)}", "messages": msgs,
                        "add_generation_prompt": agp, "kwargs": kwargs}
                try:
                    case["expected"] = tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=agp, **kwargs)
                except jinja2.TemplateError as e:
                    case["error"] = f"{type(e).__name__}: {e}"
                except Exception as e:  # TypeError etc. raised inside the template
                    case["error"] = f"{type(e).__name__}: {e}"
                cases.append(case)
        golden = {"repo": repo, "revision": revision,
                  "special_tokens": tok.special_tokens_map, "cases": cases}
        with open(os.path.join(out, "golden.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(golden, f, ensure_ascii=False, indent=1)
            f.write("\n")
        n_err = sum("error" in c for c in cases)
        print(f"{name:22s} {repo:45s} {len(cases)} cases ({n_err} raise)")


if __name__ == "__main__":
    main()
