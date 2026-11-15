#!/usr/bin/env python3
"""LLaDA-8B-Instruct / Dream-v0-Instruct-7B real-weight check (Colab CPU, F32).

    python scripts/colab/masked_diffusion_check.py llada|dream <work_dir> [gen_length]

Downloads the release into <work_dir> (LLaDA ~16 GB, Dream ~15 GB), loads it
with its own modeling code in F32, and records in <work_dir>/ref.json: a
chat prompt's ids, the logits of the first masked positions of prompt +
gen_length masks, the argmax at every position, and a deterministic
generation (LLaDA: ML-GSAI/LLaDA generate(), one block, one token per
step, greedy low-confidence remasking; Dream: _sample with the entropy
order at temperature 0, one token per step).
models/masked_diffusion_real_test.go (LIGHT_MASKED_DIR=<work_dir>,
LIGHT_MASKED_KIND=llada|dream) compares Light.

V_SRC=<folder> V_CODE=<code folder> skip the download (a dry run on the
tiny fixtures, which have no tokenizer: a fixed id prompt is used).
"""
import importlib.util, json, os, sys, types, urllib.request

import torch

kind, work = sys.argv[1], sys.argv[2]
G = int(sys.argv[3]) if len(sys.argv) > 3 else 32
REPO = {"llada": "GSAI-ML/LLaDA-8B-Instruct", "dream": "Dream-org/Dream-v0-Instruct-7B"}[kind]
SRC, CODE = os.environ.get("V_SRC"), os.environ.get("V_CODE")
if SRC:
    work, code = SRC, CODE
else:
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, local_dir=work)
    code = work
    if kind == "llada":
        urllib.request.urlretrieve("https://raw.githubusercontent.com/ML-GSAI/LLaDA/main/generate.py",
                                   os.path.join(work, "generate.py"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pkg = kind + "_ref"
sys.modules[pkg] = types.ModuleType(pkg)
sys.modules[pkg].__path__ = [code]
cfgd = json.load(open(os.path.join(work, "config.json")))
MASK = cfgd["mask_token_id"]
if kind == "llada":
    lc = load(pkg + ".configuration_llada", os.path.join(code, "configuration_llada.py"))
    lm = load(pkg + ".modeling_llada", os.path.join(code, "modeling_llada.py"))
    lg = load("llada_generate", os.path.join(code, "generate.py"))
    keep = {k: v for k, v in cfgd.items() if k not in ("architectures", "auto_map", "model_type", "transformers_version")}
    cfg = lc.LLaDAConfig(**keep)
    cfg.use_cache = False
    cfg.init_device = "cpu"
    model = lm.LLaDAModelLM(cfg, init_params=False)
else:
    from transformers import modeling_rope_utils as _rope
    if "default" not in _rope.ROPE_INIT_FUNCTIONS:
        def _default_rope(config=None, device=None, seq_len=None, **kw):
            base = kw.get("base", getattr(config, "rope_theta", 10000.0))
            dim = kw.get("dim") or config.hidden_size // config.num_attention_heads
            return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)), 1.0
        _rope.ROPE_INIT_FUNCTIONS["default"] = _default_rope
    dc = load(pkg + ".configuration_dream", os.path.join(code, "configuration_dream.py"))
    gu = load(pkg + ".generation_utils", os.path.join(code, "generation_utils.py"))
    dm = load(pkg + ".modeling_dream", os.path.join(code, "modeling_dream.py"))
    keep = {k: v for k, v in cfgd.items() if k not in ("architectures", "auto_map", "model_type", "transformers_version", "_name_or_path")}
    cfg = dc.DreamConfig(**keep)
    cfg._attn_implementation = "sdpa"
    model = dm.DreamModel(cfg)
from safetensors.torch import load_file  # noqa: E402

model = model.float().eval()
params = dict(model.named_parameters())
seen = set()
with torch.no_grad():  # one shard at a time, copied into the F32 parameters (8B: 32 GB, not 64)
    for f in sorted(os.listdir(work)):
        if f.endswith(".safetensors"):
            for k, v in load_file(os.path.join(work, f)).items():
                if k in params:
                    params[k].copy_(v.float())
                    seen.add(k)
missing = [k for k in params if k not in seen]
assert not missing, missing[:10]
print("loaded", flush=True)

if os.path.exists(os.path.join(work, "tokenizer_config.json")):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(work, trust_remote_code=True)
    msgs = [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if isinstance(ids, dict):
        ids = ids["input_ids"]
else:
    tok, ids = None, [0, 17, 99, 3, 42, 8, 120, 77]
prompt = torch.tensor([ids])
x = torch.cat([prompt, torch.full((1, G), MASK)], 1)
with torch.no_grad():
    lo = model(x).logits[0] if kind == "llada" else model(x, "full").logits[0]
first = [lo[len(ids) + i].tolist() for i in range(3)] if kind == "llada" else [lo[len(ids) - 1 + i].tolist() for i in range(3)]
argmax = lo.argmax(-1).tolist()
print("forward done", flush=True)


def dream_sample(input_ids, steps, eps=1e-3):  # Dream's _sample, entropy, temperature 0 (see ref_masked_diffusion.py)
    x = torch.nn.functional.pad(input_ids, (0, G), value=MASK)
    ts = torch.linspace(1, eps, steps + 1)
    for i in range(steps):
        mi = x == MASK
        logits = model(x, "full", None).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        conf, x0 = gu.sample_tokens(logits[mi], temperature=0.0, top_p=None, top_k=None, neg_entropy=True)
        nm = mi.sum() / mi.shape[0]
        n = int(nm * (1 - ts[i + 1] / ts[i])) if i < steps - 1 else int(nm)
        full = torch.full_like(x, -torch.inf, dtype=logits.dtype)
        full[mi] = conf
        if n > 0:
            _, ti = torch.topk(full, n)
            x_ = torch.zeros_like(x) + MASK
            x_[mi] = x0.clone()
            rows = torch.arange(x.size(0)).unsqueeze(1).expand_as(ti)
            x[rows, ti] = x_[rows, ti]
    return x


with torch.no_grad():
    if kind == "llada":
        y = lg.generate(model, prompt, attention_mask=torch.ones_like(prompt), steps=G, gen_length=G,
                        block_length=G, temperature=0.0, remasking="low_confidence", mask_id=MASK)
    else:
        y = dream_sample(prompt.clone(), G)
gen = y[0, len(ids):].tolist()
text = tok.decode(gen, skip_special_tokens=True) if tok else ""
print("prompt", ids, "gen", gen, repr(text), flush=True)
out = os.environ.get("V_OUT", os.path.join(work, "ref.json"))
json.dump({"kind": kind, "ids": ids, "mask_id": MASK, "gen_length": G, "first_logits": first,
           "argmax": argmax, "gen": gen, "text": text}, open(out, "w"))
print("wrote", out)
