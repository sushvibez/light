#!/usr/bin/env python3
"""Tiny LLaDA and Dream references for models/masked_diffusion_test.go, run
through each release's own code.

    python -I scripts/verify/ref_masked_diffusion.py <llada_code_dir> <dream_code_dir> <testdata_dir>

<llada_code_dir> holds GSAI-ML/LLaDA-8B-Instruct's configuration_llada.py,
modeling_llada.py and ML-GSAI/LLaDA's generate.py; <dream_code_dir> holds
Dream-org/Dream-v0-Instruct-7B's configuration_dream.py, modeling_dream.py
and generation_utils.py. Seeded tiny models are written in the released
naming (<testdata_dir>/llada-tiny, dream-tiny) with every position's logits
for a prompt plus masks, and deterministic generations: LLaDA's
low-confidence remasking (2 blocks, with and without classifier-free
guidance) and Dream's maskgit_plus / topk_margin / entropy at temperature 0.
"""
import importlib.util, json, os, sys

import torch
from safetensors.torch import save_file

llada_dir, dream_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]


def load(name, path, package=None):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def seed_weights(model, gen):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("norm.weight") or "_norm" in n or "ln_f" in n:
                p.copy_(1 + 0.2 * torch.randn(p.shape, generator=gen))
            elif "wte" in n or "embed_tokens" in n:
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
            else:
                p.copy_(torch.randn(p.shape, generator=gen) * 0.15)


# --- LLaDA ------------------------------------------------------------------
pkg = "llada_ref"
sys.modules[pkg] = type(sys)(pkg)
sys.modules[pkg].__path__ = [llada_dir]
lc = load(pkg + ".configuration_llada", os.path.join(llada_dir, "configuration_llada.py"))
lm = load(pkg + ".modeling_llada", os.path.join(llada_dir, "modeling_llada.py"))
lg = load("llada_generate", os.path.join(llada_dir, "generate.py"))
V, MASK = 160, 150
cfg = lc.LLaDAConfig(d_model=64, n_heads=4, n_kv_heads=4, n_layers=3, mlp_hidden_size=96, mlp_ratio=4,
                     vocab_size=V, embedding_size=V, max_sequence_length=256, rope=True, rope_theta=500000.0,
                     rope_full_precision=True, block_type="llama", layer_norm_type="rms", rms_norm_eps=1e-5,
                     activation_type="silu", include_bias=False, include_qkv_bias=False, weight_tying=False,
                     alibi=False, flash_attention=False, attention_layer_norm=False, input_emb_norm=False,
                     scale_logits=False, init_device="cpu", mask_token_id=MASK, eos_token_id=1, pad_token_id=1,
                     bias_for_layer_norm=False, layer_norm_with_affine=True, block_group_size=1)
cfg.use_cache = False
model = lm.LLaDAModelLM(cfg, init_params=True).float().eval()
seed_weights(model, torch.Generator().manual_seed(3))
d = os.path.join(out, "llada-tiny")
os.makedirs(d, exist_ok=True)
save_file({k: v.contiguous() for k, v in model.state_dict().items()}, os.path.join(d, "model.safetensors"))
c = cfg.to_dict()
c.update(model_type="llada", architectures=["LLaDAModelLM"])
json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=2, default=str)
prompt = torch.tensor([[0, 17, 99, 3, 42, 8, 120, 77]])
x = torch.cat([prompt, torch.full((1, 6), MASK)], 1)
x[0, 10] = 33
with torch.no_grad():
    fwd = model(x).logits[0]
gens = {}
for name, kw in {"greedy": dict(cfg_scale=0.0), "cfg": dict(cfg_scale=1.5)}.items():
    with torch.no_grad():
        # an all-ones mask: upstream's cfg path needs one (attention_mask_)
        y = lg.generate(model, prompt, attention_mask=torch.ones_like(prompt), steps=8, gen_length=16,
                        block_length=8, temperature=0.0,
                        remasking="low_confidence", mask_id=MASK, **kw)
    gens[name] = y[0, prompt.shape[1]:].tolist()
    print("llada", name, gens[name])
json.dump({"input": x[0].tolist(), "logits": [round(v, 5) for v in fwd.flatten().tolist()], "vocab": V,
           "mask_id": MASK, "prompt": prompt[0].tolist(), "steps": 8, "gen_length": 16, "block_length": 8,
           "cfg_scale": 1.5, "gens": gens}, open(os.path.join(out, "llada-tiny.json"), "w"))

# --- Dream ------------------------------------------------------------------
pkg = "dream_ref"
sys.modules[pkg] = type(sys)(pkg)
sys.modules[pkg].__path__ = [dream_dir]
from transformers import modeling_rope_utils as _rope  # noqa: E402

if "default" not in _rope.ROPE_INIT_FUNCTIONS:  # transformers 5 dropped it; Dream's code (4.46) asks for it
    def _default_rope(config=None, device=None, seq_len=None, **kw):
        base = kw.get("base", getattr(config, "rope_theta", 10000.0))
        dim = kw.get("dim") or getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        return inv, 1.0
    _rope.ROPE_INIT_FUNCTIONS["default"] = _default_rope
dc = load(pkg + ".configuration_dream", os.path.join(dream_dir, "configuration_dream.py"))
gu = load(pkg + ".generation_utils", os.path.join(dream_dir, "generation_utils.py"))
dm = load(pkg + ".modeling_dream", os.path.join(dream_dir, "modeling_dream.py"))
V, MASK = 160, 151
cfg = dc.DreamConfig(vocab_size=V, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                     num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                     rope_theta=1000000.0, rms_norm_eps=1e-6, tie_word_embeddings=False, mask_token_id=MASK,
                     bos_token_id=0, eos_token_id=1, pad_token_id=1, attn_implementation="sdpa")
cfg._attn_implementation = "sdpa"
model = dm.DreamModel(cfg).float().eval()
seed_weights(model, torch.Generator().manual_seed(4))
d = os.path.join(out, "dream-tiny")
os.makedirs(d, exist_ok=True)
save_file({k: v.contiguous() for k, v in model.state_dict().items()}, os.path.join(d, "model.safetensors"))
c = cfg.to_dict()
c.update(model_type="Dream", architectures=["DreamModel"])
json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=2, default=str)
prompt = torch.tensor([[0, 17, 99, 3, 42, 8, 120, 77]])
x = torch.cat([prompt, torch.full((1, 6), MASK)], 1)
x[0, 10] = 33
with torch.no_grad():
    fwd = model(x, "full").logits[0]
gens = {}


def dream_sample(model, input_ids, max_new_tokens, steps, alg, eps=1e-3):
    """DreamGenerationMixin._sample (generation_utils.py) for temperature 0,
    no top-p / top-k, alg_temp 0 and no attention mask, transcribed because
    its generation-config plumbing predates transformers 5; sample_tokens is
    the release's own."""
    x = torch.nn.functional.pad(input_ids, (0, max_new_tokens), value=MASK)
    timesteps = torch.linspace(1, eps, steps + 1)
    for i in range(steps):
        mask_index = x == MASK
        logits = model(x, "full", None).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        mask_logits = logits[mask_index]
        t, s_ = timesteps[i], timesteps[i + 1]
        confidence, x0 = gu.sample_tokens(mask_logits, temperature=0.0, top_p=None, top_k=None,
                                          margin_confidence=alg == "topk_margin", neg_entropy=alg == "entropy")
        num_mask_token = mask_index.sum() / mask_index.shape[0]
        n = int(num_mask_token * (1 - s_ / t)) if i < steps - 1 else int(num_mask_token)
        full_confidence = torch.full_like(x, -torch.inf, dtype=logits.dtype)
        full_confidence[mask_index] = confidence
        if n > 0:
            _, transfer_index = torch.topk(full_confidence, n)
            x_ = torch.zeros_like(x) + MASK
            x_[mask_index] = x0.clone()
            rows = torch.arange(x.size(0)).unsqueeze(1).expand_as(transfer_index)
            x[rows, transfer_index] = x_[rows, transfer_index]
    return x


for alg in ("maskgit_plus", "topk_margin", "entropy"):
    with torch.no_grad():
        y = dream_sample(model, prompt.clone(), 12, 6, alg)
    gens[alg] = y[0, prompt.shape[1]:].tolist()
    print("dream", alg, gens[alg])
json.dump({"input": x[0].tolist(), "logits": [round(v, 5) for v in fwd.flatten().tolist()], "vocab": V,
           "mask_id": MASK, "prompt": prompt[0].tolist(), "steps": 6, "max_new_tokens": 12, "eps": 1e-3,
           "gens": gens}, open(os.path.join(out, "dream-tiny.json"), "w"))
print("wrote", out)
