#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash real-weight check on a truncated model (Colab, high-mem CPU).

    python scripts/colab/deepseek_v41_check.py <work_dir> [layers]

Downloads only the shards holding the embeddings, the first `layers` decoder
layers (default 4: two sliding-window layers, then the first ratio-2
compressed layers with their kv / index source) and the head (about 32 GB),
plus DeepSeek's inference code; writes a truncated checkpoint folder (config
with num_hidden_layers = layers, no engram layers - layer 1's 101 GB engram
table is not fetched - and no candidate source, which sits at layer 20) and
records the reference logits of a short prompt in <work_dir>/ref.json.

The reference is DeepSeek's own inference model.py with plain-PyTorch
kernels, every weight dequantized to F32 (FP8 x UE8M0 32 x 32 blocks; FP4
e2m1 x UE8M0 1 x 32), and the activation fake-quantization as the identity -
what Light computes. The routed experts are read from the shards only when
the router picks them, so the run needs about 15 GB of RAM.
models/deepseek_v41_real_test.go (LIGHT_V41_TRUNC=<work_dir>) loads the same
folder with Light and compares.
"""
import json, os, re, shutil, sys, types

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
work = sys.argv[1]
layers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
os.makedirs(work, exist_ok=True)
# V41_SRC=<folder> (with V41_CODE=<inference code folder>) reads a local
# checkpoint instead, e.g. the tiny test fixture for a dry run.
SRC, CODE = os.environ.get("V41_SRC"), os.environ.get("V41_CODE")


def fetch(f):
    if SRC:
        if f.startswith("inference/"):
            return os.path.join(CODE, f[len("inference/"):])
        return os.path.join(SRC, f)
    return hf_hub_download(REPO, f)


def link(p, dst):
    try:
        os.symlink(os.path.abspath(p), dst)
    except OSError:  # Windows without the symlink privilege
        shutil.copy(p, dst)


if SRC and not os.path.exists(os.path.join(SRC, "model.safetensors.index.json")):
    with safe_open(os.path.join(SRC, "model.safetensors"), framework="pt") as h:
        idx = {"weight_map": {k: "model.safetensors" for k in h.keys()}}
else:
    idx = json.load(open(fetch("model.safetensors.index.json")))
wm = idx["weight_map"]
keep = {}
for k, f in wm.items():
    if k.startswith(("vision.", "aligner.", "mtp.", "image_")) or ".engram." in k:
        continue
    m = re.match(r"layers\.(\d+)\.", k)
    if m and int(m.group(1)) >= layers:
        continue
    keep[k] = f
shards = sorted(set(keep.values()))
print("shards", shards, flush=True)
for f in ["config.json", "tokenizer.json", "tokenizer_config.json"] + shards:
    p = fetch(f)
    dst = os.path.join(work, f)
    if os.path.exists(p) and not os.path.exists(dst):
        link(p, dst)
code = os.path.join(work, "inference")
os.makedirs(code, exist_ok=True)
for f in ["model.py", "engram.py", "config.json"]:
    p = fetch("inference/" + f)
    dst = os.path.join(code, f)
    if os.path.exists(p) and not os.path.exists(dst):
        link(p, dst)


def trunc(t, n):
    return [i for i in t if i < n]


cfg = json.load(open(os.path.join(work, "config.json")))
os.remove(os.path.join(work, "config.json"))
t = cfg.get("text_config", cfg)
t["num_hidden_layers"] = layers
t["compress_ratios"] = t["compress_ratios"][:layers]
t["kv_source_layer_ids"] = trunc(t["kv_source_layer_ids"], layers)
t["index_source_layer_ids"] = trunc(t["index_source_layer_ids"], layers)
if t.get("candidate_source_layer_id") is not None and t["candidate_source_layer_id"] >= layers:
    t["candidate_source_layer_id"] = None
t["engram_layer_ids"], t["engram_num_embeddings"] = [], []
t["num_nextn_predict_layers"] = 0
json.dump(cfg, open(os.path.join(work, "config.json"), "w"), indent=2)
json.dump({"metadata": idx.get("metadata", {}), "weight_map": keep},
          open(os.path.join(work, "model.safetensors.index.json"), "w"))

# --- DeepSeek's inference code with plain-PyTorch kernels ----------------------
sys.path.insert(0, code)
kernel = types.ModuleType("kernel")


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    assert inplace, "only the in-place fake quantization runs with unquantized weights"
    return x


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=None):
    assert inplace
    return x


def _quantized_gemm(*a, **k):
    raise RuntimeError("weights are unquantized here")


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, s, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    g = kv[torch.arange(b)[:, None, None], idx.clamp(min=0)]
    sc = torch.einsum("bshd,bskd->bshk", q.float(), g.float()) * softmax_scale
    sc = sc.masked_fill(~valid[:, :, None, :], float("-inf"))
    sink = attn_sink.float().view(1, 1, h, 1).expand(b, s, h, 1)
    p = torch.cat([sc, sink], -1).softmax(-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", p, g.float()).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    n = hc_mult
    pre = torch.sigmoid(mixes[..., :n] * hc_scale[0] + hc_base[:n]) + eps
    post = 2 * torch.sigmoid(mixes[..., n:2 * n] * hc_scale[1] + hc_base[n:2 * n])
    comb = (mixes[..., 2 * n:] * hc_scale[2] + hc_base[2 * n:]).unflatten(-1, (n, n))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


kernel.act_quant, kernel.fp4_act_quant = act_quant, fp4_act_quant
kernel.fp4_gemm = kernel.fp8_gemm = _quantized_gemm
kernel.sparse_attn, kernel.hc_split_sinkhorn = sparse_attn, hc_split_sinkhorn
sys.modules["kernel"] = kernel
ip = types.ModuleType("image_processor")
ip.IMAGE, ip.IMAGE_END, ip.IMAGE_NEW_LINE, ip.IMAGE_START = 0, 1, 2, 3
sys.modules["image_processor"] = ip
vis = types.ModuleType("vision")
vis.ViT = vis.Aligner = None
sys.modules["vision"] = vis
import model as M  # noqa: E402

# --- the released weights, dequantized -----------------------------------------
handles = {f: safe_open(os.path.join(work, f), framework="pt") for f in shards}


def raw(name):
    return handles[keep[name]].get_tensor(name)


def e8m0(s):
    """UE8M0 scale bytes (stored U8 or float8_e8m0fnu) as F32 powers of two."""
    b = (s if s.dtype == torch.uint8 else s.view(torch.uint8)).float()
    return torch.exp2(b - 127)


FP4 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


def weight(name):
    """<name>.weight in F32: FP8 x block scale, FP4 x row-block scale, or cast."""
    w = raw(name + ".weight")
    sn = name + ".scale"
    if sn not in keep:
        return w.float()
    s = e8m0(raw(sn))
    if w.dtype == torch.float8_e4m3fn:
        N, K = w.shape
        bn, bk = -(-N // s.shape[0]), -(-K // s.shape[1])
        full = s.repeat_interleave(bn, 0)[:N].repeat_interleave(bk, 1)[:, :K]
        return w.float() * full
    b = w.view(torch.uint8).long()  # e2m1 pairs, low nibble first
    v = torch.stack([FP4[b & 15], FP4[b >> 4]], -1).flatten(-2)
    return v * s.repeat_interleave(v.shape[1] // s.shape[1], 1)


class LazyExpert(torch.nn.Module):
    """A routed expert read from the shards each time it is picked."""

    def __init__(self, dim, inter_dim, dtype=None, swiglu_limit=0.0):
        super().__init__()
        self.swiglu_limit, self.name = swiglu_limit, None

    def forward(self, x, weights=None):
        w1, w2, w3 = (weight(f"{self.name}.{w}") for w in ("w1", "w2", "w3"))
        gate, up = F.linear(x.float(), w1), F.linear(x.float(), w3)
        if self.swiglu_limit > 0:
            up = up.clamp(-self.swiglu_limit, self.swiglu_limit)
            gate = gate.clamp(max=self.swiglu_limit)
        h = F.silu(gate) * up
        if weights is not None:
            h = weights * h
        return F.linear(h, w2).to(x.dtype)


RealExpert = M.Expert
M.Expert = lambda dim, inter, dtype=None, swiglu_limit=0.0: (
    LazyExpert(dim, inter, dtype, swiglu_limit) if dtype is not None else RealExpert(dim, inter, dtype, swiglu_limit))

a = json.load(open(os.path.join(code, "config.json")))
a.update(n_layers=layers, n_mtp_layers=0, dspark_block_size=0, dspark_target_layer_ids=(), vision_n_layers=0,
         engram_layer_ids=(), engram_num_embeddings=(), max_batch_size=1, max_seq_len=128,
         dtype="bf16", expert_dtype="fp4", temperature=0,
         compress_ratios=tuple(a["compress_ratios"][:layers]),
         kv_source_layers=tuple(trunc(a["kv_source_layers"], layers)),
         index_source_layers=tuple(trunc(a["index_source_layers"], layers)))
if a.get("candidate_source_layer", -1) >= layers:
    a["candidate_source_layer"] = -1
fields = set(M.ModelArgs.__dataclass_fields__)
args = M.ModelArgs(**{k: tuple(v) if isinstance(v, list) else v for k, v in a.items() if k in fields})
torch.set_default_dtype(torch.float32)
model = M.Transformer(args, tokenizer=object())
M.Expert = RealExpert
missing = []
with torch.no_grad():
    for i, ly in enumerate(model.layers):
        for e, ex in enumerate(ly.ffn.experts):
            ex.name = f"layers.{i}.ffn.experts.{e}"
    for mod_name, mod in list(model.named_modules()):
        if isinstance(mod, LazyExpert):
            continue
        for pn, p in list(mod._parameters.items()):
            if p is None:
                continue
            n = f"{mod_name}.{pn}" if mod_name else pn
            if pn == "scale" and not n.endswith(("hc_attn_scale", "hc_ffn_scale")):
                mod._parameters[pn] = None  # folded into the F32 weight
                continue
            if n not in keep:
                missing.append(n)
                continue
            v = weight(n[:-len(".weight")]) if pn == "weight" else raw(n).float()
            assert v.shape == p.shape, (n, v.shape, p.shape)
            mod._parameters[pn] = torch.nn.Parameter(v, requires_grad=False)
assert not missing, missing[:20]
print("weights loaded", flush=True)

from tokenizers import Tokenizer  # noqa: E402

if os.path.exists(os.path.join(work, "tokenizer.json")):
    tok = Tokenizer.from_file(os.path.join(work, "tokenizer.json"))
    ids = tok.encode("The capital of France is Paris. The capital of Germany is", add_special_tokens=False).ids
else:  # the tiny fixture: no tokenizer
    tok, ids = None, [5, 17, 99, 3, 42, 8, 120, 77, 9, 61, 33, 200]
bos = None
if tok is not None:
    bos = json.load(open(os.path.join(work, "tokenizer_config.json"))).get("bos_token")
if isinstance(bos, dict):
    bos = bos.get("content")
if bos and tok.token_to_id(bos) is not None:
    ids = [tok.token_to_id(bos)] + ids
_head = M.ParallelHead.forward
M.ParallelHead.forward = lambda self, x, full_logits=False: _head(self, x, True)
with torch.no_grad():
    _, logits, _ = model(torch.tensor([ids]))
logits = logits[0].float()
last = logits[-1]
top = last.topk(10)
print("prompt ids", ids, "top10", top.indices.tolist(), [tok.decode([i]) for i in top.indices.tolist()] if tok else "", flush=True)
json.dump({"ids": ids, "last_logits": [round(x, 5) for x in last.tolist()],
           "all_logits_l2": [round(x, 4) for x in logits.norm(dim=-1).tolist()],
           "top10": top.indices.tolist(), "layers": layers},
          open(os.path.join(work, "ref.json"), "w"))
print("wrote", os.path.join(work, "ref.json"))
