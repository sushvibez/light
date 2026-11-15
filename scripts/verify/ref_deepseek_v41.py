#!/usr/bin/env python3
"""Tiny DeepSeek-V4.1 (deepseek_v41) text-model reference for
models/deepseek_v41_test.go, run through DeepSeek's own inference code.

    python -I scripts/verify/ref_deepseek_v41.py <official_dir> <out_dir> <out.json>

<official_dir> holds the official inference code (model.py, engram.py; the
tilelang kernel.py is replaced here by plain-PyTorch equivalents, and the
vision / image_processor modules by stubs). A seeded tiny Transformer with
every text feature on - engram n-gram layers, sliding-window attention,
compressors at ratios 2 and 1 shared across layers, index-key owners and
plain indexers, the two-level candidate-block pre-filter, 4-stream
hyper-connections, sqrtsoftplus routing with a shared expert, the clamped
SwiGLU, the grouped low-rank output projection and attention sinks - has its
weights written in the released checkpoint's naming and storage: FP8 E4M3
projections with 32 x 32 E8M0 block scales, FP4 (e2m1, packed in I8) routed
experts with 1 x 32 E8M0 scales, the FP8 engram table with per-row E8M0
scales, BF16 / F32 elsewhere. The reference runs the dequantized weights
in F32 with the activation fake-quantization (act_quant / fp4_act_quant in
place) as the identity, which is what Light computes. Records the logits of
every position for a (2, 20) prompt, long enough for the window, the
compressed top-k and the candidate blocks to drop keys.
"""
import json, math, os, struct, sys, types

import numpy as np
import torch
import torch.nn.functional as F

official, out_dir, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, official)

# --- kernel.py in plain PyTorch -----------------------------------------------
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
    """sparse_attn_kernel: softmax over the gathered keys plus a per-head sink
    logit; the value is the key (one latent KV)."""
    b, s, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    g = kv[torch.arange(b)[:, None, None], idx.clamp(min=0)]  # b, s, k, d
    sc = torch.einsum("bshd,bskd->bshk", q.float(), g.float()) * softmax_scale
    sc = sc.masked_fill(~valid[:, :, None, :], float("-inf"))
    sink = attn_sink.float().view(1, 1, h, 1).expand(b, s, h, 1)
    p = torch.cat([sc, sink], -1).softmax(-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", p, g.float()).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    n = hc_mult
    pre = torch.sigmoid(mixes[..., :n] * hc_scale[0] + hc_base[:n]) + eps
    post = 2 * torch.sigmoid(mixes[..., n:2 * n] * hc_scale[1] + hc_base[n:2 * n])
    comb = mixes[..., 2 * n:] * hc_scale[2] + hc_base[2 * n:]
    comb = comb.unflatten(-1, (n, n))
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

import engram  # noqa: E402

VOCAB, COMP_VOCAB = 256, 200
gen = torch.Generator().manual_seed(1234)
TOKEN_MAP = torch.randint(0, COMP_VOCAB, (VOCAB,), generator=gen).tolist()
TOKEN_MAP[:COMP_VOCAB] = torch.randperm(COMP_VOCAB, generator=gen).tolist()  # every compressed id used
engram.build_compressed_token_map = lambda tokenizer: (TOKEN_MAP, COMP_VOCAB)

import model as M  # noqa: E402

# --- config ---------------------------------------------------------------------
L = 8
ratios = [0, 2, 2, 2, 1, 1, 1, 0]
eng_layers, eng_ngram, eng_heads, eng_hd, eng_vocab = [1, 5], 3, 2, 64, 50
# the bucket primes (EngramLayout): consecutive primes from engram_vocab_size,
# so each layer's table holds exactly the sum of its primes
lay = engram.EngramLayout.from_args(types.SimpleNamespace(
    engram_layer_ids=eng_layers, engram_max_ngram_size=eng_ngram, engram_n_heads=eng_heads,
    engram_vocab_size=eng_vocab, engram_num_embeddings=(), engram_head_dim=eng_hd))
eng_rows = [sum(p for ng in layer for p in ng) for layer in lay.primes]

args = M.ModelArgs(
    max_batch_size=2, max_seq_len=32, temperature=0, dtype="bf16", expert_dtype=None, vocab_size=VOCAB,
    dim=64, moe_inter_dim=32, n_layers=L, n_mtp_layers=0, n_heads=4, n_routed_experts=8, n_shared_experts=1,
    n_activated_experts=3, score_func="sqrtsoftplus", route_scale=1.5, swiglu_limit=1.0, q_lora_rank=32,
    head_dim=32, rope_head_dim=16, norm_eps=1e-20, o_groups=2, o_lora_rank=32, window_size=4,
    compress_ratios=tuple(ratios), kv_source_layers=(1, 4), index_source_layers=(1, 3, 4, 6),
    compress_rope_theta=160000.0, original_seq_len=64, rope_theta=10000.0, rope_factor=4, beta_fast=32,
    beta_slow=1, index_n_heads=8, index_head_dim=32, index_topk=4, candidate_source_layer=4,
    candidate_topk_blocks=3, candidate_block_size=2, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
    engram_layer_ids=tuple(eng_layers), engram_num_embeddings=tuple(eng_rows), engram_max_ngram_size=eng_ngram,
    engram_vocab_size=eng_vocab, engram_n_heads=eng_heads, engram_head_dim=eng_hd, engram_pad_id=2,
    engram_compressed_vocab_size=COMP_VOCAB, vision_n_layers=0)

SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 2  # a seed whose top-k boundaries are clear
torch.manual_seed(SEED)
model = M.Transformer(args, tokenizer=object())
# the engram rows come out in BF16 (exactly: FP8 times a power of two); the
# F32 wkv here takes them at F32
_eng = M.ParallelEngramEmbedding.forward
M.ParallelEngramEmbedding.forward = lambda self, indices: _eng(self, indices).float()

# --- weights, quantized the way the release stores them ------------------------
E4M3 = torch.float8_e4m3fn
FP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def pow2_scale(amax, qmax):
    """ue8m0 scale: the power of two at or above amax / qmax."""
    return torch.exp2(torch.ceil(torch.log2((amax / qmax).clamp(min=2.0 ** -100))))


def e8m0(s):
    return (torch.log2(s).round().to(torch.int32) + 127).to(torch.uint8)


def quant_fp8_block(w, bs=32):
    N, K = w.shape
    nb, kb = (N + bs - 1) // bs, (K + bs - 1) // bs
    sc = torch.zeros(nb, kb)
    for i in range(nb):
        for j in range(kb):
            sc[i, j] = pow2_scale(w[i * bs:(i + 1) * bs, j * bs:(j + 1) * bs].abs().max(), 448.0)
    full = sc.repeat_interleave(bs, 0)[:N].repeat_interleave(bs, 1)[:, :K]
    q = (w / full).to(E4M3)
    return q, e8m0(sc), q.float() * full


def quant_fp8_rows(w, bs=32):  # the engram table: one scale per 32 values of a row
    N, K = w.shape
    sc = pow2_scale(w.view(N, K // bs, bs).abs().amax(-1), 448.0)
    full = sc.repeat_interleave(bs, 1)
    q = (w / full).to(E4M3)
    return q, e8m0(sc), q.float() * full


def quant_fp4(w, bs=32):  # e2m1, 1 x 32 blocks along K, low nibble first
    N, K = w.shape
    sc = pow2_scale(w.view(N, K // bs, bs).abs().amax(-1), 6.0)
    x = (w / sc.repeat_interleave(bs, 1)).clamp(-6, 6)
    code = (x.abs().unsqueeze(-1) - FP4_LUT).abs().argmin(-1)
    nib = (code | ((x < 0) & (code > 0)).long() << 3).to(torch.uint8)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).view(torch.int8)
    deq = torch.where(x < 0, -1.0, 1.0) * FP4_LUT[code] * sc.repeat_interleave(bs, 1)
    return packed, e8m0(sc), deq


def is_fp8(n):
    return (n.endswith(("attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight", "attn.wo_a.weight",
                        "attn.wo_b.weight", "indexer.wq_b.weight", "engram.wkv.weight"))
            or ".shared_experts.w" in n)


released = {}
with torch.no_grad():
    for mod_name, mod in list(model.named_modules()):
        for pn, p in list(mod._parameters.items()):
            if p is None:
                continue
            n = f"{mod_name}.{pn}" if mod_name else pn
            if pn == "scale" and isinstance(mod, M.ParallelEngramEmbedding):
                # the table holds the dequantized rows here: unit scales
                mod._parameters[pn] = torch.nn.Parameter(torch.ones(p.shape), requires_grad=False)
                continue
            if pn == "scale" and not n.endswith(("hc_attn_scale", "hc_ffn_scale")):
                mod._parameters[pn] = None  # unquantized here; the released scale is written below
                continue
            shape = p.shape
            if n.endswith("engram.embed.weight"):
                v = torch.randn(shape) * 0.5
            elif n.endswith(("norm.weight", "q_weight", "k_weight")):
                v = 1 + 0.2 * torch.randn(shape)
            elif n.endswith("attn_sink"):
                v = torch.randn(shape) * 0.5
            elif n.endswith("gate.bias"):
                v = torch.randn(shape) * 0.1
            elif n.endswith(("hc_attn_scale", "hc_ffn_scale")):
                v = torch.rand(shape) + 0.5
            elif n.endswith(("hc_attn_base", "hc_ffn_base")):
                v = torch.randn(shape) * 0.5
            elif n == "embed.weight":
                v = torch.randn(shape) * 0.5
            else:
                v = torch.randn(shape) * 0.1
            if n.endswith("engram.embed.weight"):
                q, s, v = quant_fp8_rows(v)
                released[n], released[n[:-len("weight")] + "scale"] = q, s.view(torch.float8_e8m0fnu)
            elif ".experts." in n and ".shared_experts." not in n:
                q, s, v = quant_fp4(v)
                released[n], released[n[:-len("weight")] + "scale"] = q, s.view(torch.float8_e8m0fnu)
            elif is_fp8(n):
                q, s, v = quant_fp8_block(v)
                released[n], released[n[:-len("weight")] + "scale"] = q, s.view(torch.float8_e8m0fnu)
            elif n.endswith(("attn_sink", "gate.bias")) or ".hc_" in n:
                released[n] = v.clone()
            else:
                v = v.bfloat16().float()
                released[n] = v.bfloat16()
            mod._parameters[pn] = torch.nn.Parameter(v.float(), requires_grad=False)
    for i in range(L):  # vision routing bias, unused by text (and by Light)
        released[f"layers.{i}.ffn.gate.bias_vl"] = torch.randn(args.n_routed_experts) * 0.1

os.makedirs(out_dir, exist_ok=True)
from safetensors.torch import save_file  # noqa: E402

save_file({k: v.contiguous() for k, v in released.items()}, os.path.join(out_dir, "model.safetensors"),
          metadata={"format": "pt"})
cfg = {
    "architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41", "dtype": "bfloat16",
    "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2, "image_token_id": 250,
    "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [32, 32],
                            "scale_fmt": "ue8m0", "expert_dtype": "fp4"},
    "text_config": {
        "model_type": "deepseek_v41_text", "vocab_size": VOCAB, "hidden_size": args.dim,
        "moe_intermediate_size": args.moe_inter_dim, "num_hidden_layers": L, "num_attention_heads": args.n_heads,
        "num_key_value_heads": 1, "head_dim": args.head_dim, "qk_rope_head_dim": args.rope_head_dim,
        "q_lora_rank": args.q_lora_rank, "o_lora_rank": args.o_lora_rank, "o_groups": args.o_groups,
        "hidden_act": "silu", "swiglu_limit": args.swiglu_limit, "rms_norm_eps": args.norm_eps,
        "tie_word_embeddings": False, "max_position_embeddings": 1024, "rope_theta": args.rope_theta,
        "rope_scaling": {"rope_type": "yarn", "factor": args.rope_factor, "beta_fast": args.beta_fast,
                         "beta_slow": args.beta_slow, "original_max_position_embeddings": args.original_seq_len},
        "n_routed_experts": args.n_routed_experts, "n_shared_experts": 1,
        "num_experts_per_tok": args.n_activated_experts, "scoring_func": args.score_func, "topk_method": "noaux_tc",
        "norm_topk_prob": True, "routed_scaling_factor": args.route_scale, "sliding_window": args.window_size,
        "compress_ratios": ratios, "compress_rope_theta": args.compress_rope_theta,
        "kv_source_layer_ids": list(args.kv_source_layers), "index_source_layer_ids": list(args.index_source_layers),
        "index_n_heads": args.index_n_heads, "index_head_dim": args.index_head_dim, "index_topk": args.index_topk,
        "candidate_source_layer_id": args.candidate_source_layer, "candidate_topk_blocks": args.candidate_topk_blocks,
        "candidate_block_size": args.candidate_block_size, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
        "engram_layer_ids": eng_layers, "engram_num_embeddings": eng_rows, "engram_max_ngram_size": eng_ngram,
        "engram_vocab_size": eng_vocab, "engram_n_heads": eng_heads, "engram_head_dim": eng_hd,
        "engram_pad_token_id": 2, "engram_compressed_vocab_size": COMP_VOCAB, "num_nextn_predict_layers": 0,
    },
}
json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
json.dump({"token_map": TOKEN_MAP, "compressed_vocab_size": COMP_VOCAB},
          open(os.path.join(out_dir, "engram_token_map.json"), "w"))

# --- forward, with every top-k boundary checked for near-ties --------------------
gaps = []
_topk = torch.Tensor.topk


def topk_checked(self, k, dim=-1, largest=True, sorted=True):
    v = self.float().sort(dim=dim, descending=True).values
    if k < v.size(dim):
        a, b = v.narrow(dim, k - 1, 1), v.narrow(dim, k, 1)
        ok = torch.isfinite(a) & torch.isfinite(b)
        if ok.any():
            gaps.append((a - b)[ok].min().item())
    return _topk(self, k, dim=dim, largest=largest, sorted=sorted)


torch.Tensor.topk = topk_checked
_head = M.ParallelHead.forward
M.ParallelHead.forward = lambda self, x, full_logits=False: _head(self, x, True)
ids = torch.randint(3, VOCAB, (2, 20), generator=gen)
ids[1, 5:9] = ids[0, 2:6]  # a repeated n-gram
with torch.no_grad():
    _, logits, _ = model(ids)
torch.Tensor.topk = _topk
print("seed", SEED, "logits", tuple(logits.shape), "top-k boundary gaps", ["%.2g" % g for g in gaps])
assert min(gaps) > 2e-4, "a top-k boundary is a near-tie; reseed"
json.dump({"ids": ids.tolist(), "logits": [round(x, 6) for x in logits.flatten().tolist()], "vocab": VOCAB},
          open(out_json, "w"))
