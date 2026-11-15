#!/usr/bin/env python3
"""Tiny Kimi K3.1 (kimi_k31) text-model reference for models/kimi_k31_test.go.

    python -I scripts/verify/ref_kimi_k31.py <out_dir> <out.json>

Neither the Hugging Face repo nor transformers has Kimi K3.1's modeling code;
the reference implementation is vLLM's (vllm/models/kimi_k3/nvidia/: model.py,
dsa.py, kda.py, ops/attn_res.py; vllm/model_executor/layers/mla.py and
activation.py's SituAndMul; the grouped top-k router), at vLLM commit
8657860cd37d2ba777851b5fe4e026828ded402f. This is an independent float64
PyTorch port of that forward pass (prefill, one sequence per batch row):

- Attention Residuals (attn_res_block_size): every sub-layer input is a
  softmax mix of the block snapshots and the running in-block sum, scored by
  qk_w . RMSNorm(source) and RMS-normed by the sub-layer's own norm; the model
  output mixes the blocks through output_attn_res_*;
- Kimi Delta Attention layers with K3's full-rank output gate (g_proj), the
  bounded forget gate (gate_lower_bound * sigmoid(exp(A_log) * (f + dt_bias)))
  and the separate q / k / v short convolutions;
- Gated DSA layers: NoPE MLA (the rope dims carry no rotation) whose keys are
  restricted to the indexer's top-k (sum_h relu(q_h . k) * w_h, LayerNorm'd
  index keys, no index RoPE), times a sigmoid output gate;
- a dense first layer and latent MoE layers: grouped sigmoid routing with a
  correction bias, experts run in the routed_expert_down_proj latent, the
  routed sum RMS-normed (latent_moe_use_norm) and projected back up, plus
  the shared experts on the hidden state; SiTU-GLU activations throughout.

The weights are written in the released checkpoint's naming (language_model.*)
and storage: BF16 projections, F32 KDA conv / A_log / dt_bias / o_norm and
router bias, A_log padded past num_heads as the release stores it (vLLM reads
the first num_heads), routed experts as compressed-tensors mxfp4-pack-quantized
(weight_packed: two e2m1 codes per byte, the even element in the low nibble;
weight_scale: one E8M0 exponent per 32 inputs). The reference runs the stored
(BF16-rounded, MXFP4-dequantized) values in float64. The KDA core is
cross-checked against transformers' kimi_linear chunked kernel. Records every
position's logits for a (2, 12) prompt, longer than index_topk so the DSA
top-k drops keys (every top-k boundary is checked for near-ties).
"""
import json, math, os, sys

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

out_dir, out_json = sys.argv[1], sys.argv[2]
torch.set_default_dtype(torch.float64)
SEED = int(os.environ.get("KIMI_K31_SEED", "0"))
gen = torch.Generator().manual_seed(SEED)

V, D, L = 300, 64, 7
H, NOPE, ROPE, DV, QL, KVL = 4, 16, 8, 16, 32, 24
KH, KD, CONV = 4, 16, 4  # KDA heads, head dim, short conv
NH, DH, TOPK = 8, 16, 5  # indexer heads, head dim, top-k
E, K, IM, DL, NSH, NG, TG = 8, 3, 64, 32, 2, 4, 2  # experts, top-k, expert I, latent, shared, groups
IDENSE, BS = 96, 2  # dense-layer intermediate, attn_res block size
EPS, BETA, LBETA, RSF, GLB = 1e-5, 1.0, 0.5, 2.5, -5.0
KDA_LAYERS = [1, 2, 4, 5]  # 1-based, as the release's config
DSA_LAYERS = [3, 6, 7]
A_LOG_LEN = 16  # the release stores A_log with 128 entries for 96 heads

text = {
    "architectures": ["KimiLinearForCausalLM"], "model_type": "kimi_linear",
    "vocab_size": V, "hidden_size": D, "intermediate_size": IDENSE, "moe_intermediate_size": IM,
    "num_hidden_layers": L, "num_attention_heads": H, "num_key_value_heads": H,
    "hidden_act": "situ", "activation_situ_beta": BETA, "activation_situ_linear_beta": LBETA,
    "rms_norm_eps": EPS, "tie_word_embeddings": False,
    "q_lora_rank": QL, "kv_lora_rank": KVL, "qk_nope_head_dim": NOPE, "qk_rope_head_dim": ROPE,
    "v_head_dim": DV, "mla_use_nope": True, "mla_use_output_gate": True,
    "num_experts": E, "num_experts_per_token": K, "num_shared_experts": NSH, "moe_renormalize": True,
    "moe_router_activation_func": "sigmoid", "routed_scaling_factor": RSF, "first_k_dense_replace": 1,
    "moe_layer_freq": 1, "use_grouped_topk": True, "num_expert_group": NG, "topk_group": TG,
    "topk_method": "noaux_tc", "routed_expert_hidden_size": DL, "latent_moe_use_norm": True,
    "attn_res_block_size": BS, "num_nextn_predict_layers": 0, "max_position_embeddings": 4096,
    "linear_attn_config": {"full_attn_layers": DSA_LAYERS, "kda_layers": KDA_LAYERS, "num_heads": KH,
                           "head_dim": KD, "short_conv_kernel_size": CONV, "gate_lower_bound": GLB,
                           "use_full_rank_gate": True},
    "dsa_config": {"dsa_layers": DSA_LAYERS, "index_topk": TOPK, "index_head_dim": DH, "index_num_heads": NH,
                   "index_rope_dim": 0, "index_k_norm_type": "layernorm", "index_topk_freq": 1,
                   "index_skip_topk_offset": 1},
    "quantization_config": {
        "config_groups": {"group_0": {"format": "mxfp4-pack-quantized", "input_activations": None,
                                      "output_activations": None, "targets": ["Linear"],
                                      "weights": {"group_size": 32, "num_bits": 4, "scale_dtype": "torch.uint8",
                                                  "strategy": "group", "symmetric": True, "type": "float",
                                                  "dynamic": False, "observer": "minmax"}}},
        "format": "mxfp4-pack-quantized",
        "ignore": ["re:.*self_attn.*", "re:.*shared_experts.*", "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
                   "re:.*block_sparse_moe\\.gate$", "re:.*block_sparse_moe\\.routed_expert_(up|down)_proj.*",
                   "re:.*lm_head.*", "re:.*vision_tower.*", "re:.*mm_projector.*"],
        "quant_method": "compressed-tensors", "quantization_status": "compressed"},
}

# --- weights, stored as the release stores them -------------------------------
sd, W = {}, {}  # safetensors entries; the float64 values the reference runs
P = "language_model.model."


def bf16(name, shape, std=0.1, mean=0.0):
    t = (torch.randn(shape, generator=gen) * std + mean).to(torch.bfloat16)
    sd[name] = t.contiguous()
    W[name] = t.double()
    return W[name]


def f32(name, t):
    t = t.float()
    sd[name] = t.contiguous()
    W[name] = t.double()
    return W[name]


E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])


def mxfp4(name, shape, std):
    w = torch.randn(shape, generator=gen) * std
    N, Kd = shape
    g = w.view(N, Kd // 32, 32)
    e = torch.floor(torch.log2(g.abs().amax(-1).clamp(min=1e-30))) - 2  # 6 = 1.5 * 2^2
    s = torch.pow(2.0, e)
    q = g / s[..., None]
    idx = (q.abs()[..., None] - E2M1).abs().argmin(-1)  # nearest e2m1 magnitude
    code = (idx | ((q < 0).long() << 3)).view(N, Kd)
    sd[name + ".weight_packed"] = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8).contiguous()
    sd[name + ".weight_scale"] = (e + 127).to(torch.uint8).contiguous()
    deq = (E2M1[idx] * torch.where(q < 0, -1.0, 1.0) * s[..., None]).view(N, Kd)
    W[name] = deq
    return deq


def norm_w(name, n):
    return bf16(name, (n,), 0.2, 1.0)


bf16(P + "embed_tokens.weight", (V, D), 1.0)
norm_w(P + "norm.weight", D)
norm_w(P + "output_attn_res_norm.weight", D)
bf16(P + "output_attn_res_proj.weight", (1, D), 0.15)
bf16("language_model.lm_head.weight", (V, D), 0.1)
for i in range(L):
    p = f"{P}layers.{i}."
    for n in ("input_layernorm", "post_attention_layernorm", "self_attention_res_norm", "mlp_res_norm"):
        norm_w(p + n + ".weight", D)
    bf16(p + "self_attention_res_proj.weight", (1, D), 0.15)
    bf16(p + "mlp_res_proj.weight", (1, D), 0.15)
    a = p + "self_attn."
    if i + 1 in KDA_LAYERS:
        for n in ("q", "k", "v"):
            bf16(a + n + "_proj.weight", (KH * KD, D))
            f32(a + n + "_conv1d.weight", torch.randn(KH * KD, 1, CONV, generator=gen) * 0.5)
        bf16(a + "f_a_proj.weight", (KD, D))
        bf16(a + "f_b_proj.weight", (KH * KD, KD), 0.3)
        f32(a + "dt_bias", torch.randn(KH * KD, generator=gen) * 0.5)
        f32(a + "A_log", torch.log(torch.rand(A_LOG_LEN, generator=gen) * 2 + 0.5))
        bf16(a + "b_proj.weight", (KH, D), 0.3)
        bf16(a + "g_proj.weight", (KH * KD, D), 0.2)
        f32(a + "o_norm.weight", 1 + 0.2 * torch.randn(KD, generator=gen))
        bf16(a + "o_proj.weight", (D, KH * KD))
    else:
        assert i + 1 in DSA_LAYERS
        bf16(a + "q_a_proj.weight", (QL, D))
        norm_w(a + "q_a_layernorm.weight", QL)
        bf16(a + "q_b_proj.weight", (H * (NOPE + ROPE), QL), 0.2)
        bf16(a + "kv_a_proj_with_mqa.weight", (KVL + ROPE, D), 0.2)
        norm_w(a + "kv_a_layernorm.weight", KVL)
        bf16(a + "kv_b_proj.weight", (H * (NOPE + DV), KVL), 0.2)
        bf16(a + "o_proj.weight", (D, H * DV))
        bf16(a + "g_proj.weight", (H * DV, D), 0.2)
        bf16(a + "indexer.wq_b.weight", (NH * DH, QL), 0.3)
        bf16(a + "indexer.wk.weight", (DH, D), 0.3)
        bf16(a + "indexer.weights_proj.weight", (NH, D), 0.3)
        norm_w(a + "indexer.k_norm.weight", DH)
        bf16(a + "indexer.k_norm.bias", (DH,), 0.1)
    if i == 0:
        bf16(p + "mlp.gate_proj.weight", (IDENSE, D))
        bf16(p + "mlp.up_proj.weight", (IDENSE, D))
        bf16(p + "mlp.down_proj.weight", (D, IDENSE))
    else:
        m = p + "block_sparse_moe."
        bf16(m + "gate.weight", (E, D), 0.2)
        f32(m + "gate.e_score_correction_bias", torch.randn(E, generator=gen) * 0.05)
        bf16(m + "routed_expert_down_proj.weight", (DL, D), 0.2)
        norm_w(m + "routed_expert_norm.weight", DL)
        bf16(m + "routed_expert_up_proj.weight", (D, DL), 0.1)
        bf16(m + "shared_experts.gate_proj.weight", (NSH * IM, D), 0.15)
        bf16(m + "shared_experts.up_proj.weight", (NSH * IM, D), 0.15)
        bf16(m + "shared_experts.down_proj.weight", (D, NSH * IM))
        for e in range(E):
            mxfp4(f"{m}experts.{e}.w1", (IM, DL), 0.3)
            mxfp4(f"{m}experts.{e}.w3", (IM, DL), 0.3)
            mxfp4(f"{m}experts.{e}.w2", (DL, IM), 0.15)

os.makedirs(out_dir, exist_ok=True)
save_file(sd, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
json.dump({"architectures": ["KimiK31ForConditionalGeneration"], "model_type": "kimi_k31",
           "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0, "tie_word_embeddings": False,
           "text_config": text}, open(os.path.join(out_dir, "config.json"), "w"), indent=2)


# --- forward ------------------------------------------------------------------
def rms(x, w, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def lin(x, name):
    return x @ W[name].T


def situ(g, u):  # SituAndMul.forward_native
    return BETA * torch.tanh(g / BETA) * torch.sigmoid(g) * (LBETA * torch.tanh(u / LBETA))


def mlp(x, p):
    return lin(situ(lin(x, p + "gate_proj.weight"), lin(x, p + "up_proj.weight")), p + "down_proj.weight")


def attn_res(sources, norm, proj, out_norm):  # ops/attn_res.py
    v = torch.stack(sources, dim=-2)  # (B, S, n, D)
    logits = (rms(v, W[norm]) * W[proj][0]).sum(-1)
    mixed = (torch.softmax(logits, -1)[..., None] * v).sum(-2)
    return rms(mixed, W[out_norm]) if out_norm else mixed


def l2norm(x):  # fla's l2norm
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


kda_checked = []


def kda(x, a):
    B, S, _ = x.shape
    conv_w = torch.cat([W[a + n + "_conv1d.weight"] for n in ("q", "k", "v")])  # (3HK, 1, CONV)
    mixed = torch.cat([lin(x, a + n + "_proj.weight") for n in ("q", "k", "v")], -1).transpose(1, 2)
    conv = F.silu(F.conv1d(mixed, conv_w, padding=CONV - 1, groups=conv_w.shape[0])[..., :S]).transpose(1, 2)
    q, k, v = (t.reshape(B, S, KH, KD) for t in conv.split(KH * KD, -1))
    f = (lin(lin(x, a + "f_a_proj.weight"), a + "f_b_proj.weight") + W[a + "dt_bias"]).view(B, S, KH, KD)
    A = W[a + "A_log"][:KH].exp().view(KH, 1)
    g = GLB * torch.sigmoid(A * f)  # log decay, bounded below by gate_lower_bound
    beta = torch.sigmoid(lin(x, a + "b_proj.weight"))
    qn, kn = l2norm(q) * KD ** -0.5, l2norm(k)
    state = torch.zeros(B, KH, KD, KD)
    out = torch.zeros(B, S, KH, KD)
    for t in range(S):
        state = state * g[:, t].exp()[..., None]
        delta = (v[:, t] - (state * kn[:, t, ..., None]).sum(-2)) * beta[:, t, :, None]
        state = state + kn[:, t, ..., None] * delta[..., None, :]
        out[:, t] = (state * qn[:, t, ..., None]).sum(-2)
    if not kda_checked:  # cross-check the core against transformers' kimi_linear (F32)
        from transformers.models.kimi_linear.modeling_kimi_linear import chunk_kimi_delta_attention
        ref, _ = chunk_kimi_delta_attention(q.float(), k.float(), v.float(), g.float(), beta.float(),
                                            chunk_size=8, use_qk_l2norm_in_kernel=True)
        kda_checked.append((ref.double() - out).abs().max().item())
    gate = lin(x, a + "g_proj.weight").view(B, S, KH, KD)
    o = rms(out, W[a + "o_norm.weight"]) * torch.sigmoid(gate)
    return lin(o.reshape(B, S, KH * KD), a + "o_proj.weight")


gaps, router_gaps = [], []


def dsa(x, a):
    B, S, _ = x.shape
    dk = NOPE + ROPE
    qr = rms(lin(x, a + "q_a_proj.weight"), W[a + "q_a_layernorm.weight"])
    q = lin(qr, a + "q_b_proj.weight").view(B, S, H, dk)
    kva = lin(x, a + "kv_a_proj_with_mqa.weight")
    kvc, kpe = rms(kva[..., :KVL], W[a + "kv_a_layernorm.weight"]), kva[..., KVL:]
    kv = lin(kvc, a + "kv_b_proj.weight").view(B, S, H, NOPE + DV)
    k = torch.cat([kv[..., :NOPE], kpe[:, :, None].expand(B, S, H, ROPE)], -1)  # NoPE: k_pe unrotated
    v = kv[..., NOPE:]
    # indexer: sum_h relu(q_h . k) * w_h, w = weights_proj(x) * dh^-0.5 * nh^-0.5
    iq = lin(qr, a + "indexer.wq_b.weight").view(B, S, NH, DH)
    ik = F.layer_norm(lin(x, a + "indexer.wk.weight"), (DH,), W[a + "indexer.k_norm.weight"],
                      W[a + "indexer.k_norm.bias"], EPS)
    iw = lin(x, a + "indexer.weights_proj.weight") * DH ** -0.5 * NH ** -0.5
    score = (torch.einsum("bthd,bsd->bths", iq, ik).relu() * iw[..., None]).sum(2)  # (B, S, S)
    causal = torch.ones(S, S, dtype=torch.bool).tril()
    score = score.masked_fill(~causal, float("-inf"))
    keep = torch.zeros(B, S, S, dtype=torch.bool)
    for b in range(B):
        for t in range(S):
            if t + 1 <= TOPK:
                keep[b, t, : t + 1] = True
                continue
            srt = score[b, t, : t + 1].sort(descending=True).values
            gaps.append((srt[TOPK - 1] - srt[TOPK]).item())
            keep[b, t, score[b, t].topk(TOPK).indices] = True
    logits = torch.einsum("bthd,bshd->bhts", q, k) * dk ** -0.5
    logits = logits.masked_fill(~keep[:, None], float("-inf"))
    o = torch.einsum("bhts,bshd->bthd", torch.softmax(logits, -1), v).reshape(B, S, H * DV)
    o = o * torch.sigmoid(lin(x, a + "g_proj.weight"))
    return lin(o, a + "o_proj.weight")


def moe(x, m):
    B, S, _ = x.shape
    flat = x.reshape(B * S, D)
    latent = lin(flat, m + "routed_expert_down_proj.weight")
    scores = torch.sigmoid(lin(flat, m + "gate.weight"))
    sel = scores + W[m + "gate.e_score_correction_bias"]
    gs = sel.view(-1, NG, E // NG).topk(2, -1).values.sum(-1)
    gidx = gs.topk(TG, -1).indices
    gmask = torch.zeros_like(gs).scatter(1, gidx, 1).bool()
    smask = gmask[..., None].expand(-1, NG, E // NG).reshape(-1, E)
    masked = sel.masked_fill(~smask, float("-inf"))
    ids = masked.topk(K, -1).indices
    for t, k in ((gs, TG), (masked, K)):  # router top-k boundaries
        srt = t.sort(-1, descending=True).values
        router_gaps.append((srt[:, k - 1] - srt[:, k]).min().item() if k < srt.shape[1] else 1)
    w = scores.gather(1, ids)
    w = w / w.sum(-1, keepdim=True) * RSF
    routed = torch.zeros(B * S, DL)
    for t in range(B * S):
        for j in range(K):
            e = f"{m}experts.{ids[t, j].item()}."
            h = situ(latent[t] @ W[e + "w1"].T, latent[t] @ W[e + "w3"].T)
            routed[t] += w[t, j] * (h @ W[e + "w2"].T)
    routed = rms(routed, W[m + "routed_expert_norm.weight"])
    out = lin(routed, m + "routed_expert_up_proj.weight") + mlp(flat, m + "shared_experts.")
    return out.view(B, S, D)


def forward(ids):
    prefix, delta, blocks = W[P + "embed_tokens.weight"][ids], None, []
    for i in range(L):
        p = f"{P}layers.{i}."
        if delta is not None:
            prefix = prefix + delta
        prev, write = -(-i // BS), i % BS == 0
        if write:
            assert len(blocks) == i // BS
            blocks.append(prefix)
        h = attn_res(blocks[:prev] + [prefix], p + "self_attention_res_norm.weight",
                     p + "self_attention_res_proj.weight", p + "input_layernorm.weight")
        out = kda(h, p + "self_attn.") if i + 1 in KDA_LAYERS else dsa(h, p + "self_attn.")
        prefix = out if write else prefix + out
        h = attn_res(blocks[: prev + write] + [prefix], p + "mlp_res_norm.weight", p + "mlp_res_proj.weight",
                     p + "post_attention_layernorm.weight")
        delta = mlp(h, p + "mlp.") if i == 0 else moe(h, p + "block_sparse_moe.")
    prefix = prefix + delta
    assert len(blocks) == -(-L // BS)
    h = attn_res(blocks + [prefix], P + "output_attn_res_norm.weight", P + "output_attn_res_proj.weight", None)
    return lin(rms(h, W[P + "norm.weight"]), "language_model.lm_head.weight")


ids = torch.randint(3, V, (2, 12), generator=gen)
with torch.no_grad():
    logits = forward(ids)
print("seed", SEED, "logits", tuple(logits.shape), "kda vs transformers %.2g" % kda_checked[0],
      "min top-k boundary gap %.2g" % min(gaps), "router %.2g" % min(router_gaps), "safetensors %d B" % os.path.getsize(os.path.join(out_dir, "model.safetensors")))
assert kda_checked[0] < 1e-4
assert min(gaps + router_gaps) > 2e-4, "a top-k boundary is a near-tie; reseed"
json.dump({"ids": ids.tolist(), "logits": [round(x, 6) for x in logits.flatten().tolist()], "vocab": V},
          open(out_json, "w"))
