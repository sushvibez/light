#!/usr/bin/env python3
"""Writes the tiny Flux transformer in the BFL single-file layout (what
flux1-schnell.safetensors and ComfyUI checkpoints use) for
models/flux_bfl_test.go, with the inverse of diffusers'
convert_flux_transformer_checkpoint_to_diffusers.

    .venv/bin/python scripts/verify/make_flux_bfl.py \\
        diffusion/testdata/flux-tiny-pipe/transformer models/testdata/flux-tiny-bfl.safetensors

The inverse is checked first on a random 3072-wide one-block model: its
BFL form goes through diffusers' converter (which hard-codes 3072) and
must come back equal.
"""
import json, os, sys
import torch
from safetensors.torch import load_file, save_file
from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers


def to_bfl(sd, D):
    """diffusers FluxTransformer2DModel state dict -> BFL layout."""
    out = {}
    pop = sd.pop
    for a, b in [("time_text_embed.timestep_embedder.linear_1", "time_in.in_layer"),
                 ("time_text_embed.timestep_embedder.linear_2", "time_in.out_layer"),
                 ("time_text_embed.text_embedder.linear_1", "vector_in.in_layer"),
                 ("time_text_embed.text_embedder.linear_2", "vector_in.out_layer"),
                 ("time_text_embed.guidance_embedder.linear_1", "guidance_in.in_layer"),
                 ("time_text_embed.guidance_embedder.linear_2", "guidance_in.out_layer"),
                 ("context_embedder", "txt_in"), ("x_embedder", "img_in"), ("proj_out", "final_layer.linear")]:
        for s in ("weight", "bias"):
            if f"{a}.{s}" in sd:
                out[f"{b}.{s}"] = pop(f"{a}.{s}")
    for s in ("weight", "bias"):  # diffusers [scale, shift] -> BFL [shift, scale]
        scale, shift = pop(f"norm_out.linear.{s}").chunk(2, dim=0)
        out[f"final_layer.adaLN_modulation.1.{s}"] = torch.cat([shift, scale], dim=0)
    nd = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("transformer_blocks."))
    ns = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("single_transformer_blocks."))
    for i in range(nd):
        p, q = f"transformer_blocks.{i}.", f"double_blocks.{i}."
        for s in ("weight", "bias"):
            out[q + "img_mod.lin." + s] = pop(p + "norm1.linear." + s)
            out[q + "txt_mod.lin." + s] = pop(p + "norm1_context.linear." + s)
            out[q + "img_attn.qkv." + s] = torch.cat([pop(p + f"attn.to_{x}." + s) for x in "qkv"], dim=0)
            out[q + "txt_attn.qkv." + s] = torch.cat([pop(p + f"attn.add_{x}_proj." + s) for x in "qkv"], dim=0)
            out[q + "img_attn.proj." + s] = pop(p + "attn.to_out.0." + s)
            out[q + "txt_attn.proj." + s] = pop(p + "attn.to_add_out." + s)
            out[q + "img_mlp.0." + s] = pop(p + "ff.net.0.proj." + s)
            out[q + "img_mlp.2." + s] = pop(p + "ff.net.2." + s)
            out[q + "txt_mlp.0." + s] = pop(p + "ff_context.net.0.proj." + s)
            out[q + "txt_mlp.2." + s] = pop(p + "ff_context.net.2." + s)
        out[q + "img_attn.norm.query_norm.scale"] = pop(p + "attn.norm_q.weight")
        out[q + "img_attn.norm.key_norm.scale"] = pop(p + "attn.norm_k.weight")
        out[q + "txt_attn.norm.query_norm.scale"] = pop(p + "attn.norm_added_q.weight")
        out[q + "txt_attn.norm.key_norm.scale"] = pop(p + "attn.norm_added_k.weight")
    for i in range(ns):
        p, q = f"single_transformer_blocks.{i}.", f"single_blocks.{i}."
        for s in ("weight", "bias"):
            out[q + "modulation.lin." + s] = pop(p + "norm.linear." + s)
            out[q + "linear1." + s] = torch.cat([pop(p + f"attn.to_{x}." + s) for x in "qkv"] + [pop(p + "proj_mlp." + s)], dim=0)
            out[q + "linear2." + s] = pop(p + "proj_out." + s)
        out[q + "norm.query_norm.scale"] = pop(p + "attn.norm_q.weight")
        out[q + "norm.key_norm.scale"] = pop(p + "attn.norm_k.weight")
    assert not sd, f"left over: {sorted(sd)[:5]}"
    return out


def random_diffusers(D, heads, hd, in_ch, joint, pooled, guidance):
    torch.manual_seed(0)
    sd = {}
    lin = lambda name, o, i: sd.update({name + ".weight": torch.randn(o, i), name + ".bias": torch.randn(o)})
    lin("x_embedder", D, in_ch); lin("context_embedder", D, joint)
    lin("time_text_embed.timestep_embedder.linear_1", D, 256); lin("time_text_embed.timestep_embedder.linear_2", D, D)
    lin("time_text_embed.text_embedder.linear_1", D, pooled); lin("time_text_embed.text_embedder.linear_2", D, D)
    if guidance:
        lin("time_text_embed.guidance_embedder.linear_1", D, 256); lin("time_text_embed.guidance_embedder.linear_2", D, D)
    lin("norm_out.linear", 2 * D, D); lin("proj_out", in_ch, D)
    p = "transformer_blocks.0."
    lin(p + "norm1.linear", 6 * D, D); lin(p + "norm1_context.linear", 6 * D, D)
    for x in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj", "to_out.0", "to_add_out"):
        lin(p + "attn." + x, D, D)
    for n in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
        sd[p + "attn." + n + ".weight"] = torch.randn(hd)
    lin(p + "ff.net.0.proj", 4 * D, D); lin(p + "ff.net.2", D, 4 * D); lin(p + "ff_context.net.0.proj", 4 * D, D); lin(p + "ff_context.net.2", D, 4 * D)
    p = "single_transformer_blocks.0."
    lin(p + "norm.linear", 3 * D, D)
    for x in ("to_q", "to_k", "to_v"):
        lin(p + "attn." + x, D, D)
    sd[p + "attn.norm_q.weight"] = torch.randn(hd); sd[p + "attn.norm_k.weight"] = torch.randn(hd)
    lin(p + "proj_mlp", 4 * D, D); lin(p + "proj_out", D, 5 * D)
    return sd


if __name__ == "__main__":
    # 1. The inverse against diffusers' converter (3072-wide, one block each).
    ref = random_diffusers(3072, 24, 128, 64, 4096, 768, True)
    back = convert_flux_transformer_checkpoint_to_diffusers(to_bfl({k: v.clone() for k, v in ref.items()}, 3072))
    assert set(back) == set(ref), (set(back) ^ set(ref))
    for k in ref:
        assert torch.equal(back[k], ref[k]), k
    print("inverse agrees with diffusers' converter on", len(ref), "tensors")

    # 2. The tiny model, in BFL layout.
    src, dst = sys.argv[1], sys.argv[2]
    cfg = json.load(open(os.path.join(src, "config.json")))
    sd = load_file(os.path.join(src, "diffusion_pytorch_model.safetensors"))
    D = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    bfl = to_bfl(sd, D)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    save_file({k: v.contiguous() for k, v in bfl.items()}, dst)
    json.dump(cfg, open(dst + ".config.json", "w"))
    print("wrote", dst, len(bfl), "tensors")
