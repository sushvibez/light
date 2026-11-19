#!/usr/bin/env python3
"""F32 reference of the real FLUX.1 transformer's first double block (and
the embedders feeding it) on a ref_real_pipe.py reference's step-0
inputs, CPU: a full-width FluxTransformer2DModel with one double block and
no single blocks, loaded from the checkpoint's matching tensors.
Decides between Light and the BF16 diffusers trace where they disagree.

    .venv/bin/python scripts/verify/ref_flux_block0.py <ref.json> <out.safetensors>
"""
import json, os, sys
import torch
from diffusers import FluxTransformer2DModel
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

ref_json, out = sys.argv[1:3]
ref = json.load(open(ref_json))
d = snapshot_download(ref["repo"], allow_patterns=["transformer/*"])
cfg = json.load(open(os.path.join(d, "transformer", "config.json")))
cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
cfg["num_layers"], cfg["num_single_layers"] = 1, 0
m = FluxTransformer2DModel(**cfg).eval()
want = set(m.state_dict())
sd = {}
import glob
for shard in sorted(glob.glob(os.path.join(d, "transformer", "*.safetensors"))):
    with safe_open(shard, "pt") as f:
        for k in f.keys():
            if k in want:
                sd[k] = f.get_tensor(k).float()
missing = want - set(sd)
assert not missing, sorted(missing)[:5]
m.load_state_dict(sd)

B, C, h, w = ref["latents_shape"]
lat = torch.tensor(ref["latents"]).reshape(B, C, h, w)
x = lat.view(B, C, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, (h // 2) * (w // 2), C * 4)
pe = torch.tensor(ref["prompt_embeds"]).reshape(ref["prompt_embeds_shape"])
pooled = torch.tensor(ref["pooled"]).reshape(ref["pooled_shape"])
S, St = x.shape[1], pe.shape[1]
img_ids = torch.zeros(S, 3)
img_ids[:, 1] = torch.arange(h // 2).repeat_interleave(w // 2)
img_ids[:, 2] = torch.arange(w // 2).repeat(h // 2)
txt_ids = torch.zeros(St, 3)
traced = {}
def hook(mod, inp, outp):
    traced["double0.txt"], traced["double0.img"] = outp[0].float(), outp[1].float()
m.transformer_blocks[0].register_forward_hook(hook)
# Sub-steps of block 0 and what feeds it, to place a mismatch.
def grab(name, idx=None):
    def f(mod, inp, outp):
        o = outp if idx is None else outp[idx]
        traced[name] = o.float().clone()
    return f
b = m.transformer_blocks[0]
m.x_embedder.register_forward_hook(grab("x_embed"))
m.context_embedder.register_forward_hook(grab("ctx_embed"))
m.time_text_embed.register_forward_hook(grab("temb"))
b.norm1.register_forward_hook(grab("norm1", 0))           # modulated image stream
b.norm1_context.register_forward_hook(grab("norm1_ctx", 0))
b.attn.register_forward_hook(grab("attn_img", 0))         # after to_out / to_add_out
b.attn.register_forward_hook(grab("attn_txt", 1))
b.ff.register_forward_hook(grab("ff_img"))
b.ff_context.register_forward_hook(grab("ff_txt"))
b.attn.to_q.register_forward_hook(grab("q_img"))
b.attn.add_q_proj.register_forward_hook(grab("q_txt"))
b.attn.norm_q.register_forward_hook(grab("qn_img"))        # after RMSNorm, (B, H, S, Dh), before rope
b.attn.norm_k.register_forward_hook(grab("kn_img"))
b.attn.norm_added_q.register_forward_hook(grab("qn_txt"))
b.attn.norm_added_k.register_forward_hook(grab("kn_txt"))
b.attn.to_v.register_forward_hook(grab("v_img"))
b.attn.add_v_proj.register_forward_hook(grab("v_txt"))
def pre(name):
    def f(mod, inp):
        traced[name] = inp[0].float().clone()
    return f
traced["w_norm_q"] = b.attn.norm_q.weight.detach().float().clone()
b.attn.to_out[0].register_forward_pre_hook(pre("core_img"))  # attention core output, image tokens
b.attn.to_add_out.register_forward_pre_hook(pre("core_txt"))
with torch.no_grad():
    m(hidden_states=x, encoder_hidden_states=pe, pooled_projections=pooled, timestep=torch.tensor([ref["sigmas"][0]]),
      img_ids=img_ids, txt_ids=txt_ids, guidance=None, return_dict=False)
# The attention's inputs after rope and its output, as the processor makes
# them (joint sequence: text then image), in (B, H, S, Dh) for Light's trace.
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.attention_dispatch import dispatch_attention_fn
with torch.no_grad():
    rot = m.pos_embed(torch.cat([txt_ids, img_ids], dim=0))
    Hn = m.config.num_attention_heads
    query = torch.cat([traced["qn_txt"], traced["qn_img"]], dim=1)
    key = torch.cat([traced["kn_txt"], traced["kn_img"]], dim=1)
    value = torch.cat([traced["v_txt"].unflatten(-1, (Hn, -1)), traced["v_img"].unflatten(-1, (Hn, -1))], dim=1)
    qr = apply_rotary_emb(query, rot, sequence_dim=1)
    kr = apply_rotary_emb(key, rot, sequence_dim=1)
    o = dispatch_attention_fn(qr, kr, value)
    traced["q_rope"], traced["k_rope"], traced["v_in"] = qr.transpose(1, 2).contiguous(), kr.transpose(1, 2).contiguous(), value.transpose(1, 2).contiguous()
    traced["sdpa_out"] = o.transpose(1, 2).contiguous()
    traced["rope_cos"], traced["rope_sin"] = rot[0].float(), rot[1].float()
    core = o.flatten(2, 3)
    print("core check (recomputed vs hooked):", (core[:, St:] - traced["core_img"]).abs().max().item())
for k in ("v_img", "v_txt", "qn_img", "kn_img", "qn_txt", "kn_txt"):
    traced.pop(k)
save_file({k: v.contiguous() for k, v in traced.items()}, out)
print("wrote", out, {k: (tuple(v.shape), v.abs().mean().item()) for k, v in traced.items()})
