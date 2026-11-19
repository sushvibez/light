#!/usr/bin/env python3
"""Per-block trace of the real FLUX.1 transformer for one forward pass on
the inputs of a ref_real_pipe.py reference (step 0: the packed noise at
sigma 1), for diffusion's TestFlux1BlocksRealCUDA: the outputs of double
blocks 0 and 18 (image and text streams), single blocks 0 and 37 and the
final velocity, F32, in a safetensors file.

    .venv/bin/python scripts/verify/ref_flux_blocks.py <ref.json> <out.safetensors>
"""
import json, sys
import torch
from diffusers import FluxTransformer2DModel
from safetensors.torch import save_file

ref_json, out = sys.argv[1:3]
ref = json.load(open(ref_json))
repo = ref["repo"]
m = FluxTransformer2DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=torch.bfloat16)
m.eval()
from accelerate import cpu_offload
cpu_offload(m, execution_device="cuda", offload_buffers=True)

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
sigma = ref["sigmas"][0]
dev, dt = "cuda", torch.bfloat16

traced = {}
def hook(name):
    def f(mod, inp, outp):
        if isinstance(outp, tuple):
            traced[name + ".txt"], traced[name + ".img"] = outp[0].float().cpu(), outp[1].float().cpu()
        else:
            traced[name] = outp.float().cpu()
    return f
m.transformer_blocks[0].register_forward_hook(hook("double0"))
m.transformer_blocks[-1].register_forward_hook(hook("double_last"))
m.single_transformer_blocks[0].register_forward_hook(hook("single0"))
m.single_transformer_blocks[-1].register_forward_hook(hook("single_last"))
with torch.no_grad():
    y = m(hidden_states=x.to(dev, dt), encoder_hidden_states=pe.to(dev, dt), pooled_projections=pooled.to(dev, dt),
          timestep=torch.tensor([sigma], device=dev, dtype=dt), img_ids=img_ids.to(dev), txt_ids=txt_ids.to(dev),
          guidance=None, return_dict=False)[0]
traced["out"] = y.float().cpu()
traced["x"] = x
save_file({k: v.contiguous() for k, v in traced.items()}, out)
print("wrote", out, {k: tuple(v.shape) for k, v in traced.items()})
