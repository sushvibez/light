#!/usr/bin/env python3
"""FP8 single-file reference for diffusion/flux_test.go
(TestFluxPipelineTinyFP8): the tiny transformer with its 2-D weights cast
to float8_e4m3fn (as the community flux1-*-fp8 checkpoints are made:
weight-only, no scales, upcast at compute), saved in the BFL single-file
layout, and the image the tiny pipeline makes when those weights are
upcast back to F32: Light's weight-only FP8 path must match it.

    .venv/bin/python scripts/verify/make_flux_fp8.py diffusion/testdata/flux-tiny-pipe \\
        diffusion/testdata/flux-tiny-fp8.safetensors diffusion/testdata/flux-tiny-fp8.json
"""
import json, os, sys
import torch
from safetensors.torch import load_file, save_file
from diffusers import FluxPipeline

sys.path.insert(0, os.path.dirname(__file__))
from make_flux_bfl import to_bfl  # noqa: E402  (runs its self-check on import)

pipe_dir, out_st, out_json = sys.argv[1:4]
base = json.load(open(os.path.join(os.path.dirname(out_json), "flux-tiny-pipe.json")))
sd = load_file(os.path.join(pipe_dir, "transformer", "diffusion_pytorch_model.safetensors"))
fp8 = {k: (v.to(torch.float8_e4m3fn) if v.dim() == 2 and k.endswith(".weight") else v) for k, v in sd.items()}
cfg = json.load(open(os.path.join(pipe_dir, "transformer", "config.json")))
bfl = to_bfl({k: v.clone() for k, v in fp8.items()}, cfg["num_attention_heads"] * cfg["attention_head_dim"])
save_file({k: v.contiguous() for k, v in bfl.items()}, out_st)
json.dump(cfg, open(out_st + ".config.json", "w"))

pipe = FluxPipeline.from_pretrained(pipe_dir, torch_dtype=torch.float32)
with torch.no_grad():
    pipe.transformer.load_state_dict({k: v.to(torch.float32) for k, v in fp8.items()})
    latents = torch.tensor(base["latents"]).reshape(base["latents_shape"])
    img = pipe(base["prompt"], height=base["height"], width=base["width"], num_inference_steps=base["steps"],
               max_sequence_length=base["max_seq_len"], latents=latents, output_type="pt").images
r = lambda a: [round(v, 7) for v in a.flatten().tolist()]
n8 = sum(v.numel() for v in fp8.values() if v.dtype == torch.float8_e4m3fn)
json.dump({"image": r(img), "image_shape": list(img.shape), "fp8_params": n8}, open(out_json, "w"))
print("wrote", out_st, out_json, "fp8 params", n8, "image mean", img.mean().item())
