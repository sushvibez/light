#!/usr/bin/env python3
"""LoRA reference for diffusion/flux_test.go (TestFluxPipelineTinyLoRA): a
random rank-2 PEFT adapter on the tiny Flux transformer, saved in the
diffusers layout, and the image the tiny pipeline makes with it at
strength 0.7 on the same latents as flux-tiny-pipe.json.

    .venv/bin/python scripts/verify/ref_flux_lora.py diffusion/testdata/flux-tiny-pipe \\
        diffusion/testdata/flux-tiny-lora.safetensors diffusion/testdata/flux-tiny-lora.json
"""
import json, os, sys
import torch
from diffusers import FluxPipeline
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

pipe_dir, out_st, out_json = sys.argv[1:4]
base = json.load(open(os.path.join(os.path.dirname(out_json), "flux-tiny-pipe.json")))
pipe = FluxPipeline.from_pretrained(pipe_dir, torch_dtype=torch.float32)
torch.manual_seed(11)
targets = ["to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
           "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2",
           "norm1.linear", "norm1_context.linear", "norm.linear", "proj_mlp", "proj_out", "x_embedder", "context_embedder"]
pipe.transformer.add_adapter(LoraConfig(r=2, lora_alpha=3, init_lora_weights=True, target_modules=targets))
with torch.no_grad():
    for n, p in pipe.transformer.named_parameters():
        if "lora_B" in n:  # PEFT zero-inits B; make the adapter do something
            p.copy_(0.3 * torch.randn_like(p))
        elif "lora_A" in n:
            p.copy_(0.3 * torch.randn_like(p))
sd = get_peft_model_state_dict(pipe.transformer)
FluxPipeline.save_lora_weights(os.path.dirname(out_st), transformer_lora_layers=sd, safe_serialization=True,
                               weight_name=os.path.basename(out_st))
# alpha / rank = 1.5 is folded by PEFT into the forward (scaling); the file carries no alpha,
# so Light must scale by strength only, and the test tells it alpha = 3 explicitly.
strength = 0.7
latents = torch.tensor(base["latents"]).reshape(base["latents_shape"])
with torch.no_grad():
    img = pipe(base["prompt"], height=base["height"], width=base["width"], num_inference_steps=base["steps"],
               max_sequence_length=base["max_seq_len"], latents=latents, output_type="pt",
               joint_attention_kwargs={"scale": strength}).images
r = lambda a: [round(v, 7) for v in a.flatten().tolist()]
json.dump({"strength": strength, "alpha": 3, "rank": 2, "image": r(img), "image_shape": list(img.shape),
           "modules": sorted({k.rsplit(".lora_", 1)[0] for k in sd})}, open(out_json, "w"))
print("wrote", out_st, out_json, "image mean", img.mean().item(), "modules", len(sd) // 2)
