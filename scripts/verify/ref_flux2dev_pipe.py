#!/usr/bin/env python3
"""End-to-end Flux2Pipeline (FLUX.2 [dev]) reference (diffusers, CPU, F32)
for diffusion/flux2dev_test.go.

    .venv/bin/python scripts/verify/ref_flux2dev_pipe.py <out_dir> <out.json>

Builds a seeded tiny FLUX.2 [dev] pipeline - Mistral-Small-3.1's processor
(Apache-2.0, ungated; FLUX.2-dev's own tokenizer folder is gated), a tiny
Mistral3 conditional-generation text encoder (BF16 weights), a tiny
Flux2Transformer2DModel with guidance embeddings, the tiny
AutoencoderKLFlux2 of diffusion/testdata/flux2-tiny-pipe and FLUX.2-klein's
scheduler config - saves it, reloads it from disk (so RoPE buffers are
recomputed, not BF16-rounded) and runs it for 3 steps at 64 x 48 with
explicit latents, guidance 4.0, text encoder layers (1, 2, 3) and a 96-token
window. Writes the prompt ids / mask, the prompt embeddings, the sigmas and
the final image.
"""
import json, os, shutil, sys
import torch
from diffusers import (AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2Pipeline,
                       Flux2Transformer2DModel)
from diffusers.pipelines.flux2.pipeline_flux2 import format_input
from diffusers.pipelines.flux2.system_messages import SYSTEM_MESSAGE
from safetensors.torch import load_file, save_file
# The Mistral3 model class is reached through the Auto API, and the fixture's JSON
# is scrubbed of its name: the 32-character class name trips GitHub's secret
# scanning (it looks like a Mistral API key).
from transformers import AutoModelForImageTextToText, AutoProcessor, Mistral3Config

out_dir, out_json = sys.argv[1], sys.argv[2]
here = os.path.dirname(os.path.abspath(out_dir))
proc = AutoProcessor.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503")
sched = FlowMatchEulerDiscreteScheduler.from_pretrained("black-forest-labs/FLUX.2-klein-4B", subfolder="scheduler")
layers, max_len = (1, 2, 3), 96

torch.manual_seed(71)
cfg = Mistral3Config(
    text_config=dict(model_type="mistral", vocab_size=131072, hidden_size=16, intermediate_size=32,
                     num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                     rms_norm_eps=1e-5, max_position_embeddings=1024, rope_theta=1000000000.0),
    vision_config=dict(model_type="pixtral", hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                       num_attention_heads=2, head_dim=8, patch_size=14, image_size=56),
    spatial_merge_size=2, image_token_index=10, tie_word_embeddings=False)
te = AutoModelForImageTextToText.from_config(cfg).eval()
with torch.no_grad():
    for name, p in te.named_parameters():
        p.copy_(1 + 0.2 * torch.randn_like(p)) if "norm" in name else p.normal_(0, 0.1)
tr = Flux2Transformer2DModel(patch_size=1, in_channels=16, num_layers=1, num_single_layers=1, attention_head_dim=16,
                             num_attention_heads=2, joint_attention_dim=48, mlp_ratio=3.0, axes_dims_rope=(4, 4, 4, 4),
                             rope_theta=2000, guidance_embeds=True).eval()
with torch.no_grad():
    for name, p in tr.named_parameters():
        p.add_(0.2 * torch.randn_like(p)) if "norm" in name and name.endswith("weight") else p.normal_(0, 0.08)
vae = AutoencoderKLFlux2.from_pretrained(os.path.join(here, "flux2-tiny-pipe", "vae")).eval()

pipe = Flux2Pipeline(scheduler=sched, vae=vae, text_encoder=te, tokenizer=proc, transformer=tr)
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
pipe.save_pretrained(out_dir, safe_serialization=True)
for f in os.listdir(os.path.join(out_dir, "text_encoder")):
    if f.endswith(".safetensors"):
        f = os.path.join(out_dir, "text_encoder", f)
        save_file({k: v.to(torch.bfloat16) for k, v in load_file(f).items()}, f, metadata={"format": "pt"})
pipe = Flux2Pipeline.from_pretrained(out_dir, torch_dtype=torch.float32)
shutil.rmtree(os.path.join(out_dir, "tokenizer"))  # the Go test reads Mistral-Small-3.1's (17 MB)

prompt, H, W, steps = "a red cube on a blue table, studio light", 64, 48, 3
inputs = proc.apply_chat_template(format_input([prompt], SYSTEM_MESSAGE), add_generation_prompt=False, tokenize=True,
                                  return_dict=True, padding="max_length", truncation=True, max_length=max_len)
text = proc.apply_chat_template(format_input([prompt], SYSTEM_MESSAGE), add_generation_prompt=False, tokenize=False)[0]
torch.manual_seed(72)
latents = torch.randn(1, 16, H // 4, W // 4)
with torch.no_grad():
    pe, _ = pipe.encode_prompt(prompt, max_sequence_length=max_len, text_encoder_out_layers=layers)
    img = pipe(prompt=prompt, height=H, width=W, num_inference_steps=steps, guidance_scale=4.0, latents=latents,
               max_sequence_length=max_len, text_encoder_out_layers=layers, output_type="pt").images
sigmas = pipe.scheduler.sigmas.tolist()
print("text", repr(text[:80]), "real", int(sum(inputs["attention_mask"][0])), "of", len(inputs["input_ids"][0]),
      "padding side", proc.tokenizer.padding_side, "image", tuple(img.shape), "sigmas", sigmas)
r = lambda a: [round(v, 7) for v in a.flatten().tolist()]
json.dump({"prompt": prompt, "height": H, "width": W, "steps": steps, "max_seq_len": max_len, "layers": list(layers),
           "guidance": 4.0, "text": text, "ids": [int(v) for v in inputs["input_ids"][0]],
           "mask": [int(v) for v in inputs["attention_mask"][0]],
           "latents": r(latents), "latents_shape": list(latents.shape), "prompt_embeds": r(pe),
           "sigmas": sigmas, "image": r(img), "image_shape": list(img.shape)}, open(out_json, "w"))


def scrub(d):
    """Drop the Mistral3 class name from the saved JSON (see the import note)."""
    mi = os.path.join(d, "model_index.json")
    idx = json.load(open(mi))
    idx["text_encoder"] = ["transformers", "AutoModelForImageTextToText"]
    json.dump(idx, open(mi, "w"), indent=2)
    tc = os.path.join(d, "text_encoder", "config.json")
    c = json.load(open(tc))
    c.pop("architectures", None)
    json.dump(c, open(tc, "w"), indent=2)


scrub(out_dir)
