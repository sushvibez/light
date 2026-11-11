#!/usr/bin/env python3
"""End-to-end FLUX 3 (distilled, flow map) text-to-image reference
(diffusers 0.42.1 + transformers, CPU, F32) for diffusion/flux3_test.go.

    .venv/bin/python scripts/verify/ref_flux3_pipe.py <out_dir> <out.json>

Builds a seeded tiny pipeline folder - a tiny Qwen3-VL text encoder
(Qwen3VLForConditionalGeneration, BF16 weights) with Qwen3-VL-4B-Instruct's
tokenizer, a tiny flow-map Flux3Transformer2DModel and the tiny
AutoencoderKLFlux2 of diffusion/testdata/flux2-tiny-pipe - and runs the
modular pipeline's distilled text-to-image recipe step by step with its own
building blocks: Flux3TextEncoderStep's prompt embeddings (chat template,
right padding to a multiple of 80, padding at position 1, stacked hidden
states), Flux3SetTimestepsStep's schedule (linspace sigmas, shift 6.93),
the image / text RoPE ids, Flux3DistilledLoopDenoiser's timestep / target,
FlowMatchEulerDiscreteScheduler.step and Flux3DecodeStep's batch-norm
inverse, unpatchify and VAE decode. Explicit latents; 3 steps at 64 x 48.
"""
import json, os, shutil, sys
import numpy as np
import torch
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux3Transformer2DModel
from diffusers.modular_pipelines.flux3.before_denoise import _prepare_image_ids, retrieve_timesteps
from diffusers.modular_pipelines.flux3.before_denoise import Flux3TextRoPEInputsStep
from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from safetensors.torch import load_file, save_file

out_dir, out_json = sys.argv[1], sys.argv[2]
here = os.path.dirname(os.path.abspath(out_dir))
proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
layers = (1, 2, 3)

torch.manual_seed(61)
cfg = Qwen3VLConfig(
    text_config=dict(vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                     num_attention_heads=2, num_key_value_heads=1, head_dim=8, rms_norm_eps=1e-6,
                     max_position_embeddings=1024, rope_theta=5000000.0,
                     rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True}),
    vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16,
                       deepstack_visual_indexes=[0]),
    tie_word_embeddings=True)
te = Qwen3VLForConditionalGeneration(cfg).eval()
with torch.no_grad():
    for name, p in te.named_parameters():
        if "norm" in name:
            p.copy_(1 + 0.2 * torch.randn_like(p))
        else:
            p.normal_(0, 0.1)
te = te.to(torch.bfloat16).to(torch.float32)

D3 = 16 * len(layers)
tr = Flux3Transformer2DModel(in_channels=16, cond_in_channels=16, num_layers=1, num_single_layers=2,
                             attention_head_dim=16, num_attention_heads=2, joint_attention_dim=D3,
                             axes_dims_rope=(4, 4, 4, 4), flow_map=True).eval()
with torch.no_grad():
    for name, p in tr.named_parameters():
        p.copy_(1 + 0.2 * torch.randn_like(p)) if "norm" in name else p.normal_(0, 0.08)
vae = AutoencoderKLFlux2.from_pretrained(os.path.join(here, "flux2-tiny-pipe", "vae")).eval()

if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
te.save_pretrained(os.path.join(out_dir, "text_encoder"), safe_serialization=True)
f = os.path.join(out_dir, "text_encoder", "model.safetensors")
save_file({k: v.to(torch.bfloat16) for k, v in load_file(f).items()}, f, metadata={"format": "pt"})
tr.save_pretrained(os.path.join(out_dir, "transformer"), safe_serialization=True)
vae.save_pretrained(os.path.join(out_dir, "vae"), safe_serialization=True)
json.dump({"_class_name": "Flux3DistilledModularPipeline"}, open(os.path.join(out_dir, "modular_model_index.json"), "w"))

# Reload the encoder: the in-memory BF16 round trip above also rounded its RoPE inv_freq buffer, which a
# checkpoint load recomputes exactly.
te = Qwen3VLForConditionalGeneration.from_pretrained(os.path.join(out_dir, "text_encoder"), dtype=torch.float32).eval()

prompt, H, W, steps = "a red cube on a blue table, studio light", 64, 48, 3
with torch.no_grad():
    embeds = Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(te, proc, prompt, device="cpu", max_sequence_length=8192,
                                                             pad_multiple=80, hidden_states_layers=layers)
    embeds64 = Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(te.double(), proc, prompt, device="cpu",
                                                               max_sequence_length=8192, pad_multiple=80,
                                                               hidden_states_layers=layers)
    te.float()
    print("F32 vs F64 embeds", (embeds - embeds64).abs().max().item())
text = proc.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
ids = proc.tokenizer(text).input_ids
txt_ids = Flux3TextRoPEInputsStep._prepare_text_ids(embeds)

sched = FlowMatchEulerDiscreteScheduler(shift=6.93)
timesteps, _ = retrieve_timesteps(sched, steps, "cpu", sigmas=np.linspace(1.0, 1 / steps, steps))
sigmas = sched.sigmas.tolist()

vs = 2 ** (len(vae.config.block_out_channels) - 1)
lh, lw = H // (vs * 2), W // (vs * 2)
torch.manual_seed(62)
latents = torch.randn(1, 16, lh, lw)
x = latents.flatten(2).transpose(1, 2)
img_ids = _prepare_image_ids(lh, lw, 0.5)
with torch.no_grad():
    for i, t in enumerate(timesteps):
        nt = timesteps[i + 1] if i + 1 < len(timesteps) else torch.zeros_like(t)
        v = tr(hidden_states=x, encoder_hidden_states=embeds, timestep=(t / 1000).expand(1),
               target_timestep=(nt / 1000).expand(1), img_ids=img_ids, txt_ids=txt_ids).sample
        x = sched.step(v, t, x, return_dict=False)[0]
    z = x.view(1, lh, lw, 16).permute(0, 3, 1, 2)
    mean = vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    z = z * std + mean
    z = z.reshape(1, 4, 2, 2, lh, lw).permute(0, 1, 4, 2, 5, 3).reshape(1, 4, lh * 2, lw * 2)
    img = vae.decode(z, return_dict=False)[0]
    img = (img / 2 + 0.5).clamp(0, 1)
print("ids", len(ids), "embeds", tuple(embeds.shape), "sigmas", sigmas, "image", tuple(img.shape))
r = lambda a: [round(v, 7) for v in a.flatten().tolist()]
json.dump({"prompt": prompt, "height": H, "width": W, "steps": steps, "layers": list(layers), "ids": ids,
           "embeds": r(embeds), "embeds_f64": r(embeds64), "embeds_shape": list(embeds.shape), "sigmas": sigmas, "latents": r(latents),
           "latents_shape": list(latents.shape), "image": r(img), "image_shape": list(img.shape)},
          open(out_json, "w"))
