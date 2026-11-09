#!/usr/bin/env python3
"""End-to-end Flux2KleinPipeline reference (diffusers, CPU, F32) for
diffusion/flux2_test.go.

    .venv/bin/python scripts/verify/ref_flux2_pipe.py <out_dir> <out.json>

Builds a seeded tiny klein pipeline - FLUX.2-klein-4B's tokenizer and
scheduler, a tiny random Qwen3 text encoder (BF16 weights), a tiny
Flux2Transformer2DModel and a tiny AutoencoderKLFlux2 (decoder + batch-norm
stats only on disk) - saves it to <out_dir>, and runs it non-distilled
(classifier-free guidance 3.0) for 2 steps at 64 x 64 with explicit latents,
text encoder layers (1, 2, 3) and 40 text tokens (padded). Writes the prompt ids,
the prompt embeddings, the sigmas and the final image to <out.json>.
"""
import json, os, shutil, sys
import torch
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

out_dir, out_json = sys.argv[1], sys.argv[2]
klein = "black-forest-labs/FLUX.2-klein-4B"
tok = AutoTokenizer.from_pretrained(klein, subfolder="tokenizer")
sched = FlowMatchEulerDiscreteScheduler.from_pretrained(klein, subfolder="scheduler")
torch.manual_seed(5)
te = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                                  num_attention_heads=2, num_key_value_heads=1, head_dim=8, tie_word_embeddings=True,
                                  max_position_embeddings=512, rope_theta=1000000.0, rms_norm_eps=1e-6)).eval()
with torch.no_grad():
    for name, p in te.named_parameters():
        if "norm" in name:
            p.add_(0.2 * torch.randn_like(p))
        else:
            p.normal_(0, 0.1)
te = te.to(torch.bfloat16).to(torch.float32)  # the saved BF16 weights, exactly
tr = Flux2Transformer2DModel(patch_size=1, in_channels=16, num_layers=1, num_single_layers=1, attention_head_dim=16,
                             num_attention_heads=2, joint_attention_dim=48, mlp_ratio=3.0, axes_dims_rope=(4, 4, 4, 4),
                             rope_theta=2000, guidance_embeds=False).eval()
with torch.no_grad():
    for name, p in tr.named_parameters():
        if "norm" in name and name.endswith("weight"):
            p.add_(0.2 * torch.randn_like(p))
        else:
            p.normal_(0, 0.08)
vae = AutoencoderKLFlux2(block_out_channels=(32, 32), down_block_types=("DownEncoderBlock2D",) * 2,
                         up_block_types=("UpDecoderBlock2D",) * 2, layers_per_block=1, latent_channels=4,
                         norm_num_groups=32).eval()
with torch.no_grad():
    for p in vae.parameters():
        p.add_(0.05 * torch.randn_like(p))
    vae.bn.running_mean.copy_(0.1 * torch.randn_like(vae.bn.running_mean))
    vae.bn.running_var.copy_(0.5 + torch.rand_like(vae.bn.running_var))
pipe = Flux2KleinPipeline(scheduler=sched, vae=vae, text_encoder=te, tokenizer=tok, transformer=tr, is_distilled=False)
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
pipe.save_pretrained(out_dir, safe_serialization=True)
f = os.path.join(out_dir, "text_encoder", "model.safetensors")
save_file({k: v.to(torch.bfloat16) for k, v in load_file(f).items()}, f, metadata={"format": "pt"})
f = os.path.join(out_dir, "vae", "diffusion_pytorch_model.safetensors")
save_file({k: v for k, v in load_file(f).items() if not k.startswith(("encoder.", "quant_conv."))}, f)

# The tokenizer (11 MB) is not kept: the Go test reads FLUX.2-klein-4B's.
shutil.rmtree(os.path.join(out_dir, "tokenizer"))
prompt, H, W, steps, max_len, layers = "a red cube on a blue table", 64, 64, 2, 40, (1, 2, 3)
torch.manual_seed(6)
latents = torch.randn(1, 16, 16, 16)
with torch.no_grad():
    pe, _ = pipe.encode_prompt(prompt, max_sequence_length=max_len, text_encoder_out_layers=layers)
    img = pipe(prompt=prompt, height=H, width=W, num_inference_steps=steps, guidance_scale=3.0, latents=latents,
               max_sequence_length=max_len, text_encoder_out_layers=layers, output_type="pt").images
msgs = [{"role": "user", "content": prompt}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
enc = tok(text, padding="max_length", truncation=True, max_length=max_len)
# The conditioning in float64 too: transformers' F32 Qwen3 drifts by ~2e-4
# here (its F32 rotary angles), Light by ~2e-7.
te64 = Qwen3ForCausalLM.from_pretrained(os.path.join(out_dir, "text_encoder"), dtype=torch.float64).eval()
with torch.no_grad():
    hs = te64(input_ids=torch.tensor([enc["input_ids"]]), attention_mask=torch.tensor([enc["attention_mask"]]),
              output_hidden_states=True).hidden_states
pe64 = torch.stack([hs[k] for k in layers], 1).permute(0, 2, 1, 3).reshape(1, max_len, -1)
print("ids", enc["input_ids"], "image", tuple(img.shape), "sigmas", pipe.scheduler.sigmas.tolist())
r = lambda a: [round(v, 7) for v in a.flatten().tolist()]
json.dump({"prompt": prompt, "height": H, "width": W, "steps": steps, "max_seq_len": max_len, "layers": list(layers),
           "guidance": 3.0, "text": text, "ids": enc["input_ids"], "mask": enc["attention_mask"],
           "latents": r(latents), "latents_shape": list(latents.shape), "prompt_embeds": r(pe), "prompt_embeds_f64": r(pe64),
           "sigmas": pipe.scheduler.sigmas.tolist(), "image": r(img), "image_shape": list(img.shape)},
          open(out_json, "w"))
