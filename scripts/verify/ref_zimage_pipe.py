#!/usr/bin/env python3
"""End-to-end ZImagePipeline reference (diffusers, CPU, F32) for
diffusion/zimage_test.go.

    .venv/bin/python scripts/verify/ref_zimage_pipe.py <out_dir> <out.json>

Builds a seeded tiny Z-Image pipeline - Z-Image-Turbo's scheduler, a Qwen3
tokenizer (FLUX.2-klein-4B's, already cached; Z-Image's encoder is Qwen3
too), a tiny random Qwen3Model text encoder (BF16 weights), a tiny
ZImageTransformer2DModel and a tiny 16-channel-style AutoencoderKL with
scaling / shift factors - saves it (without the tokenizer) to <out_dir>
and runs it with classifier-free guidance 2.0 for 3 steps at 64 x 64 from
explicit latents. Writes the prompt ids, the caption features (in float64
and transformers' F32), the sigmas and the final image to <out.json>.
"""
import json, os, shutil, sys
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline, ZImageTransformer2DModel
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

out_dir, out_json = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained("black-forest-labs/FLUX.2-klein-4B", subfolder="tokenizer")
sched = FlowMatchEulerDiscreteScheduler.from_pretrained("Tongyi-MAI/Z-Image-Turbo", subfolder="scheduler")
torch.manual_seed(9)
te = Qwen3Model(Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=8, tie_word_embeddings=False,
                            max_position_embeddings=512, rope_theta=1000000.0, rms_norm_eps=1e-6)).eval()
with torch.no_grad():
    for name, p in te.named_parameters():
        if "norm" in name:
            p.add_(0.2 * torch.randn_like(p))
        else:
            p.normal_(0, 0.1)
te = te.to(torch.bfloat16).to(torch.float32)
tr = ZImageTransformer2DModel(all_patch_size=(2,), all_f_patch_size=(1,), in_channels=4, dim=64, n_layers=2,
                              n_refiner_layers=1, n_heads=4, n_kv_heads=4, norm_eps=1e-5, qk_norm=True,
                              cap_feat_dim=16, rope_theta=256.0, t_scale=1000.0, axes_dims=[4, 6, 6],
                              axes_lens=[256, 32, 32]).eval()
with torch.no_grad():
    for name, p in tr.named_parameters():
        if "norm" in name and name.endswith("weight"):
            p.add_(0.2 * torch.randn_like(p))
        else:
            p.normal_(0, 0.08)
vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(32, 32),
                    down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2,
                    layers_per_block=1, norm_num_groups=32, use_quant_conv=False, use_post_quant_conv=False,
                    scaling_factor=0.3611, shift_factor=0.1159).eval()
with torch.no_grad():
    for p in vae.parameters():
        p.add_(0.05 * torch.randn_like(p))
pipe = ZImagePipeline(scheduler=sched, vae=vae, text_encoder=te, tokenizer=tok, transformer=tr)
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
pipe.save_pretrained(out_dir, safe_serialization=True)
f = os.path.join(out_dir, "text_encoder", "model.safetensors")
save_file({k: v.to(torch.bfloat16) for k, v in load_file(f).items()}, f, metadata={"format": "pt"})
f = os.path.join(out_dir, "vae", "diffusion_pytorch_model.safetensors")
save_file({k: v for k, v in load_file(f).items() if k.startswith("decoder.")}, f)
shutil.rmtree(os.path.join(out_dir, "tokenizer"))  # the Go test reads klein's (11 MB)

prompt, H, W, steps, max_len = "a red cube on a blue table", 64, 64, 3, 64
torch.manual_seed(10)
latents = torch.randn(1, 4, 32, 32)
with torch.no_grad():
    img = pipe(prompt=prompt, height=H, width=W, num_inference_steps=steps, guidance_scale=2.0,
               latents=latents, max_sequence_length=max_len, output_type="pt").images
sigmas = pipe.scheduler.sigmas.tolist()
text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                               enable_thinking=True)
ids = tok(text).input_ids
te64 = Qwen3Model.from_pretrained(os.path.join(out_dir, "text_encoder"), dtype=torch.float64).eval()
with torch.no_grad():
    cap64 = te64(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states[-2][0]
print("ids", ids, "image", tuple(img.shape), "sigmas", sigmas)
r = lambda a: [round(v, 7) for v in a.flatten().tolist()]
json.dump({"prompt": prompt, "height": H, "width": W, "steps": steps, "max_seq_len": max_len, "guidance": 2.0,
           "text": text, "ids": ids, "latents": r(latents[0]), "latents_shape": list(latents.shape[1:]),
           "caption_f64": r(cap64), "sigmas": sigmas, "image": r(img), "image_shape": list(img.shape)},
          open(out_json, "w"))
