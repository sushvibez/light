#!/usr/bin/env python3
"""Stable Diffusion 1.5 reference (diffusers, CPU, F32) for
models/sd15_test.go (LIGHT_HEAVY_TESTS).

    .venv/bin/python scripts/verify/ref_sd15_pipe.py <init.png> <out.json>

Runs stable-diffusion-v1-5 with DDIM from the repo's scheduler config
(leading spacing, steps offset 1, final alpha ᾱ_0) at 128 x 128 for 4 steps
with CFG 7.5: text-to-image from explicit latents, then image-to-image on
<init.png> (written here: a deterministic 128 x 128 picture) at strength
0.75 with the VAE's mode latents and explicit noise. Writes both images
(output_type "pt", in [0, 1], rounded to 5 decimals) to <out.json>.
"""
import json, sys
import numpy as np
import torch
from PIL import Image, ImageDraw
import diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img as m
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

init_path, out_json = sys.argv[1], sys.argv[2]
repo = "stable-diffusion-v1-5/stable-diffusion-v1-5"
sched = DDIMScheduler.from_pretrained(repo, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(repo, scheduler=sched, safety_checker=None,
                                               requires_safety_checker=False, torch_dtype=torch.float32)
prompt, H, W, steps, cfg = "a watercolor painting of a lighthouse on a cliff", 128, 128, 4, 7.5

torch.manual_seed(5)
latents = torch.randn(1, 4, H // 8, W // 8)
with torch.no_grad():
    img = pipe(prompt=prompt, height=H, width=W, num_inference_steps=steps, guidance_scale=cfg, latents=latents,
               output_type="pt").images
print("txt2img", tuple(img.shape), img.mean().item(), "timesteps", pipe.scheduler.timesteps.tolist())

# A deterministic init picture: sky gradient, sea, a sun and a box.
yy, xx = np.mgrid[0:H, 0:W]
a = np.zeros((H, W, 3), np.uint8)
a[..., 0] = 90 + xx * 120 // W
a[..., 1] = 140 + yy * 80 // H
a[..., 2] = 230 - yy * 60 // H
a[H * 2 // 3:] = (20, 70, 140)
im = Image.fromarray(a)
d = ImageDraw.Draw(im)
d.ellipse((80, 14, 110, 44), fill=(250, 220, 80))
d.rectangle((24, 50, 52, 96), fill=(200, 40, 40))
im.save(init_path)

m.retrieve_latents = lambda enc, generator=None, sample_mode="sample": enc.latent_dist.mode()
torch.manual_seed(6)
noise = torch.randn(1, 4, H // 8, W // 8)
m.randn_tensor = lambda shape, generator=None, device=None, dtype=None: noise.clone()
i2p = StableDiffusionImg2ImgPipeline(**pipe.components)
with torch.no_grad():
    i2 = i2p(prompt=prompt, image=Image.open(init_path).convert("RGB"), strength=0.75, num_inference_steps=steps,
             guidance_scale=cfg, output_type="pt").images
print("img2img", tuple(i2.shape), i2.mean().item())
r = lambda t: [round(v, 5) for v in t.flatten().tolist()]
json.dump({"prompt": prompt, "height": H, "width": W, "steps": steps, "guidance": cfg, "latents": latents.flatten().tolist(),
           "image": r(img), "i2i_strength": 0.75, "i2i_noise": noise.flatten().tolist(), "i2i_image": r(i2)},
          open(out_json, "w"))
