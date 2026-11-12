#!/usr/bin/env python3
"""Real-weight end-to-end reference of Z-Image-Turbo or FLUX.2 klein 4B
(diffusers, BF16 on CUDA with CPU offload) for diffusion's *RealCUDA tests.
    .venv/bin/python scripts/verify/ref_real_pipe.py zimage|klein <out.json> [size] [steps]
Runs the cached pipeline on a fixed prompt with seeded initial noise, which
it captures by patching the pipeline module's randn_tensor (so the JSON
carries the exact latents and their shape, as the tiny references do), and
writes the prompt, size, steps, guidance, latents and the final image
(float, (3, H, W) in [0, 1]) to <out.json>. Light runs the same latents on
the GPU and compares.
"""
import json, sys
import torch

which, out_json = sys.argv[1], sys.argv[2]
size = int(sys.argv[3]) if len(sys.argv) > 3 else 512
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 4
prompt = "a watercolour painting of a red lighthouse on a rocky shore at dawn, soft light"
torch.manual_seed(7)
gen = torch.Generator("cpu").manual_seed(7)
captured = {}

def patched_randn(shape, generator=None, device=None, dtype=None, layout=None):
    if "latents" not in captured:
        captured["latents"] = torch.randn(shape, generator=gen, dtype=torch.float32)
    return captured["latents"].to(device=device, dtype=dtype)

if which == "zimage":
    from diffusers import ZImagePipeline
    import diffusers.pipelines.z_image.pipeline_z_image as m
    repo, guidance = "Tongyi-MAI/Z-Image-Turbo", 0.0
    pipe = ZImagePipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
    m.randn_tensor = patched_randn
    pipe.enable_model_cpu_offload()
    img = pipe(prompt=prompt, height=size, width=size, num_inference_steps=steps, guidance_scale=guidance, output_type="pt").images
elif which == "klein":
    from diffusers import Flux2KleinPipeline
    import diffusers.pipelines.flux2.pipeline_flux2_klein as m
    repo, guidance = "black-forest-labs/FLUX.2-klein-4B", 0.0
    pipe = Flux2KleinPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
    m.randn_tensor = patched_randn
    pipe.enable_model_cpu_offload()
    img = pipe(prompt=prompt, height=size, width=size, num_inference_steps=steps, output_type="pt").images
else:
    sys.exit("zimage or klein")

lat = captured["latents"]
img = img[0].float().clamp(0, 1).cpu()
json.dump({"repo": repo, "prompt": prompt, "height": size, "width": size, "steps": steps, "guidance": guidance,
           "latents": lat.flatten().tolist(), "latents_shape": list(lat.shape),
           "image": img.flatten().tolist(), "image_shape": list(img.shape)}, open(out_json, "w"))
print("wrote", out_json, "latents", list(lat.shape), "image", list(img.shape))
