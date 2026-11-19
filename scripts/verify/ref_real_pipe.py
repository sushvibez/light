#!/usr/bin/env python3
"""Real-weight end-to-end reference of Z-Image-Turbo, FLUX.2 klein 4B or
FLUX.1 (schnell / dev; diffusers, BF16 on CUDA with CPU offload) for
diffusion's *RealCUDA tests.
    .venv/bin/python scripts/verify/ref_real_pipe.py zimage|klein|flux2dev|schnell|flux1dev <out.json> [size] [steps]
flux2dev (FLUX.2-dev: 32B transformer + Mistral-3 24B encoder, 112 GB) needs
an 80 GB card with model CPU offload (an H100 on Colab).
FLUX.1's 12B transformer exceeds a 16 GB card, so it runs with sequential
CPU offload (each module uploaded for its forward); the weights stay
memory-mapped on the host.
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
elif which == "flux2dev":
    from diffusers import Flux2Pipeline
    import diffusers.pipelines.flux2.pipeline_flux2 as m
    repo, guidance = "black-forest-labs/FLUX.2-dev", 4.0
    pipe = Flux2Pipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
    m.randn_tensor = patched_randn
    pipe.enable_model_cpu_offload()
    img = pipe(prompt=prompt, height=size, width=size, num_inference_steps=steps, guidance_scale=guidance, output_type="pt").images
elif which in ("schnell", "flux1dev"):
    from diffusers import FluxPipeline
    import diffusers.pipelines.flux.pipeline_flux as m
    repo = "black-forest-labs/FLUX.1-schnell" if which == "schnell" else "black-forest-labs/FLUX.1-dev"
    guidance = 0.0 if which == "schnell" else 3.5
    pipe = FluxPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
    m.randn_tensor = patched_randn
    pipe.enable_sequential_cpu_offload()
    seq = 256 if which == "schnell" else 512
    # Intermediates for the Go test: text conditioning and the latents after
    # the first step, so a mismatch can be placed.
    extra = {}
    def on_step(p, i, t, kw):
        if i == 0:
            extra["latents_step0"] = kw["latents"].float().cpu()
        return {}
    with torch.no_grad():
        pe, pooled, _ = pipe.encode_prompt(prompt, None, max_sequence_length=seq)
    extra["prompt_embeds"], extra["pooled"] = pe.float().cpu(), pooled.float().cpu()
    img = pipe(prompt=prompt, height=size, width=size, num_inference_steps=steps, guidance_scale=guidance,
               max_sequence_length=seq, output_type="pt", callback_on_step_end=on_step,
               callback_on_step_end_tensor_inputs=["latents"]).images
    extra["sigmas"] = pipe.scheduler.sigmas.float().cpu()
else:
    sys.exit("zimage, klein, flux2dev, schnell or flux1dev")

lat = captured["latents"]
img = img[0].float().clamp(0, 1).cpu()
out = {"repo": repo, "prompt": prompt, "height": size, "width": size, "steps": steps, "guidance": guidance,
       "latents": lat.flatten().tolist(), "latents_shape": list(lat.shape),
       "image": img.flatten().tolist(), "image_shape": list(img.shape)}
for k, v in globals().get("extra", {}).items():
    out[k], out[k + "_shape"] = v.flatten().tolist(), list(v.shape)
json.dump(out, open(out_json, "w"))
print("wrote", out_json, "latents", list(lat.shape), "image", list(img.shape))
