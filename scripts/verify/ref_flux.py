#!/usr/bin/env python3
"""Flux transformer reference (diffusers, CPU, F32) for models/flux_test.go.

    .venv/bin/python scripts/verify/ref_flux.py --make-tiny <dir> <out.json>
    .venv/bin/python scripts/verify/ref_flux.py --make-tiny-vae <dir> <out.json>
    .venv/bin/python scripts/verify/ref_flux.py --make-tiny-flux2 <dir> <out.json>

Writes a seeded tiny-random FluxTransformer2DModel (2 double + 2 single
blocks, guidance embedder, 3-axis RoPE) to <dir> with save_pretrained, runs
one forward pass on seeded inputs (8 text tokens, a 4 x 6 latent grid) and
writes the inputs and the output to <out.json>.
"""
import json, sys
import torch
from diffusers import FluxTransformer2DModel

out_dir, out_json = sys.argv[2], sys.argv[3]
r = lambda a: [round(v, 7) for v in a.flatten().tolist()]
if sys.argv[1] == "--make-tiny-vae":
    # A tiny Flux-style AutoencoderKL: 16 latent channels, no post-quant conv.
    from diffusers import AutoencoderKL
    torch.manual_seed(1)
    vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=16, block_out_channels=(32, 64),
                        down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2,
                        layers_per_block=1, norm_num_groups=32, use_quant_conv=False, use_post_quant_conv=False,
                        scaling_factor=0.3611, shift_factor=0.1159).eval()
    with torch.no_grad():
        for name, p in vae.named_parameters():
            p.add_(0.05 * torch.randn_like(p))
    vae.save_pretrained(out_dir, safe_serialization=True)
    # Keep only the decoder (the test decodes; the encoder would triple the file).
    from safetensors.torch import load_file, save_file
    f = out_dir + "/diffusion_pytorch_model.safetensors"
    save_file({k: v for k, v in load_file(f).items() if k.startswith("decoder.")}, f)
    z = torch.randn(1, 16, 3, 5)
    with torch.no_grad():
        img = vae.decode(z).sample
    print("image", tuple(img.shape), "mean abs", img.abs().mean().item())
    json.dump({"z": r(z), "z_shape": list(z.shape), "image": r(img), "image_shape": list(img.shape)}, open(out_json, "w"))
    sys.exit(0)
if sys.argv[1] == "--make-tiny-flux2":
    # FLUX.2 (klein-style: no guidance embedder): bias-free, shared
    # modulations, SwiGLU, fused single-stream projection, 4-axis RoPE.
    from diffusers import Flux2Transformer2DModel
    torch.manual_seed(3)
    m = Flux2Transformer2DModel(patch_size=1, in_channels=16, num_layers=2, num_single_layers=2,
                                attention_head_dim=16, num_attention_heads=2, joint_attention_dim=24,
                                mlp_ratio=3.0, axes_dims_rope=(4, 4, 4, 4), rope_theta=2000,
                                guidance_embeds=False).eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "norm" in name and name.endswith("weight"):
                p.add_(0.2 * torch.randn_like(p))
            else:
                p.normal_(0, 0.08)
    m.save_pretrained(out_dir, safe_serialization=True)
    St, Hh, Ww = 6, 4, 5
    x = torch.randn(1, Hh * Ww, 16)
    txt = torch.randn(1, St, 24)
    t = torch.tensor([0.6])
    txt_ids = torch.zeros(St, 4)
    txt_ids[:, 3] = torch.arange(St)
    img_ids = torch.zeros(Hh, Ww, 4)
    img_ids[..., 1] = torch.arange(Hh)[:, None]
    img_ids[..., 2] = torch.arange(Ww)[None, :]
    img_ids = img_ids.reshape(Hh * Ww, 4)
    with torch.no_grad():
        y = m(hidden_states=x, encoder_hidden_states=txt, timestep=t, img_ids=img_ids, txt_ids=txt_ids,
              guidance=None, return_dict=False)[0]
    print("out", tuple(y.shape), "mean abs", y.abs().mean().item())
    json.dump({"x": r(x), "x_shape": list(x.shape), "txt": r(txt), "txt_shape": list(txt.shape),
               "timestep": t.tolist(), "img_ids": r(img_ids), "txt_ids": r(txt_ids),
               "out": r(y), "out_shape": list(y.shape)}, open(out_json, "w"))
    sys.exit(0)
assert sys.argv[1] == "--make-tiny"
torch.manual_seed(0)
cfg = dict(patch_size=1, in_channels=16, out_channels=None, num_layers=2, num_single_layers=2,
           attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
           pooled_projection_dim=24, guidance_embeds=True, axes_dims_rope=(4, 6, 6))
m = FluxTransformer2DModel(**cfg).eval()
with torch.no_grad():
    for name, p in m.named_parameters():
        if "norm" in name and name.endswith("weight"):
            p.add_(0.2 * torch.randn_like(p))  # RMSNorm weights start at 1
        else:
            p.normal_(0, 0.08)                   # incl. the zero-initialised AdaLN / out layers
m.save_pretrained(out_dir, safe_serialization=True)

B, St, H, W = 1, 8, 4, 6
x = torch.randn(B, H * W, 16)
txt = torch.randn(B, St, 32)
pooled = torch.randn(B, 24)
t = torch.tensor([0.7])
g = torch.tensor([3.5])
txt_ids = torch.zeros(St, 3)
img_ids = torch.zeros(H, W, 3)
img_ids[..., 1] = torch.arange(H)[:, None]
img_ids[..., 2] = torch.arange(W)[None, :]
img_ids = img_ids.reshape(H * W, 3)
with torch.no_grad():
    y = m(hidden_states=x, encoder_hidden_states=txt, pooled_projections=pooled, timestep=t,
          img_ids=img_ids, txt_ids=txt_ids, guidance=g, return_dict=False)[0]
print("out", tuple(y.shape), "mean abs", y.abs().mean().item())
json.dump({"x": r(x), "x_shape": list(x.shape), "txt": r(txt), "txt_shape": list(txt.shape),
           "pooled": r(pooled), "timestep": t.tolist(), "guidance": g.tolist(),
           "img_ids": r(img_ids), "txt_ids": r(txt_ids), "out": r(y), "out_shape": list(y.shape)},
          open(out_json, "w"))
