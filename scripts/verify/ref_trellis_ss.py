#!/usr/bin/env python3
"""Tiny TRELLIS stage-1 reference (microsoft/TRELLIS, CPU, F32) for
models/trellis_test.go and diffusion/trellis_test.go.

    python scripts/verify/ref_trellis_ss.py <trellis src> <dinov2 src> <out_dir> <out.json>

<trellis src> holds TRELLIS's trellis/ package (only the dense modules are
imported: pipelines/trellis_image_to_3d.py, pipelines/samplers,
models/sparse_structure_{flow,vae}.py, modules/{attention,transformer,norm,
spatial,utils}.py; the sparse package and rembg are stubbed, attention runs
on the sdpa backend). <dinov2 src> holds facebookresearch/dinov2's dinov2/
package (the torch.hub code TRELLIS loads dinov2_vitl14_reg from).

Builds seeded tiny versions of
  - DINOv2 with registers: the torch.hub DinoVisionTransformer with
    dinov2_vitl14_reg's arguments (4 registers, layer scale, MLP FFN), saved
    under facebook/dinov2-with-registers-large's names (transformers'
    Dinov2WithRegistersModel, old attention.attention.query/... keys) and
    checked against transformers loading that file;
  - SparseStructureFlowModel and SparseStructureDecoder, saved as
    ckpts/<name>.json + .safetensors with the torso (blocks / middle_block
    linears and convs) in F16 as the released *_fp16 checkpoints are (the
    reference computes in F32 from the same F16-rounded values),
plus a pipeline.json, and records with TRELLIS's own pipeline code:
  dinov2    hub x_prenorm vs transformers hidden_states[-1] (max diff), and
            TrellisImageTo3DPipeline.encode_image's tokens for a small image
  flow      one forward (x, t, cond)
  sample    sample_sparse_structure's latent with pipeline.json's sampler
            (25 Euler steps, CFG 5 in t in [0.5, 1], rescale_t 3) from the
            recorded torch.randn noise, the decoder's logits and the coords
  preprocess TrellisImageTo3DPipeline.preprocess_image on an RGBA image
            (saved as object.png -> preprocessed.png, 518 x 518)
"""
import json, os, sys, types

import numpy as np
import torch

trellis_src, dinov2_src, out_dir, out_json = sys.argv[1:5]
os.environ["ATTN_BACKEND"] = "sdpa"
os.environ["XFORMERS_DISABLED"] = "1"


# --- import TRELLIS's dense modules without its heavy package __init__s -------
def stub_package(name, path):
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m
    return m


sys.path.insert(0, trellis_src)
stub_package("trellis", os.path.join(trellis_src, "trellis"))
stub_package("trellis.pipelines", os.path.join(trellis_src, "trellis", "pipelines"))
mods = stub_package("trellis.modules", os.path.join(trellis_src, "trellis", "modules"))
sp = types.ModuleType("trellis.modules.sparse")
for n in ("SparseConv3d", "SparseInverseConv3d", "SparseLinear", "SparseTensor"):
    setattr(sp, n, type(n, (), {}))
sys.modules["trellis.modules.sparse"] = sp
mods.sparse = sp
sys.modules.setdefault("rembg", types.ModuleType("rembg"))
if "easydict" not in sys.modules:
    try:
        import easydict  # noqa: F401
    except ImportError:
        ed = types.ModuleType("easydict")

        class EasyDict(dict):
            __getattr__ = dict.__getitem__

            def __setattr__(self, k, v):
                self[k] = v

        ed.EasyDict = EasyDict
        sys.modules["easydict"] = ed

from trellis.models.sparse_structure_flow import SparseStructureFlowModel  # noqa: E402
from trellis.models.sparse_structure_vae import SparseStructureDecoder  # noqa: E402
from trellis.pipelines import samplers  # noqa: E402
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline  # noqa: E402

sys.path.insert(0, dinov2_src)
from functools import partial  # noqa: E402

from dinov2.layers import MemEffAttention, NestedTensorBlock  # noqa: E402
from dinov2.models.vision_transformer import DinoVisionTransformer  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from torchvision import transforms  # noqa: E402

r = lambda t: [round(v, 7) for v in t.detach().flatten().tolist()]
out = {}
os.makedirs(os.path.join(out_dir, "ckpts"), exist_ok=True)

# --- DINOv2 with registers ---------------------------------------------------
D, S, P = 64, 56, 14
torch.manual_seed(1)
hub = DinoVisionTransformer(img_size=S, patch_size=P, embed_dim=D, depth=2, num_heads=4, mlp_ratio=4,
                            block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention), init_values=1.0,
                            ffn_layer="mlp", block_chunks=0, num_register_tokens=4, interpolate_antialias=True,
                            interpolate_offset=0.0).eval()
with torch.no_grad():
    for n, p in hub.named_parameters():
        if n.endswith("gamma"):  # layer scale
            p.copy_(0.5 + 0.2 * torch.randn_like(p))
        elif "norm" in n and n.endswith("weight"):
            p.copy_(1 + 0.2 * torch.randn_like(p))
        else:
            p.normal_(0, 0.1)
sd = {k: v.detach().clone() for k, v in hub.state_dict().items()}
hf = {
    "embeddings.cls_token": sd["cls_token"], "embeddings.mask_token": sd["mask_token"],
    "embeddings.position_embeddings": sd["pos_embed"], "embeddings.register_tokens": sd["register_tokens"],
    "embeddings.patch_embeddings.projection.weight": sd["patch_embed.proj.weight"],
    "embeddings.patch_embeddings.projection.bias": sd["patch_embed.proj.bias"],
    "layernorm.weight": sd["norm.weight"], "layernorm.bias": sd["norm.bias"],
}
for i in range(2):
    b, e = f"blocks.{i}.", f"encoder.layer.{i}."
    for part, name in zip(sd[b + "attn.qkv.weight"].chunk(3), ("query", "key", "value")):
        hf[e + f"attention.attention.{name}.weight"] = part
    for part, name in zip(sd[b + "attn.qkv.bias"].chunk(3), ("query", "key", "value")):
        hf[e + f"attention.attention.{name}.bias"] = part
    pairs = {"attn.proj": "attention.output.dense", "norm1": "norm1", "norm2": "norm2", "mlp.fc1": "mlp.fc1",
             "mlp.fc2": "mlp.fc2"}
    for a, z in pairs.items():
        hf[e + z + ".weight"], hf[e + z + ".bias"] = sd[b + a + ".weight"], sd[b + a + ".bias"]
    hf[e + "layer_scale1.lambda1"], hf[e + "layer_scale2.lambda1"] = sd[b + "ls1.gamma"], sd[b + "ls2.gamma"]
ddir = os.path.join(out_dir, "dinov2")
os.makedirs(ddir, exist_ok=True)
save_file({k: v.contiguous() for k, v in hf.items()}, os.path.join(ddir, "model.safetensors"))
dcfg = {"architectures": ["Dinov2WithRegistersModel"], "model_type": "dinov2_with_registers", "hidden_size": D,
        "num_hidden_layers": 2, "num_attention_heads": 4, "mlp_ratio": 4, "image_size": S, "patch_size": P,
        "num_channels": 3, "num_register_tokens": 4, "layer_norm_eps": 1e-6, "hidden_act": "gelu",
        "qkv_bias": True, "layerscale_value": 1.0, "use_swiglu_ffn": False, "drop_path_rate": 0.0,
        "interpolate_antialias": True, "interpolate_offset": 0.0, "torch_dtype": "float32"}
json.dump(dcfg, open(os.path.join(ddir, "config.json"), "w"), indent=2)

from transformers import Dinov2WithRegistersModel  # noqa: E402

tm, info = Dinov2WithRegistersModel.from_pretrained(ddir, output_loading_info=True)
tm.eval()
assert not info["missing_keys"] and not info["unexpected_keys"], info
torch.manual_seed(2)
image = torch.rand(1, 3, S, S)
norm_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
with torch.no_grad():
    hub_out = hub(norm_img, is_training=True)
    t_out = tm(pixel_values=norm_img, output_hidden_states=True)
d_pre = (hub_out["x_prenorm"] - t_out.hidden_states[-1]).abs().max().item()
d_norm = (hub_out["x_norm_patchtokens"] - t_out.last_hidden_state[:, 5:]).abs().max().item()
print(f"dinov2: hub x_prenorm vs transformers hidden_states[-1] max diff {d_pre:.3g}, "
      f"x_norm_patchtokens vs last_hidden_state {d_norm:.3g}")
assert d_pre < 1e-5 and d_norm < 1e-5

# --- the sparse-structure flow model and decoder --------------------------------
flow_args = {"resolution": 4, "in_channels": 8, "out_channels": 8, "model_channels": 64, "cond_channels": D,
             "num_blocks": 2, "num_heads": 4, "mlp_ratio": 4, "patch_size": 1, "pe_mode": "ape",
             "qk_rms_norm": True, "use_fp16": True}
dec_args = {"out_channels": 1, "latent_channels": 8, "num_res_blocks": 2, "num_res_blocks_middle": 2,
            "channels": [32, 16, 8], "use_fp16": True}


def randomize(model, seed, torso):
    """Seeded weights at fan-in scale; the torso (FP16 in the released
    checkpoints) rounded to F16 so the F16 file and the F32 model agree."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "gamma" in name:
                p.copy_(1 + 0.2 * torch.randn_like(p))
            elif "norm" in name and name.endswith("weight") or name.startswith("out_layer.0.weight"):
                p.copy_(1 + 0.1 * torch.randn_like(p))
            elif p.dim() >= 2:
                p.normal_(0, 1 / np.sqrt(p[0].numel()))
            else:
                p.normal_(0, 0.05)
            if name.startswith(torso) and "norm" not in name and "gamma" not in name:
                p.copy_(p.half().float())
    return model.eval()


def save_ckpt(model, name, cfg_name, args, torso):
    sd = {}
    for k, v in model.state_dict().items():
        half = k.startswith(torso) and "norm" not in k and "gamma" not in k
        sd[k] = (v.half() if half else v.float()).contiguous()
    save_file(sd, os.path.join(out_dir, "ckpts", name + ".safetensors"))
    json.dump({"name": cfg_name, "args": args}, open(os.path.join(out_dir, "ckpts", name + ".json"), "w"), indent=4)
    model.load_state_dict({k: v.float() for k, v in load_file(os.path.join(out_dir, "ckpts", name + ".safetensors")).items()})


# use_fp16 builds FP16 torsos; run them in F32 (the comparison is exact up to rounding).
flow = SparseStructureFlowModel(**{**flow_args, "use_fp16": False})
flow = randomize(flow, 3, ("blocks.",))
save_ckpt(flow, "ss_flow_tiny_fp16", "SparseStructureFlowModel", flow_args, ("blocks.",))
dec = SparseStructureDecoder(**{**dec_args, "use_fp16": False})
dec = randomize(dec, 4, ("blocks.", "middle_block."))
with torch.no_grad():
    dec.out_layer[2].bias.fill_(-0.3)
save_ckpt(dec, "ss_dec_tiny_fp16", "SparseStructureDecoder", dec_args, ("blocks.", "middle_block."))
flow.eval(), dec.eval()

pipe_cfg = {"name": "TrellisImageTo3DPipeline", "args": {
    "models": {"sparse_structure_decoder": "ckpts/ss_dec_tiny_fp16",
               "sparse_structure_flow_model": "ckpts/ss_flow_tiny_fp16"},
    "sparse_structure_sampler": {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-5},
                                 "params": {"steps": 25, "cfg_strength": 5.0, "cfg_interval": [0.5, 1.0],
                                            "rescale_t": 3.0}},
    "image_cond_model": "dinov2_vitl14_reg"}}
json.dump(pipe_cfg, open(os.path.join(out_dir, "pipeline.json"), "w"), indent=4)

# --- TRELLIS's pipeline around the tiny models -----------------------------------
pipe = TrellisImageTo3DPipeline()
pipe.models = {"image_cond_model": hub, "sparse_structure_flow_model": flow, "sparse_structure_decoder": dec}
pipe.image_cond_model_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
ss = pipe_cfg["args"]["sparse_structure_sampler"]
pipe.sparse_structure_sampler = getattr(samplers, ss["name"])(**ss["args"])
pipe.sparse_structure_sampler_params = ss["params"]

with torch.no_grad():
    cond = pipe.get_cond(image)
assert torch.equal(cond["neg_cond"], torch.zeros_like(cond["cond"]))
out["dinov2"] = {"pixels": r(image), "size": S, "tokens": r(cond["cond"]), "shape": list(cond["cond"].shape),
                 "hub_vs_transformers_prenorm": d_pre}

torch.manual_seed(5)
x = torch.randn(1, 8, 4, 4, 4)
fc = torch.randn(1, 21, D)
with torch.no_grad():
    v = flow(x, torch.tensor([637.5]), fc)
out["flow"] = {"x": r(x), "t": 637.5, "cond": r(fc), "cond_len": 21, "v": r(v)}

noises = []
_randn = torch.randn


def randn_rec(*a, **k):
    t = _randn(*a, **k)
    noises.append(t.clone())
    return t


torch.manual_seed(42)  # TrellisImageTo3DPipeline.run's default seed
torch.randn = randn_rec
z_holder = {}
_sample = pipe.sparse_structure_sampler.sample


def sample_rec(*a, **k):
    res = _sample(*a, **k)
    z_holder["z"] = res.samples
    return res


pipe.sparse_structure_sampler.sample = sample_rec
with torch.no_grad():
    coords = pipe.sample_sparse_structure(cond, 1)
torch.randn = _randn
z = z_holder["z"]
with torch.no_grad():
    logits = dec(z)
print("latent", tuple(z.shape), "|z| max", z.abs().max().item(), "logits", tuple(logits.shape),
      "occupied", coords.shape[0], "of", logits.numel(), "min |logit|", logits.abs().min().item())
out["sample"] = {"noise": r(noises[0]), "steps": 25, "cfg": 5.0, "interval": [0.5, 1.0], "rescale_t": 3.0,
                 "latent": r(z), "logits": r(logits), "coords": coords.flatten().tolist()}

# --- preprocess_image on an RGBA object --------------------------------------------
from PIL import Image, ImageDraw  # noqa: E402

img = Image.new("RGBA", (96, 72), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.ellipse((20, 14, 70, 60), fill=(200, 60, 40, 255))
d.rectangle((40, 8, 58, 30), fill=(30, 150, 220, 255))
d.polygon([(62, 40), (88, 64), (60, 66)], fill=(240, 220, 60, 160))
img.save(os.path.join(out_dir, "object.png"))
pre = pipe.preprocess_image(Image.open(os.path.join(out_dir, "object.png")))
pre.save(os.path.join(out_dir, "preprocessed.png"))
out["preprocess"] = {"size": pre.size[0]}

json.dump(out, open(out_json, "w"))
print("wrote", out_json)
