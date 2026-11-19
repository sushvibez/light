#!/usr/bin/env python3
"""TRELLIS-image-large stage-1 real-weight check (Colab; a GPU is optional).

    python scripts/colab/trellis_ss_check.py <work_dir> [image.png] [seed]

Downloads into <work_dir>:
  TRELLIS-image-large/            pipeline.json and the two stage-1
                                  checkpoints (ss_flow_img_dit_L_16l8_fp16,
                                  ss_dec_conv3d_16l8_fp16; ~1.1 GB)
  dinov2-with-registers-large/    the transformers port of dinov2_vitl14_reg
                                  (~1.2 GB), for Light
  trellis_src/                    TRELLIS's dense .py files (no git clone,
                                  no spconv / flash-attn / kaolin needed)
and runs TRELLIS's own stage 1 - preprocess_image, get_cond (DINOv2 from
torch.hub, as TRELLIS loads it), sample_sparse_structure with pipeline.json's
sampler under torch.manual_seed(seed) (default 42, run()'s default), the
decoder and the coords - on <image.png> (an RGBA image with the object cut
out; default: a synthetic object drawn here, saved as object.png). Records
in <work_dir>/ref.safetensors:
  pixels   (3, 518, 518)        what encode_image sees, in [0, 1]
  cond     (1, 1374, 1024)      the image tokens
  noise    (1, 8, 16, 16, 16)   the torch.randn draw
  latent   (1, 8, 16, 16, 16)   the sampled occupancy latent
  logits   (1, 1, 64, 64, 64)   the decoder's output
  coords   (N, 4) int32         argwhere(logits > 0)[:, [0, 2, 3, 4]]
and ref.json (settings, counts, and the torch.hub DINOv2 vs transformers
Dinov2WithRegistersModel difference on this image).

The flow model and decoder run in F32 (their FP16 torsos widened; TF32
off), as Light computes; TRELLIS_FP16=1 keeps TRELLIS's native FP16 torsos
(CUDA only) to see how far that drifts.

Then, with Light:
    LIGHT_TRELLIS_DIR=<work_dir> go test -p 1 -run TrellisReal -v ./diffusion
(diffusion/trellis_real_test.go; add -tags nocuda for CPU, which is slow:
~45 flow-model evaluations at 4096 tokens).
"""
import json, os, sys, types, urllib.request

import numpy as np
import torch

work = sys.argv[1]
image_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
FP16 = os.environ.get("TRELLIS_FP16") == "1"
os.makedirs(work, exist_ok=True)
os.environ["ATTN_BACKEND"] = "sdpa"
os.environ.setdefault("XFORMERS_DISABLED", "1")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = "cuda" if torch.cuda.is_available() else "cpu"
if FP16 and dev != "cuda":
    sys.exit("TRELLIS_FP16=1 needs a GPU")

# --- downloads ------------------------------------------------------------------
from huggingface_hub import snapshot_download  # noqa: E402

root = snapshot_download("microsoft/TRELLIS-image-large", local_dir=os.path.join(work, "TRELLIS-image-large"),
                         allow_patterns=["pipeline.json", "ckpts/ss_flow_img_dit_L_16l8_fp16.*",
                                         "ckpts/ss_dec_conv3d_16l8_fp16.*"])
dino_dir = snapshot_download("facebook/dinov2-with-registers-large",
                             local_dir=os.path.join(work, "dinov2-with-registers-large"),
                             allow_patterns=["config.json", "model.safetensors"])
src = os.path.join(work, "trellis_src")
base = "https://raw.githubusercontent.com/microsoft/TRELLIS/main/"
for f in """trellis/pipelines/trellis_image_to_3d.py trellis/pipelines/base.py trellis/pipelines/samplers/__init__.py
trellis/pipelines/samplers/base.py trellis/pipelines/samplers/flow_euler.py
trellis/pipelines/samplers/guidance_interval_mixin.py trellis/pipelines/samplers/classifier_free_guidance_mixin.py
trellis/models/__init__.py trellis/models/sparse_structure_flow.py trellis/models/sparse_structure_vae.py
trellis/modules/norm.py trellis/modules/utils.py trellis/modules/spatial.py trellis/modules/attention/__init__.py
trellis/modules/attention/full_attn.py trellis/modules/attention/modules.py trellis/modules/transformer/__init__.py
trellis/modules/transformer/blocks.py trellis/modules/transformer/modulated.py""".split():
    p = os.path.join(src, f)
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(urllib.request.urlopen(base + f, timeout=60).read())


# --- TRELLIS's dense modules without its heavy package __init__s ----------------------
def stub_package(name, path):
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m
    return m


sys.path.insert(0, src)
stub_package("trellis", os.path.join(src, "trellis"))
stub_package("trellis.pipelines", os.path.join(src, "trellis", "pipelines"))
mods = stub_package("trellis.modules", os.path.join(src, "trellis", "modules"))
sp = types.ModuleType("trellis.modules.sparse")
for n in ("SparseConv3d", "SparseInverseConv3d", "SparseLinear", "SparseTensor"):
    setattr(sp, n, type(n, (), {}))
sys.modules["trellis.modules.sparse"] = sp
mods.sparse = sp
sys.modules.setdefault("rembg", types.ModuleType("rembg"))
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

import trellis.models as tmodels  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from torchvision import transforms  # noqa: E402
from trellis.pipelines import samplers  # noqa: E402
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline  # noqa: E402

args = json.load(open(os.path.join(root, "pipeline.json")))["args"]
flow = tmodels.from_pretrained(os.path.join(root, args["models"]["sparse_structure_flow_model"])).eval()
dec = tmodels.from_pretrained(os.path.join(root, args["models"]["sparse_structure_decoder"])).eval()
if not FP16:
    flow.convert_to_fp32()
    flow.dtype = torch.float32  # convert_to_fp32 leaves the forward's cast at FP16
    dec.convert_to_fp32()
flow.to(dev), dec.to(dev)

pipe = TrellisImageTo3DPipeline()
pipe.models = {"sparse_structure_flow_model": flow, "sparse_structure_decoder": dec}
pipe._init_image_cond_model(args["image_cond_model"])  # torch.hub, as TRELLIS
pipe.models["image_cond_model"].to(dev)
ss = args["sparse_structure_sampler"]
pipe.sparse_structure_sampler = getattr(samplers, ss["name"])(**ss["args"])
pipe.sparse_structure_sampler_params = ss["params"]

# --- the image ------------------------------------------------------------------------
if image_path is None:
    img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle((150, 260, 370, 420), fill=(170, 110, 60, 255))      # a table top and legs
    d.rectangle((165, 420, 190, 480), fill=(120, 80, 40, 255))
    d.rectangle((330, 420, 355, 480), fill=(120, 80, 40, 255))
    d.ellipse((200, 120, 320, 260), fill=(60, 140, 200, 255))         # a vase on it
    d.rectangle((245, 80, 275, 130), fill=(60, 140, 200, 255))
    image_path = os.path.join(work, "object.png")
    img.save(image_path)
elif os.path.abspath(image_path) != os.path.abspath(os.path.join(work, "object.png")):
    Image.open(image_path).save(os.path.join(work, "object.png"))
image = Image.open(os.path.join(work, "object.png"))
assert image.mode == "RGBA", "give an RGBA image with the object cut out (rembg is not part of the check)"

with torch.no_grad():
    pre = pipe.preprocess_image(image)
    pre.save(os.path.join(work, "preprocessed.png"))
    cond = pipe.get_cond([pre])
    pixels = torch.from_numpy(np.array(pre.resize((518, 518), Image.LANCZOS).convert("RGB")).astype(np.float32) / 255)
    pixels = pixels.permute(2, 0, 1).contiguous()

    # torch.hub DINOv2 vs transformers on the same input.
    from transformers import Dinov2WithRegistersModel

    tm = Dinov2WithRegistersModel.from_pretrained(dino_dir).to(dev).eval()
    x = pipe.image_cond_model_transform(pixels[None].to(dev))
    hub_pre = pipe.models["image_cond_model"](x, is_training=True)["x_prenorm"]
    hf_pre = tm(pixel_values=x, output_hidden_states=True).hidden_states[-1]
    d_hub = (hub_pre - hf_pre).abs().max().item()
    print(f"DINOv2 x_prenorm: torch.hub vs transformers max diff {d_hub:.3g} (max |x| {hub_pre.abs().max().item():.3g})")
    del tm

    noises, zs = [], []
    _randn = torch.randn

    def randn_rec(*a, **k):
        t = _randn(*a, **k)
        noises.append(t.clone())
        return t

    _sample = pipe.sparse_structure_sampler.sample

    def sample_rec(*a, **k):
        res = _sample(*a, **k)
        zs.append(res.samples)
        return res

    pipe.sparse_structure_sampler.sample = sample_rec
    torch.manual_seed(seed)
    torch.randn = randn_rec
    coords = pipe.sample_sparse_structure(cond, 1)
    torch.randn = _randn
    z = zs[0]
    logits = dec(z).float()

print("latent", tuple(z.shape), "logits", tuple(logits.shape), "occupied", coords.shape[0],
      "min |logit|", logits.abs().min().item())
save_file({"pixels": pixels.float().contiguous(), "cond": cond["cond"].float().cpu().contiguous(),
           "noise": noises[0].float().cpu().contiguous(), "latent": z.float().cpu().contiguous(),
           "logits": logits.cpu().contiguous(), "coords": coords.int().cpu().contiguous()},
          os.path.join(work, "ref.safetensors"))
json.dump({"seed": seed, "fp16": FP16, "device": dev, "sampler": ss, "occupied": int(coords.shape[0]),
           "hub_vs_transformers_prenorm": d_hub, "image": "object.png"},
          open(os.path.join(work, "ref.json"), "w"), indent=2)
print("wrote", os.path.join(work, "ref.safetensors"))
