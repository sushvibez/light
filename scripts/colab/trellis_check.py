#!/usr/bin/env python3
"""TRELLIS-image-large full-pipeline real-weight check (Colab GPU): image ->
sparse structure -> structured latent -> 3D Gaussians and mesh.

    python scripts/colab/trellis_check.py <work_dir> [image.png] [seed]

Run from a checkout of the repo (it imports scripts/verify/trellis_backends.py
for the backend cross-check and the fallbacks). Downloads into <work_dir>:
  TRELLIS-image-large/            pipeline.json and the five checkpoints of
                                  the Gaussian and mesh paths
                                  (ss_flow_img_dit_L_16l8, ss_dec_conv3d_16l8,
                                  slat_flow_img_dit_L_64l8p2,
                                  slat_dec_gs_swin8_B_64l8gs32,
                                  slat_dec_mesh_swin8_B_64l8m256c; ~2.8 GB)
  dinov2-with-registers-large/    the transformers port of dinov2_vitl14_reg
                                  (~1.2 GB), for Light
  trellis_src/                    TRELLIS's .py files the Gaussian and mesh
                                  paths import, FlexiCubes from the commit
                                  TRELLIS pins as a submodule (no git clone)
Backends: TRELLIS's real ones - spconv (SubMConv3d) and xformers
(memory_efficient_attention) - pip-installed here if missing (spconv-cu12x
matching torch's CUDA; xformers with --no-deps so torch is not replaced).
Each is first checked to run; if one cannot be installed or does not run,
the plain-PyTorch stand-in of scripts/verify/trellis_backends.py takes its
place (TRELLIS_BACKENDS=standin forces both) and ref.json says which ran.
With the real ones, the stand-ins are cross-checked against them on random
data (ref.json backend_check: a 3^3 SubMConv3d on 6000 voxels of 64^3 at
128 -> 128 channels, and block-diagonal attention), which validates the
CPU reference scripts/verify/ref_trellis_slat.py uses. SPCONV_ALGO=native,
NVIDIA_TF32_OVERRIDE=0 and torch's TF32 flags off keep the convolutions and
matmuls at F32.

Then runs TRELLIS's own TrellisImageTo3DPipeline methods as run() chains
them - preprocess_image, get_cond (torch.hub DINOv2), torch.manual_seed(seed)
(default 42), sample_sparse_structure, sample_slat, decode_slat(['gaussian'])
and decode_slat(['mesh']) - on <image.png> (RGBA with the object cut out;
default: a synthetic object drawn here, saved as object.png), the models in F32 (TRELLIS_FP16=1: their
native FP16 torsos). Records in <work_dir>/ref.safetensors:
  pixels, cond, noise, latent, logits, coords    stage 1, as trellis_ss_check.py
  slat_noise     (N, 8)        sample_slat's torch.randn draw
  slat_v0        (N, 8)        one SLat flow forward at t = 1000 on it (cond)
  slat_raw       (N, 8)        the sampler's output
  slat           (N, 8)        un-normalised by slat_normalization
  gs_feats       (N, 448)      the Gaussian decoder's out_layer output
  gs_xyz, gs_features_dc, gs_scaling, gs_rotation, gs_opacity   (32 N, *)
                               the Gaussian's get_xyz, _features_dc,
                               get_scaling, get_rotation, get_opacity
  gs_ply         (32 N, 17)    the rows Gaussian.save_ply writes (ref.ply)
  mesh_feats     (64 N, 101)   the mesh decoder's out_layer output (rows in
                               SparseSubdivide's order)
  mesh_vertices, mesh_faces, mesh_colors   its MeshExtractResult (vertices
                               (V, 3), faces (F, 3) I32, vertex_attrs (V, 6))
and ref.json (settings, counts, backends, backend_check). kaolin is not
installed: FlexiCubes only calls its check_tensor (argument asserts), which
trellis_backends.py provides.

TRELLIS_CHECK_CPU=1 is a plumbing dry run (CPU, stand-ins, no pip installs),
meant for tiny local models with the downloads patched out.

Then, with Light:
    LIGHT_TRELLIS_DIR=<work_dir> go test -p 1 -run 'TrellisReal|TrellisGaussiansReal|TrellisMeshReal' -v ./diffusion
(diffusion/trellis_real_test.go: stage 1; trellis_gs_real_test.go: stage 2
and the Gaussians, writing light.ply and light_views.png there;
trellis_mesh_real_test.go: the mesh decoder, writing light.obj).
"""
import importlib, json, os, subprocess, sys, types, urllib.request

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ["SPCONV_ALGO"] = "native"
os.environ["ATTN_BACKEND"] = "sdpa"  # TRELLIS's dense attention (stage 1)
os.environ["SPARSE_BACKEND"] = "spconv"
os.environ["SPARSE_ATTN_BACKEND"] = "xformers"

import numpy as np  # noqa: E402
import torch  # noqa: E402

work = sys.argv[1]
image_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
FP16 = os.environ.get("TRELLIS_FP16") == "1"
os.makedirs(work, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DRY = os.environ.get("TRELLIS_CHECK_CPU") == "1"  # plumbing dry run on tiny models: CPU, stand-ins
if DRY:
    os.environ["TRELLIS_BACKENDS"] = "standin"
elif not torch.cuda.is_available():
    sys.exit("trellis_check.py needs a CUDA GPU (spconv / xformers); the CPU reference is scripts/verify/ref_trellis_slat.py")
dev = "cpu" if DRY else "cuda"
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "verify"))
import trellis_backends  # noqa: E402


# --- backends ---------------------------------------------------------------------------
def pip(*args):
    print("pip install", *args)
    return subprocess.call([sys.executable, "-m", "pip", "install", "-q", *args]) == 0


def try_spconv():
    try:
        import spconv.pytorch as spc
        c = spc.SubMConv3d(4, 4, 3).cuda()
        idx = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 2]], dtype=torch.int32, device=dev)
        c(spc.SparseConvTensor(torch.randn(2, 4, device=dev), idx, [4, 4, 4], 1))
        return True
    except Exception as e:  # noqa: BLE001
        print("spconv unavailable:", repr(e)[:200])
        return False


def try_xformers():
    try:
        import xformers.ops as xo
        q = torch.randn(1, 8, 2, 16, device=dev)
        xo.memory_efficient_attention(q, q, q, xo.fmha.BlockDiagonalMask.from_seqlens([3, 5]))
        return True
    except Exception as e:  # noqa: BLE001
        print("xformers unavailable:", repr(e)[:200])
        return False


backends = {"spconv": "stand-in", "xformers": "stand-in"}
if os.environ.get("TRELLIS_BACKENDS") != "standin":
    if not try_spconv():
        cu = torch.version.cuda.replace(".", "")
        for wheel in [f"spconv-cu{cu}", "spconv-cu126", "spconv-cu124", "spconv-cu121", "spconv-cu120"]:
            if pip(wheel) and try_spconv():
                break
            subprocess.call([sys.executable, "-m", "pip", "uninstall", "-q", "-y", wheel])
    if try_spconv():
        backends["spconv"] = "spconv " + importlib.import_module("spconv").__version__
    if not try_xformers():
        pip("--no-deps", "xformers")
    if try_xformers():
        backends["xformers"] = "xformers " + importlib.import_module("xformers").__version__
    else:
        # A wheel that installs but cannot import its kernels (e.g. a missing
        # mslk) still makes DINOv2's hub code take its xformers branch
        # (`from xformers.ops import unbind` resolves to a module): remove it
        # and tell DINOv2 not to look.
        subprocess.call([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "xformers"])
        os.environ["XFORMERS_DISABLED"] = "1"
print("backends:", backends)

backend_check = {}
if backends["spconv"] != "stand-in":
    import spconv.pytorch as spc
    torch.manual_seed(0)
    flat = torch.randperm(64 ** 3, device=dev)[:6000].sort().values
    idx = torch.stack([torch.zeros_like(flat), flat // 4096, flat // 64 % 64, flat % 64], 1).int()
    f = torch.randn(6000, 128, device=dev)
    real = spc.SubMConv3d(128, 128, 3).cuda()
    stand = trellis_backends.SubMConv3d(128, 128, 3).cuda()
    with torch.no_grad():
        stand.weight.copy_(real.weight), stand.bias.copy_(real.bias)
        a = real(spc.SparseConvTensor(f, idx, [64, 64, 64], 1)).features
        b = stand(trellis_backends.SparseConvTensor(f, idx, [64, 64, 64], 1)).features
    backend_check["subm_conv3d_max_diff"] = (a - b).abs().max().item()
    backend_check["subm_conv3d_max_abs"] = a.abs().max().item()
    assert real.weight.shape == (128, 3, 3, 3, 128), real.weight.shape
if backends["xformers"] != "stand-in":
    import xformers.ops as xo
    torch.manual_seed(1)
    lens = [37, 1, 200, 64]
    q, k, v = (torch.randn(1, sum(lens), 12, 64, device=dev) for _ in range(3))
    m = xo.fmha.BlockDiagonalMask.from_seqlens(lens)
    a = xo.memory_efficient_attention(q, k, v, m)
    b = trellis_backends.memory_efficient_attention(q, k, v, trellis_backends.BlockDiagonalMask.from_seqlens(lens))
    backend_check["block_diag_attention_max_diff"] = (a - b).abs().max().item()
print("backend check:", backend_check)
trellis_backends.install(spconv=backends["spconv"] == "stand-in", xformers=backends["xformers"] == "stand-in",
                         utils3d=False)
try:
    import kaolin.utils.testing  # noqa: F401
except ImportError:
    trellis_backends.install(spconv=False, xformers=False, utils3d=False, kaolin=True)
try:
    import utils3d  # noqa: F401
except ImportError:
    if DRY or not pip("git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"):
        trellis_backends.install(spconv=False, xformers=False, utils3d=True)
for mod in ("plyfile", "easydict"):
    try:
        importlib.import_module(mod)
    except ImportError:
        if mod == "easydict" and DRY:
            ed = types.ModuleType("easydict")

            class EasyDict(dict):
                __getattr__ = dict.__getitem__

                def __setattr__(self, k, v):
                    self[k] = v

            ed.EasyDict = EasyDict
            sys.modules["easydict"] = ed
        else:
            pip(mod)

# --- downloads --------------------------------------------------------------------------
from huggingface_hub import snapshot_download  # noqa: E402

ck = ["ss_flow_img_dit_L_16l8_fp16", "ss_dec_conv3d_16l8_fp16", "slat_flow_img_dit_L_64l8p2_fp16",
      "slat_dec_gs_swin8_B_64l8gs32_fp16", "slat_dec_mesh_swin8_B_64l8m256c_fp16"]
root = snapshot_download("microsoft/TRELLIS-image-large", local_dir=os.path.join(work, "TRELLIS-image-large"),
                         allow_patterns=["pipeline.json"] + [f"ckpts/{c}.*" for c in ck])
dino_dir = snapshot_download("facebook/dinov2-with-registers-large",
                             local_dir=os.path.join(work, "dinov2-with-registers-large"),
                             allow_patterns=["config.json", "model.safetensors"])
src = os.path.join(work, "trellis_src")
base = "https://raw.githubusercontent.com/microsoft/TRELLIS/main/"
for f in """trellis/pipelines/trellis_image_to_3d.py trellis/pipelines/base.py trellis/pipelines/samplers/__init__.py
trellis/pipelines/samplers/base.py trellis/pipelines/samplers/flow_euler.py
trellis/pipelines/samplers/guidance_interval_mixin.py trellis/pipelines/samplers/classifier_free_guidance_mixin.py
trellis/models/sparse_structure_flow.py trellis/models/sparse_structure_vae.py
trellis/models/structured_latent_flow.py trellis/models/sparse_elastic_mixin.py
trellis/models/structured_latent_vae/base.py trellis/models/structured_latent_vae/decoder_gs.py
trellis/modules/norm.py trellis/modules/utils.py trellis/modules/spatial.py trellis/modules/attention/__init__.py
trellis/modules/attention/full_attn.py trellis/modules/attention/modules.py trellis/modules/transformer/__init__.py
trellis/modules/transformer/blocks.py trellis/modules/transformer/modulated.py
trellis/modules/sparse/__init__.py trellis/modules/sparse/basic.py trellis/modules/sparse/linear.py
trellis/modules/sparse/nonlinearity.py trellis/modules/sparse/norm.py trellis/modules/sparse/spatial.py
trellis/modules/sparse/conv/__init__.py trellis/modules/sparse/conv/conv_spconv.py
trellis/modules/sparse/attention/__init__.py trellis/modules/sparse/attention/full_attn.py
trellis/modules/sparse/attention/serialized_attn.py trellis/modules/sparse/attention/windowed_attn.py
trellis/modules/sparse/attention/modules.py trellis/modules/sparse/transformer/__init__.py
trellis/modules/sparse/transformer/blocks.py trellis/modules/sparse/transformer/modulated.py
trellis/representations/gaussian/__init__.py trellis/representations/gaussian/gaussian_model.py
trellis/representations/gaussian/general_utils.py trellis/utils/__init__.py trellis/utils/random_utils.py
trellis/utils/elastic_utils.py trellis/models/structured_latent_vae/decoder_mesh.py
trellis/representations/mesh/__init__.py trellis/representations/mesh/cube2mesh.py
trellis/representations/mesh/utils_cube.py""".split() + [
        "https://raw.githubusercontent.com/MaxtirError/FlexiCubes/815e075a2a400d06c48d94c347674344ed6ae5c5/" + f
        for f in ("flexicubes.py", "tables.py")]:
    url = f if f.startswith("https://") else base + f
    if f.startswith("https://"):
        f = "trellis/representations/mesh/flexicubes/" + f.rsplit("/", 1)[1]
    p = os.path.join(src, f)
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(urllib.request.urlopen(url, timeout=60).read())


# --- TRELLIS's modules without the heavy package __init__s -----------------------------------
def stub_package(name, path):
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m
    return m


sys.path.insert(0, src)
T = os.path.join(src, "trellis")
stub_package("trellis", T)
stub_package("trellis.pipelines", os.path.join(T, "pipelines"))
stub_package("trellis.modules", os.path.join(T, "modules"))
stub_package("trellis.models", os.path.join(T, "models"))
stub_package("trellis.models.structured_latent_vae", os.path.join(T, "models", "structured_latent_vae"))
reps = stub_package("trellis.representations", os.path.join(T, "representations"))
sys.modules.setdefault("rembg", types.ModuleType("rembg"))
from trellis.representations.gaussian import Gaussian  # noqa: E402

reps.Gaussian = Gaussian
from trellis.representations.mesh import cube2mesh  # noqa: E402

reps.MeshExtractResult = cube2mesh.MeshExtractResult
if DRY:  # SparseFeatures2Mesh builds its grids on "cuda" by default
    _s2m = cube2mesh.SparseFeatures2Mesh.__init__
    cube2mesh.SparseFeatures2Mesh.__init__ = lambda self, device="cpu", res=64, use_color=True: _s2m(
        self, device=device, res=res, use_color=use_color)
if DRY:
    trellis_backends.cpu_gaussian(Gaussian)
from PIL import Image, ImageDraw  # noqa: E402
from plyfile import PlyData  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from trellis.models.sparse_structure_flow import SparseStructureFlowModel  # noqa: E402
from trellis.models.sparse_structure_vae import SparseStructureDecoder  # noqa: E402
from trellis.models.structured_latent_flow import SLatFlowModel  # noqa: E402
from trellis.models.structured_latent_vae.decoder_gs import SLatGaussianDecoder  # noqa: E402
from trellis.models.structured_latent_vae.decoder_mesh import SLatMeshDecoder  # noqa: E402
from trellis.modules import sparse as sp  # noqa: E402
from trellis.pipelines import samplers  # noqa: E402
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline  # noqa: E402

args = json.load(open(os.path.join(root, "pipeline.json")))["args"]
classes = {"SparseStructureFlowModel": SparseStructureFlowModel, "SparseStructureDecoder": SparseStructureDecoder,
           "SLatFlowModel": SLatFlowModel, "SLatGaussianDecoder": SLatGaussianDecoder,
           "SLatMeshDecoder": SLatMeshDecoder}


def load(name):
    """trellis.models.from_pretrained, built in F32 (use_fp16 off: the FP16
    torsos are widened as load_state_dict copies them) unless TRELLIS_FP16."""
    path = os.path.join(root, args["models"][name])
    cfg = json.load(open(path + ".json"))
    a = dict(cfg["args"])
    if not FP16:
        a["use_fp16"] = False
    m = classes[cfg["name"]](**a)
    m.load_state_dict(load_file(path + ".safetensors"))
    return m.to(dev).eval()


pipe = TrellisImageTo3DPipeline()
pipe.models = {n: load(n) for n in ("sparse_structure_flow_model", "sparse_structure_decoder", "slat_flow_model",
                                    "slat_decoder_gs", "slat_decoder_mesh")}
pipe._init_image_cond_model(args["image_cond_model"])  # torch.hub, as TRELLIS
pipe.models["image_cond_model"].to(dev)
for k in ("sparse_structure_sampler", "slat_sampler"):
    s = args[k]
    setattr(pipe, k, getattr(samplers, s["name"])(**s["args"]))
    setattr(pipe, k + "_params", s["params"])
pipe.slat_normalization = args["slat_normalization"]

# --- the image ---------------------------------------------------------------------------------
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

out = {}
with torch.no_grad():
    pre = pipe.preprocess_image(image)
    pre.save(os.path.join(work, "preprocessed.png"))
    cond = pipe.get_cond([pre])
    pixels = torch.from_numpy(np.array(pre.resize((518, 518), Image.LANCZOS).convert("RGB")).astype(np.float32) / 255)
    out["pixels"] = pixels.permute(2, 0, 1).contiguous()

    noises, zs = [], {}
    _randn = torch.randn

    def randn_rec(*a, **k):
        t = _randn(*a, **k)
        noises.append(t.clone())
        return t

    def recorder(sampler, key):
        orig = sampler.sample

        def sample(*a, **k):
            res = orig(*a, **k)
            zs[key] = res.samples
            return res
        sampler.sample = sample

    recorder(pipe.sparse_structure_sampler, "ss")
    recorder(pipe.slat_sampler, "slat")
    feats = {}
    pipe.models["slat_decoder_gs"].out_layer.register_forward_hook(lambda m, i, o: feats.__setitem__("h", o.feats.clone()))
    pipe.models["slat_decoder_mesh"].out_layer.register_forward_hook(
        lambda m, i, o: feats.__setitem__("mesh", o.feats.clone()))
    torch.manual_seed(seed)
    torch.randn = randn_rec
    coords = pipe.sample_sparse_structure(cond, 1)
    slat = pipe.sample_slat(cond, coords)
    torch.randn = _randn
    gs = pipe.decode_slat(slat, ["gaussian"])["gaussian"][0]
    mesh = pipe.decode_slat(slat, ["mesh"])["mesh"][0]
    logits = pipe.models["sparse_structure_decoder"](zs["ss"]).float()
    v0 = pipe.models["slat_flow_model"](sp.SparseTensor(feats=noises[1].to(dev), coords=coords),
                                        torch.tensor([1000.0], device=dev), cond["cond"])

    f32 = lambda t: t.detach().float().cpu().contiguous()  # noqa: E731
    out.update({"cond": f32(cond["cond"]), "noise": f32(noises[0]), "latent": f32(zs["ss"]), "logits": f32(logits),
                "coords": coords.int().cpu().contiguous(), "slat_noise": f32(noises[1]), "slat_v0": f32(v0.feats),
                "slat_raw": f32(zs["slat"].feats), "slat": f32(slat.feats), "gs_feats": f32(feats["h"]),
                "gs_xyz": f32(gs.get_xyz), "gs_features_dc": f32(gs._features_dc.reshape(-1, 3)),
                "gs_scaling": f32(gs.get_scaling), "gs_rotation": f32(gs.get_rotation),
                "gs_opacity": f32(gs.get_opacity), "mesh_feats": f32(feats["mesh"]),
                "mesh_vertices": f32(mesh.vertices), "mesh_faces": mesh.faces.int().cpu().contiguous(),
                "mesh_colors": f32(mesh.vertex_attrs)})
assert torch.equal(slat.coords, coords)
ply = os.path.join(work, "ref.ply")
gs.save_ply(ply)
with open(ply, "rb") as fh:
    v = PlyData.read(fh).elements[0]
    out["gs_ply"] = torch.from_numpy(np.stack([np.array(v[p.name], dtype=np.float32) for p in v.properties], 1))
print("occupied", coords.shape[0], "slat |x| max", slat.feats.abs().max().item(), "gaussians", gs.get_xyz.shape[0],
      "opacity", gs.get_opacity.min().item(), gs.get_opacity.max().item(), "mesh", mesh.vertices.shape[0], "vertices,",
      mesh.faces.shape[0], "faces")
save_file(out, os.path.join(work, "ref.safetensors"))
json.dump({"seed": seed, "fp16": FP16, "device": dev, "sampler": args["sparse_structure_sampler"],
           "slat_sampler": args["slat_sampler"], "occupied": int(coords.shape[0]),
           "gaussians": int(gs.get_xyz.shape[0]), "mesh_vertices": int(mesh.vertices.shape[0]),
           "mesh_faces": int(mesh.faces.shape[0]), "backends": backends, "backend_check": backend_check,
           "spconv_algo": os.environ["SPCONV_ALGO"], "image": "object.png"},
          open(os.path.join(work, "ref.json"), "w"), indent=2)
print("wrote", os.path.join(work, "ref.safetensors"), "and ref.ply")
