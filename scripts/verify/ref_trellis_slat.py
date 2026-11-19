#!/usr/bin/env python3
"""Tiny TRELLIS stage-2 reference (microsoft/TRELLIS, CPU, F32) for
models/trellis_slat_test.go and diffusion/trellis_test.go.

    python -I scripts/verify/ref_trellis_slat.py <trellis src> <tiny dir> <stage1.json> <out.safetensors>

<trellis src> holds TRELLIS's trellis/ package: the stage-1 files
ref_trellis_ss.py imports plus models/structured_latent_flow.py,
models/sparse_elastic_mixin.py, models/structured_latent_vae/{base,
decoder_gs}.py, modules/sparse/** (basic, linear, nonlinearity, norm,
spatial, conv/, attention/, transformer/), representations/gaussian/*,
utils/{random_utils,elastic_utils}.py. <tiny dir> is
models/testdata/trellis-tiny (stage 1's tiny checkpoints, written by
ref_trellis_ss.py; <stage1.json> is its trellis-tiny.json).

TRELLIS's sparse modules run unchanged with SPARSE_BACKEND=spconv and
SPARSE_ATTN_BACKEND=xformers; the compiled spconv / xformers calls (and
utils3d's two quaternion helpers, and Gaussian's .cuda() calls) are
replaced by the plain-PyTorch stand-ins of scripts/verify/trellis_backends.py,
which documents each one. The dense modules (TimestepEmbedder, the
AbsolutePositionEmbedder) run on the sdpa backend as in stage 1.

Builds seeded tiny versions of
  - SLatFlowModel: resolution 16, 8 latent channels, io_block_channels [16]
    (two SparseResBlock3d in, downsample by 2, two out with skips), 2
    ModulatedSparseTransformerCrossBlocks of 64 channels / 4 heads with qk
    RMS norm, APE, cross attention to 64-wide tokens;
  - SLatGaussianDecoder: resolution 16, 64 channels, 2 SparseTransformerBlocks
    with attn_mode swin and window 4 (block 1 shifted by 2), 4 Gaussians per
    voxel, the released representation_config otherwise,
saved as ckpts/<name>.json + .safetensors with the torsos (input_blocks /
blocks / out_blocks; blocks) in F16 as the released *_fp16 checkpoints, and
pipeline.json extended with slat_flow_model, slat_decoder_gs, slat_sampler
and the released slat_normalization. Records with TRELLIS's own code:
  flow.*    one SLatFlowModel forward on 300 voxels in random order (t 637.5)
  sample.*  run()'s stage 2 for stage 1's tokens: torch.manual_seed(42),
            sample_sparse_structure (its coords must equal stage 1's
            fixture), then sample_slat (25 Euler steps, CFG 5 in [0.5, 1],
            rescale_t 3) from the recorded torch.randn noise: the raw sample
            and the un-normalised SLat
  dec.feats the decoder's out_layer output (before to_representation)
  gs.*      decode_slat's Gaussian: get_xyz, _features_dc, get_scaling,
            get_rotation, get_opacity and the save_ply rows (17 floats per
            Gaussian, read back with plyfile)
"""
import json, os, sys, tempfile, types

import numpy as np
import torch

trellis_src, tiny_dir, stage1_json, out_path = sys.argv[1:5]
os.environ["ATTN_BACKEND"] = "sdpa"
os.environ["SPARSE_BACKEND"] = "spconv"
os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
os.environ["XFORMERS_DISABLED"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trellis_backends  # noqa: E402

trellis_backends.install(spconv=True, xformers=True, utils3d=True)


def stub_package(name, path):
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m
    return m


sys.path.insert(0, trellis_src)
T = os.path.join(trellis_src, "trellis")
stub_package("trellis", T)
stub_package("trellis.pipelines", os.path.join(T, "pipelines"))
stub_package("trellis.modules", os.path.join(T, "modules"))
stub_package("trellis.models", os.path.join(T, "models"))
stub_package("trellis.models.structured_latent_vae", os.path.join(T, "models", "structured_latent_vae"))
reps = stub_package("trellis.representations", os.path.join(T, "representations"))
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

from trellis.representations.gaussian import Gaussian  # noqa: E402

reps.Gaussian = Gaussian
trellis_backends.cpu_gaussian(Gaussian)

from plyfile import PlyData  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from trellis.models.sparse_structure_flow import SparseStructureFlowModel  # noqa: E402
from trellis.models.sparse_structure_vae import SparseStructureDecoder  # noqa: E402
from trellis.models.structured_latent_flow import SLatFlowModel  # noqa: E402
from trellis.models.structured_latent_vae.decoder_gs import SLatGaussianDecoder  # noqa: E402
from trellis.modules import sparse as sp  # noqa: E402
from trellis.pipelines import samplers  # noqa: E402
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline  # noqa: E402

assert sp.BACKEND == "spconv" and sp.ATTN == "xformers", (sp.BACKEND, sp.ATTN)
ref1 = json.load(open(stage1_json))
pipe_path = os.path.join(tiny_dir, "pipeline.json")
pipe_cfg = json.load(open(pipe_path))


def load_ckpt(name):
    cfg = json.load(open(os.path.join(tiny_dir, name + ".json")))
    return cfg, {k: v.float() for k, v in load_file(os.path.join(tiny_dir, name + ".safetensors")).items()}


# --- stage 1 from its fixture ---------------------------------------------------------
models1 = pipe_cfg["args"]["models"]
cfg, sd = load_ckpt(models1["sparse_structure_flow_model"])
ss_flow = SparseStructureFlowModel(**{**cfg["args"], "use_fp16": False})
ss_flow.load_state_dict(sd)
cfg, sd = load_ckpt(models1["sparse_structure_decoder"])
ss_dec = SparseStructureDecoder(**{**cfg["args"], "use_fp16": False})
ss_dec.load_state_dict(sd)
ss_flow.eval(), ss_dec.eval()

# --- the tiny stage-2 models ------------------------------------------------------------
D = ref1["dinov2"]["shape"][-1]
flow_args = {"resolution": 16, "in_channels": 8, "out_channels": 8, "model_channels": 64, "cond_channels": D,
             "num_blocks": 2, "num_heads": 4, "mlp_ratio": 4, "patch_size": 2, "num_io_res_blocks": 2,
             "io_block_channels": [16], "pe_mode": "ape", "qk_rms_norm": True, "use_fp16": True}
rep_cfg = {"lr": {"_xyz": 1.0, "_features_dc": 1.0, "_opacity": 1.0, "_scaling": 1.0, "_rotation": 0.1},
           "perturb_offset": True, "voxel_size": 1.5, "num_gaussians": 4, "2d_filter_kernel_size": 0.1,
           "3d_filter_kernel_size": 9e-4, "scaling_bias": 4e-3, "opacity_bias": 0.1, "scaling_activation": "softplus"}
dec_args = {"resolution": 16, "model_channels": 64, "latent_channels": 8, "num_blocks": 2, "num_heads": 4,
            "mlp_ratio": 4, "attn_mode": "swin", "window_size": 4, "use_fp16": True,
            "representation_config": rep_cfg}


def randomize(model, seed, torso):
    """Seeded weights at fan-in scale; the torso (FP16 in the released
    checkpoints, norms excepted) rounded to F16 so file and model agree."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "gamma" in name:
                p.copy_(1 + 0.2 * torch.randn_like(p))
            elif "norm" in name and name.endswith("weight"):
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
    save_file(sd, os.path.join(tiny_dir, "ckpts", name + ".safetensors"))
    json.dump({"name": cfg_name, "args": args}, open(os.path.join(tiny_dir, "ckpts", name + ".json"), "w"), indent=4)
    model.load_state_dict({k: v.float() for k, v in load_file(os.path.join(tiny_dir, "ckpts", name + ".safetensors")).items()})


flow_torso = ("input_blocks.", "blocks.", "out_blocks.")
flow = randomize(SLatFlowModel(**{**flow_args, "use_fp16": False}), 11, flow_torso)
save_ckpt(flow, "slat_flow_tiny_fp16", "SLatFlowModel", flow_args, flow_torso)
dec = randomize(SLatGaussianDecoder(**{**dec_args, "use_fp16": False}), 12, ("blocks.",))
with torch.no_grad():
    dec.out_layer.weight.mul_(0.5)  # keep the Gaussian parameters in a typical range
save_ckpt(dec, "slat_dec_gs_tiny_fp16", "SLatGaussianDecoder", dec_args, ("blocks.",))
flow.eval(), dec.eval()

released = {  # microsoft/TRELLIS-image-large pipeline.json
    "mean": [-2.1687545776367188, -0.004347046371549368, -0.13352349400520325, -0.08418072760105133,
             -0.5271206498146057, 0.7238689064979553, -1.1414450407028198, 1.2039363384246826],
    "std": [2.377650737762451, 2.386378288269043, 2.124418020248413, 2.1748552322387695,
            2.663944721221924, 2.371192216873169, 2.6217446327209473, 2.684523105621338]}
a = pipe_cfg["args"]
a["models"]["slat_flow_model"] = "ckpts/slat_flow_tiny_fp16"
a["models"]["slat_decoder_gs"] = "ckpts/slat_dec_gs_tiny_fp16"
a["slat_sampler"] = {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-5},
                     "params": {"steps": 25, "cfg_strength": 5.0, "cfg_interval": [0.5, 1.0], "rescale_t": 3.0}}
a["slat_normalization"] = released
json.dump(pipe_cfg, open(pipe_path, "w"), indent=4)

out = {}
f32 = lambda t: t.detach().float().contiguous()  # noqa: E731

# --- one flow forward on unsorted coords ------------------------------------------------
torch.manual_seed(21)
box = torch.stack(torch.meshgrid(*[torch.arange(3, 13)] * 3, indexing="ij"), -1).reshape(-1, 3)
pick = torch.randperm(box.shape[0])[:300]
fc = torch.cat([torch.zeros(300, 1, dtype=torch.int32), box[pick].int()], 1)
fx = torch.randn(300, 8)
fcond = torch.randn(1, 21, D)
with torch.no_grad():
    fv = flow(sp.SparseTensor(feats=fx, coords=fc), torch.tensor([637.5]), fcond)
assert torch.equal(fv.coords, fc)
out.update({"flow.coords": fc, "flow.x": fx, "flow.t": torch.tensor([637.5]), "flow.cond": fcond, "flow.v": f32(fv.feats)})
print("flow forward: 300 voxels, |v| max", fv.feats.abs().max().item())

# --- run()'s stages 1 and 2 ----------------------------------------------------------------
pipe = TrellisImageTo3DPipeline()
pipe.models = {"sparse_structure_flow_model": ss_flow, "sparse_structure_decoder": ss_dec,
               "slat_flow_model": flow, "slat_decoder_gs": dec}
for k in ("sparse_structure_sampler", "slat_sampler"):
    s = a[k]
    setattr(pipe, k, getattr(samplers, s["name"])(**s["args"]))
    setattr(pipe, k + "_params", s["params"])
pipe.slat_normalization = a["slat_normalization"]
cond = torch.tensor(ref1["dinov2"]["tokens"]).reshape(ref1["dinov2"]["shape"])
cond = {"cond": cond, "neg_cond": torch.zeros_like(cond)}

noises, raw = [], {}
_randn = torch.randn


def randn_rec(*a, **k):
    t = _randn(*a, **k)
    noises.append(t.clone())
    return t


_sample = pipe.slat_sampler.sample


def sample_rec(*a, **k):
    res = _sample(*a, **k)
    raw["z"] = res.samples
    return res


pipe.slat_sampler.sample = sample_rec
dec_out = {}
dec.out_layer.register_forward_hook(lambda m, i, o: dec_out.__setitem__("h", o.feats.clone()))
torch.manual_seed(42)  # run()'s default seed
torch.randn = randn_rec
with torch.no_grad():
    coords = pipe.sample_sparse_structure(cond, 1)
    slat = pipe.sample_slat(cond, coords)
torch.randn = _randn
want = torch.tensor(ref1["sample"]["coords"], dtype=torch.int32).reshape(-1, 4)
assert torch.equal(coords, want), "stage-1 coords differ from the fixture"
assert torch.allclose(noises[0], torch.tensor(ref1["sample"]["noise"]).reshape(noises[0].shape), atol=1e-6), "stage-1 noise differs"
assert torch.equal(slat.coords, coords)
with torch.no_grad():
    gs = pipe.decode_slat(slat, ["gaussian"])["gaussian"][0]
N = coords.shape[0]
print("stage 2:", N, "voxels, raw |z| max", raw["z"].feats.abs().max().item(), "slat |x| max",
      slat.feats.abs().max().item(), "decoder |h| max", dec_out["h"].abs().max().item())
out.update({"sample.coords": coords.int().contiguous(), "sample.noise": f32(noises[1]), "sample.raw": f32(raw["z"].feats),
            "sample.slat": f32(slat.feats), "dec.feats": f32(dec_out["h"])})
gfields = {"gs.xyz": gs.get_xyz, "gs.features_dc": gs._features_dc.reshape(-1, 3), "gs.scaling": gs.get_scaling,
           "gs.rotation": gs.get_rotation, "gs.opacity": gs.get_opacity}
out.update({k: f32(v) for k, v in gfields.items()})
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
    p = os.path.join(td, "g.ply")
    gs.save_ply(p)
    with open(p, "rb") as fh:
        v = PlyData.read(fh).elements[0]
        names = [pp.name for pp in v.properties]
        rows = np.stack([np.array(v[n], dtype=np.float32) for n in names], 1)
    del v
assert names == ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
                 "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"], names
out["gs.ply"] = torch.from_numpy(rows).contiguous()
print("gaussians:", rows.shape[0], "opacity", gs.get_opacity.min().item(), gs.get_opacity.max().item(),
      "scale", gs.get_scaling.min().item(), gs.get_scaling.max().item())
save_file(out, out_path)
print("wrote", out_path, os.path.getsize(out_path), "bytes")
