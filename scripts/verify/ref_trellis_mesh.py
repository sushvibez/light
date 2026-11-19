#!/usr/bin/env python3
"""Tiny TRELLIS mesh-decoder reference (microsoft/TRELLIS, CPU, F32) for
models/trellis_mesh_test.go and diffusion/trellis_mesh_test.go.

    python -I scripts/verify/ref_trellis_mesh.py <trellis src> <tiny dir> <slat ref> <out.safetensors>

<trellis src> holds TRELLIS's trellis/ package as ref_trellis_slat.py needs
it plus models/structured_latent_vae/decoder_mesh.py and
representations/mesh/{__init__,cube2mesh,utils_cube}.py with the FlexiCubes
submodule (MaxtirError/FlexiCubes@815e075: flexicubes.py, tables.py) in
representations/mesh/flexicubes/. <tiny dir> is models/testdata/trellis-tiny;
<slat ref> is models/testdata/trellis-tiny-slat.safetensors (its sample.*
SLat feeds the pipeline-scale check).

The sparse backends are scripts/verify/trellis_backends.py's stand-ins, as
in ref_trellis_slat.py, plus kaolin's check_tensor (FlexiCubes' argument
asserts). SparseFeatures2Mesh defaults to device "cuda"; it is built on the
CPU here.

Builds a seeded tiny SLatMeshDecoder: resolution 16, 256 channels (the
least SparseSubdivideBlock3d's 32-group norms allow at model_channels / 8),
2 swin blocks of 4 heads with window 4 (block 1 shifted by 2), mlp_ratio 1,
use_color (101 outputs per cube: 8 sdf, 8 x 3 deform, 21 weights, 8 x 6
colour), FlexiCubes at 64^3. Saved as ckpts/slat_dec_mesh_tiny_fp16 with
the FP16_MODULES of blocks / upsample in F16 as the released checkpoint;
pipeline.json gains slat_decoder_mesh. Records with TRELLIS's own code:
  mesh.coords, mesh.x  20 voxels in random order, an N(0, 2²) latent
  mesh.torso           the transformer's output (before the upsampling)
  mesh.up1, mesh.up2   the two SparseSubdivideBlock3d outputs (rows in
                       SparseSubdivide's order: voxel-major, 8 children)
  mesh.feats           out_layer's output (64 N, 101)
  mesh.vertices, mesh.faces, mesh.colors   the MeshExtractResult
  pipe.counts, pipe.moments   on the tiny pipeline's SLat (sample.*: 1186
                       voxels): vertex / face counts, and the sums of the
                       vertices, colours and face indices
"""
import json, os, sys, types

import numpy as np
import torch

trellis_src, tiny_dir, slat_ref, out_path = sys.argv[1:5]
os.environ["ATTN_BACKEND"] = "sdpa"
os.environ["SPARSE_BACKEND"] = "spconv"
os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
os.environ["XFORMERS_DISABLED"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trellis_backends  # noqa: E402

trellis_backends.install(spconv=True, xformers=True, utils3d=True, kaolin=True)


def stub_package(name, path):
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m
    return m


sys.path.insert(0, trellis_src)
T = os.path.join(trellis_src, "trellis")
stub_package("trellis", T)
stub_package("trellis.modules", os.path.join(T, "modules"))
stub_package("trellis.models", os.path.join(T, "models"))
stub_package("trellis.models.structured_latent_vae", os.path.join(T, "models", "structured_latent_vae"))
reps = stub_package("trellis.representations", os.path.join(T, "representations"))
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

from trellis.representations.mesh import cube2mesh  # noqa: E402

reps.MeshExtractResult = cube2mesh.MeshExtractResult
_init = cube2mesh.SparseFeatures2Mesh.__init__
cube2mesh.SparseFeatures2Mesh.__init__ = lambda self, device="cpu", res=64, use_color=True: _init(
    self, device=device, res=res, use_color=use_color)

from safetensors.torch import load_file, save_file  # noqa: E402
from trellis.models.structured_latent_vae.decoder_mesh import SLatMeshDecoder  # noqa: E402
from trellis.modules import sparse as sp  # noqa: E402
from trellis.modules.utils import FP16_MODULES  # noqa: E402

assert sp.BACKEND == "spconv" and sp.ATTN == "xformers", (sp.BACKEND, sp.ATTN)

args = {"resolution": 16, "model_channels": 256, "latent_channels": 8, "num_blocks": 2, "num_heads": 4,
        "mlp_ratio": 1, "attn_mode": "swin", "window_size": 4, "use_fp16": True,
        "representation_config": {"use_color": True}}
dec = SLatMeshDecoder(**{**args, "use_fp16": False})
half = set()  # parameters convert_to_fp16 turns to F16
for mname, mod in dec.named_modules():
    if mname.startswith(("blocks.", "upsample.")) and isinstance(mod, FP16_MODULES):
        half.update(f"{mname}.{p}" for p, _ in mod.named_parameters())
torch.manual_seed(13)
with torch.no_grad():
    for name, p in dec.named_parameters():
        if "gamma" in name:
            p.copy_(1 + 0.2 * torch.randn_like(p))
        elif p.dim() == 1 and ("act_layers" in name or ".out_layers.1." in name):  # GroupNorm affine
            p.copy_((1 if name.endswith("weight") else 0) + 0.1 * torch.randn_like(p))
        elif p.dim() >= 2:
            p.normal_(0, 1 / np.sqrt(p[0].numel()))
        else:
            p.normal_(0, 0.05)
        if name in half:
            p.copy_(p.half().float())
assert any(".conv.weight" in n for n in half) and "out_layer.weight" not in half
sd = {k: (v.half() if k in half else v.float()).contiguous() for k, v in dec.state_dict().items()}
name = "slat_dec_mesh_tiny_fp16"
save_file(sd, os.path.join(tiny_dir, "ckpts", name + ".safetensors"))
json.dump({"name": "SLatMeshDecoder", "args": args}, open(os.path.join(tiny_dir, "ckpts", name + ".json"), "w"),
          indent=4)
dec.load_state_dict({k: v.float() for k, v in load_file(os.path.join(tiny_dir, "ckpts", name + ".safetensors")).items()})
dec.eval()
pipe_path = os.path.join(tiny_dir, "pipeline.json")
pipe_cfg = json.load(open(pipe_path))
pipe_cfg["args"]["models"]["slat_decoder_mesh"] = "ckpts/" + name
json.dump(pipe_cfg, open(pipe_path, "w"), indent=4)

# The stand-in's lookup path (used at 256^3) against its dense one on this
# model's first convolution.
conv = dec.upsample[0].out_layers[0].conv
torch.manual_seed(5)
ci = torch.randperm(32 ** 3)[:3000]
cidx = torch.stack([torch.zeros_like(ci), ci // 1024, ci // 32 % 32, ci % 32], 1).int()
cf = torch.randn(3000, conv.in_channels)
with torch.no_grad():
    a = conv(trellis_backends.SparseConvTensor(cf, cidx, [32] * 3, 1)).features
    b = conv._gather(cidx.long(), cf, [32] * 3, 1)
print("stand-in conv, lookup vs dense: max diff", (a - b).abs().max().item(), "max |a|", a.abs().max().item())
assert (a - b).abs().max() < 1e-4

hooks = {}
dec.blocks[-1].register_forward_hook(lambda m, i, o: hooks.__setitem__("torso", o.feats.clone()))
dec.upsample[0].register_forward_hook(lambda m, i, o: hooks.__setitem__("up1", (o.feats.clone(), o.coords.clone())))
dec.upsample[1].register_forward_hook(lambda m, i, o: hooks.__setitem__("up2", (o.feats.clone(), o.coords.clone())))
dec.out_layer.register_forward_hook(lambda m, i, o: hooks.__setitem__("feats", o.feats.clone()))
f32 = lambda t: t.detach().float().contiguous()  # noqa: E731

torch.manual_seed(31)
box = torch.stack(torch.meshgrid(*[torch.arange(6, 10)] * 3, indexing="ij"), -1).reshape(-1, 3)
pick = torch.randperm(box.shape[0])[:20]
coords = torch.cat([torch.zeros(20, 1, dtype=torch.int32), box[pick].int()], 1)
x = 2 * torch.randn(20, 8)
with torch.no_grad():
    mesh = dec(sp.SparseTensor(feats=x, coords=coords))[0]
want = (coords[:, None, 1:] * 4 + torch.stack(torch.meshgrid(*[torch.arange(4)] * 3, indexing="ij"), -1)
        .reshape(1, 64, 3)).reshape(-1, 3)
c2 = hooks["up2"][1][:, 1:]
assert c2.shape == want.shape and set(map(tuple, c2.tolist())) == set(map(tuple, want.tolist()))
assert mesh.success
print("decoder: 20 voxels ->", hooks["feats"].shape[0], "cubes; |feats| max", hooks["feats"].abs().max().item(),
      "; mesh", mesh.vertices.shape[0], "vertices,", mesh.faces.shape[0], "faces")
out = {"mesh.coords": coords.contiguous(), "mesh.x": f32(x), "mesh.torso": f32(hooks["torso"]),
       "mesh.up1": f32(hooks["up1"][0]), "mesh.up2": f32(hooks["up2"][0]), "mesh.up2_coords": hooks["up2"][1].int(),
       "mesh.feats": f32(hooks["feats"]), "mesh.vertices": f32(mesh.vertices),
       "mesh.faces": mesh.faces.int().contiguous(), "mesh.colors": f32(mesh.vertex_attrs)}

ref = load_file(slat_ref)
with torch.no_grad():
    m2 = dec(sp.SparseTensor(feats=ref["sample.slat"], coords=ref["sample.coords"]))[0]
v, fc, col = m2.vertices.double(), m2.faces, m2.vertex_attrs.double()
out["pipe.counts"] = torch.tensor([v.shape[0], fc.shape[0]], dtype=torch.int32)
out["pipe.moments"] = torch.cat([v.sum(0), col.sum(0), fc.double().sum().reshape(1)]).float()
print("pipeline SLat:", ref["sample.coords"].shape[0], "voxels ->", v.shape[0], "vertices,", fc.shape[0], "faces")
save_file(out, out_path)
print("wrote", out_path, os.path.getsize(out_path), "bytes")
