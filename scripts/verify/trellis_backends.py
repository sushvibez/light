"""Plain-PyTorch stand-ins for the compiled backends TRELLIS's sparse
modules call, so TRELLIS's own module code runs on a CPU (imported by
ref_trellis_slat.py and, as a fallback, scripts/colab/trellis_check.py).

Call install(spconv=True, xformers=True, utils3d=True) before importing
trellis.modules.sparse. Each substitution covers exactly the calls TRELLIS
makes in the stage-2 Gaussian path (SPARSE_BACKEND=spconv,
SPARSE_ATTN_BACKEND=xformers); anything else raises.

spconv.pytorch (trellis/modules/sparse/{basic,conv/conv_spconv}.py)
  SparseConvTensor   a plain holder of what basic.py reads and writes:
                     features (property over _features), indices (N, 4)
                     (b, x, y, z), spatial_shape, batch_size, grid,
                     voxel_num, indice_dict, the benchmark / timer / algo
                     fields SparseTensor.replace copies, replace_feature.
  SubMConv3d         submanifold convolution: outputs at exactly the input
                     sites, out[p] = bias + sum_k W[:, k, :] x[p + k - k//2]
                     over the k^3 offsets (cross-correlation, absent sites
                     are zero). Computed by scattering the features into a
                     dense (B, C, X, Y, Z) grid, F.conv3d with spconv's KRSC
                     weight (out, kx, ky, kz, in) permuted to (out, in, kx,
                     ky, kz), padding k//2 * dilation, and gathering the
                     input sites; that is spconv's definition of SubMConv3d
                     (its tests compare against dense Conv3d this way). The
                     parameters are weight (out, k, k, k, in) and bias
                     (out,), spconv 2.x's state-dict layout (the released
                     TRELLIS checkpoints store [128, 3, 3, 3, 128]).
  SparseConv3d, SparseInverseConv3d, ConvAlgo   strided / inverse convs
                     are not used by the SLat flow model or the Gaussian
                     decoder; constructing one raises.
xformers.ops (trellis/modules/sparse/attention/{full,windowed}_attn.py)
  memory_efficient_attention(q, k, v, attn_bias)   (B, M, H, K) layout,
                     softmax(q k^T / sqrt(K)) v per head, by
                     F.scaled_dot_product_attention (math path); with a
                     BlockDiagonalMask the packed sequence is split into
                     its blocks and each attends only within itself.
  fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen=None)
utils3d.numpy (trellis/representations/gaussian/gaussian_model.py save_ply)
  quaternion_to_matrix, matrix_to_quaternion   copied verbatim from
                     EasternJournalist/utils3d@9a4eb15e (the commit TRELLIS's
                     setup.sh pins), utils3d/numpy/transforms.py.
Gaussian on the CPU (cpu_gaussian(Gaussian))
  Gaussian.setup_functions moves its biases with .cuda() and its default
  device is 'cuda'; cpu_gaussian makes the default device 'cpu' and turns
  .cuda() into a no-op while a Gaussian is built.
"""
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- spconv -------------------------------------------------------------------
class SparseConvTensor:
    def __init__(self, features, indices, spatial_shape, batch_size, grid=None, voxel_num=None, indice_dict=None,
                 benchmark=False, permanent_thrust_allocator=False, enable_timer=False, force_algo=None):
        self._features = features
        self.indices = indices
        self.spatial_shape = [int(s) for s in spatial_shape]
        self.batch_size = int(batch_size)
        self.grid = grid
        self.voxel_num = voxel_num
        self.indice_dict = {} if indice_dict is None else indice_dict
        self.benchmark = benchmark
        self.benchmark_record = {}
        self.thrust_allocator = None
        self._timer = None
        self.force_algo = force_algo
        self.int8_scale = None

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    def replace_feature(self, feature):
        t = SparseConvTensor(feature, self.indices, self.spatial_shape, self.batch_size, self.grid, self.voxel_num,
                             self.indice_dict, self.benchmark)
        t.force_algo = self.force_algo
        return t

    def dense(self, channels_first=True):
        idx = self.indices.long()
        f = self._features
        out = f.new_zeros(self.batch_size, f.shape[1], *self.spatial_shape)
        out[idx[:, 0], :, idx[:, 1], idx[:, 2], idx[:, 3]] = f
        return out if channels_first else out.permute(0, 2, 3, 4, 1).contiguous()


class SubMConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 indice_key=None, algo=None, **kwargs):
        super().__init__()
        k = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
        st = tuple(stride) if isinstance(stride, (list, tuple)) else (stride,) * 3
        assert all(s == 1 for s in st) and groups == 1, "SubMConv3d stand-in: stride 1, groups 1 only"
        self.in_channels, self.out_channels, self.kernel_size = in_channels, out_channels, k
        self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3
        self.weight = nn.Parameter(torch.empty(out_channels, *k, in_channels))  # spconv 2.x KRSC
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        idx = x.indices.long()
        f = x.features
        shape = [int(v) for v in (idx[:, 1:].max(0).values + 1).tolist()]
        shape = [max(a, b) for a, b in zip(shape, x.spatial_shape)]
        dense = f.new_zeros(x.batch_size, f.shape[1], *shape)
        dense[idx[:, 0], :, idx[:, 1], idx[:, 2], idx[:, 3]] = f
        w = self.weight.permute(0, 4, 1, 2, 3)
        pad = tuple(kk // 2 * d for kk, d in zip(self.kernel_size, self.dilation))
        out = F.conv3d(dense, w, None, padding=pad, dilation=self.dilation)
        o = out[idx[:, 0], :, idx[:, 1], idx[:, 2], idx[:, 3]]
        if self.bias is not None:
            o = o + self.bias
        return x.replace_feature(o)


def _unsupported(name):
    def f(*a, **k):
        raise NotImplementedError(f"{name}: not used by the SLat flow model / Gaussian decoder; no CPU stand-in")
    return f


# --- xformers -----------------------------------------------------------------
class BlockDiagonalMask:
    def __init__(self, q_seqlen, kv_seqlen):
        self.q_seqlen, self.kv_seqlen = list(q_seqlen), list(kv_seqlen)

    @classmethod
    def from_seqlens(cls, q_seqlen, kv_seqlen=None):
        return cls(q_seqlen, q_seqlen if kv_seqlen is None else kv_seqlen)


def _attend(q, k, v):
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return o.transpose(1, 2)


def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, scale=None, **kwargs):
    assert p == 0.0 and scale is None, "memory_efficient_attention stand-in: no dropout, default scale"
    if attn_bias is None:
        return _attend(query, key, value)
    assert isinstance(attn_bias, BlockDiagonalMask) and query.shape[0] == 1
    outs, qs, ks = [], 0, 0
    for lq, lk in zip(attn_bias.q_seqlen, attn_bias.kv_seqlen):
        outs.append(_attend(query[:, qs:qs + lq], key[:, ks:ks + lk], value[:, ks:ks + lk]))
        qs, ks = qs + lq, ks + lk
    return torch.cat(outs, dim=1)


# --- utils3d (verbatim, utils3d@9a4eb15e utils3d/numpy/transforms.py) ----------
def quaternion_to_matrix(quaternion: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    assert quaternion.shape[-1] == 4
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True).clip(min=eps)
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    zeros = np.zeros_like(w)
    I = np.eye(3, dtype=quaternion.dtype)
    xyz = quaternion[..., 1:]
    A = xyz[..., :, None] * xyz[..., None, :] - I * (xyz ** 2).sum(axis=-1)[..., None, None]
    B = np.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros
    ], axis=-1).reshape(*quaternion.shape[:-1], 3, 3)
    rot_mat = I + 2 * (A + w[..., None, None] * B)
    return rot_mat


def matrix_to_quaternion(rot_mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = [rot_mat[..., i, j] for i in range(3) for j in range(3)]

    diag = np.diagonal(rot_mat, axis1=-2, axis2=-1)
    M = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=rot_mat.dtype)
    wxyz = 0.5 * np.clip(1 + diag @ M.T, 0.0, None) ** 0.5
    max_idx = np.argmax(wxyz, axis=-1)
    xw = np.sign(m21 - m12)
    yw = np.sign(m02 - m20)
    zw = np.sign(m10 - m01)
    yz = np.sign(m21 + m12)
    xz = np.sign(m02 + m20)
    xy = np.sign(m01 + m10)
    ones = np.ones_like(xw)
    sign = np.where(
        max_idx[..., None] == 0,
        np.stack([ones, xw, yw, zw], axis=-1),
        np.where(
            max_idx[..., None] == 1,
            np.stack([xw, ones, xy, xz], axis=-1),
            np.where(
                max_idx[..., None] == 2,
                np.stack([yw, xy, ones, yz], axis=-1),
                np.stack([zw, xz, yz, ones], axis=-1)
            )
        )
    )
    quat = sign * wxyz
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True).clip(min=eps)
    return quat


def install(spconv=True, xformers=True, utils3d=True):
    """Registers the stand-in modules in sys.modules (only those asked for)."""
    if spconv:
        sp = types.ModuleType("spconv")
        spp = types.ModuleType("spconv.pytorch")
        spp.SparseConvTensor = SparseConvTensor
        spp.SubMConv3d = SubMConv3d
        spp.SparseConv3d = _unsupported("spconv.SparseConv3d")
        spp.SparseInverseConv3d = _unsupported("spconv.SparseInverseConv3d")
        spp.ConvAlgo = types.SimpleNamespace(Native="native", MaskImplicitGemm="implicit_gemm")
        sp.pytorch = spp
        sys.modules["spconv"], sys.modules["spconv.pytorch"] = sp, spp
    if xformers:
        xf = types.ModuleType("xformers")
        xo = types.ModuleType("xformers.ops")
        xo.memory_efficient_attention = memory_efficient_attention
        xo.fmha = types.SimpleNamespace(BlockDiagonalMask=BlockDiagonalMask)
        xf.ops = xo
        sys.modules["xformers"], sys.modules["xformers.ops"] = xf, xo
    if utils3d:
        u = types.ModuleType("utils3d")
        un = types.ModuleType("utils3d.numpy")
        un.quaternion_to_matrix = quaternion_to_matrix
        un.matrix_to_quaternion = matrix_to_quaternion
        u.numpy = un
        sys.modules["utils3d"], sys.modules["utils3d.numpy"] = u, un


def cpu_gaussian(Gaussian):
    """Makes TRELLIS's Gaussian buildable without CUDA (see the module doc)."""
    init = Gaussian.__init__

    def __init__(self, *a, device="cpu", **k):
        cuda = torch.Tensor.cuda
        torch.Tensor.cuda = lambda t, *aa, **kk: t
        try:
            init(self, *a, device=device, **k)
        finally:
            torch.Tensor.cuda = cuda

    Gaussian.__init__ = __init__
