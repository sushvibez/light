#!/usr/bin/env python3
"""Golden fixtures for ops.GatedDeltaRule and ops.CausalConv1dDepthwise.

    .venv/bin/python scripts/verify/ref_gated_delta.py ops/testdata/gated_delta

Calls transformers' own reference implementations (the Qwen3-Next /
Qwen3.5 linear-attention path, transformers 5.21) on seeded random inputs
and writes one .npz per case with the inputs and expected outputs, all F32:

  delta_*.npz  q, k (B,T,H,dk), v (B,T,H,dv), g, beta (B,T,H),
               [state (B,H,dk,dv)], out (B,T,H,dv), final (B,H,dk,dv),
               meta = [chunked(0/1), chunk_size, l2norm(0/1)]
  conv_*.npz   x (B,C,T), w (C,K), out (B,C,T) = silu(causal depthwise conv)
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.qwen3_next import modeling_qwen3_next as qn

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)


def inputs(seed, B, T, H, dk, dv, with_state):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    q, k, v = r(B, T, H, dk), r(B, T, H, dk), r(B, T, H, dv)
    # Decays as the model makes them: g = -exp(A_log) * softplus(a + dt_bias) <= 0.
    decay = -torch.exp(0.3 * r(H)) * F.softplus(r(B, T, H))
    beta = torch.sigmoid(r(B, T, H))
    state = 0.5 * r(B, H, dk, dv) if with_state else None
    return q, k, v, decay, beta, state


def delta_case(name, seed, B, T, H, dk, dv, with_state, chunked, chunk_size=64):
    q, k, v, decay, beta, state = inputs(seed, B, T, H, dk, dv, with_state)
    if chunked:
        out, final = qn.torch_chunk_gated_delta_rule(
            q, k, v, g=decay, beta=beta, chunk_size=chunk_size, initial_state=state,
            output_final_state=True, use_qk_l2norm_in_kernel=True)
    else:
        out, final = qn.torch_recurrent_gated_delta_rule(
            q, k, v, g=decay, beta=beta, initial_state=state,
            output_final_state=True, use_qk_l2norm_in_kernel=True)
    arrs = dict(q=q, k=k, v=v, g=decay, beta=beta, out=out, final=final)
    if state is not None:
        arrs["state"] = state
    arrs = {n: t.float().numpy() for n, t in arrs.items()}
    arrs["meta"] = np.array([int(chunked), chunk_size, 1], dtype=np.int64)
    np.savez(os.path.join(out_dir, f"delta_{name}.npz"), **arrs)
    print(f"delta_{name}: out {tuple(out.shape)} |out|max {out.abs().max():.3f}")


def conv_case(name, seed, B, C, T, K):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, C, T, generator=g)
    w = torch.randn(C, K, generator=g) * 0.5
    out = qn.causal_conv1d_fn(x, w, None, activation="silu")
    np.savez(os.path.join(out_dir, f"conv_{name}.npz"), x=x.numpy(), w=w.numpy(), out=out.float().numpy())
    print(f"conv_{name}: out {tuple(out.shape)}")


# Decode step: one token onto an existing state (the T=1 path the model uses).
delta_case("recurrent_t1_state", 1, B=2, T=1, H=3, dk=8, dv=6, with_state=True, chunked=False)
# Recurrent over a sequence, and the chunked form of the same inputs.
delta_case("recurrent_t37", 2, B=1, T=37, H=2, dk=8, dv=4, with_state=False, chunked=False)
delta_case("chunked_t37_c16", 2, B=1, T=37, H=2, dk=8, dv=4, with_state=False, chunked=True, chunk_size=16)
# Several 64-token chunks plus a partial one, starting from a state.
delta_case("chunked_t150_state", 3, B=2, T=150, H=2, dk=16, dv=8, with_state=True, chunked=True)
# Real head dims (128 x 128), one chunk.
delta_case("chunked_t20_d128", 4, B=1, T=20, H=1, dk=128, dv=128, with_state=False, chunked=True)
# Real head dims over many chunks from a state: what the CUDA chunked kernel
# (16-token blocks, 128 x 128 heads only) runs on.
delta_case("chunked_t150_d128_state", 7, B=2, T=150, H=2, dk=128, dv=128, with_state=True, chunked=True)

conv_case("k4", 5, B=2, C=6, T=11, K=4)
conv_case("k4_t2", 6, B=1, C=3, T=2, K=4)
