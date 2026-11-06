# Gated DeltaNet and the Qwen3.5 / 3.6 / 3.8 family

**Date:** 2026-11-06
**Status:** Draft
**Goal:** Run `model_type: qwen3_5` checkpoints (Qwen3.5-0.8B … Qwen3.8-27B) as text-only causal LMs, verified against transformers 5.21, by adding one new op family (the gated delta rule) and a hybrid model that reuses the existing Llama attention pieces.

Roadmap item 11 in `MODELS.md` ("Gated DeltaNet + gated attention + MRoPE"); "Revised next-5" #2.

---

## 1. What the architecture is

Reference: `transformers/models/qwen3_5/modular_qwen3_5.py` (built on `qwen3_next`).

Every decoder layer is pre-norm, `x + mixer(norm(x))` then `x + MLP(norm(x))`, with a dense SwiGLU MLP. `layer_types` picks the mixer: three `linear_attention` layers, then one `full_attention`, repeated (`full_attention_interval: 4`).

| | Qwen3.5-0.8B | Qwen3.8-27B |
|---|---|---|
| layers / hidden / MLP | 24 / 1024 / 3584 | 64 / 5120 / 17408 |
| full attention: Q heads / KV heads / head dim | 8 / 2 / 256 | 24 / 4 / 256 |
| linear attention: K heads / V heads / head dims | 16 / 16 / 128, 128 | 16 / 48 / 128, 128 |
| vocab / tied head | 248 320 / yes | 248 320 / no |
| BF16 size | 1.75 GB | 55.6 GB |

**Norms.** `Qwen3NextRMSNorm` is the Gemma-3 zero-centred RMSNorm, a `(1 + w)` gain. Light already has it, for Gemma.

**Full-attention layer** (`Qwen3NextAttention`), Qwen3 attention plus three differences:
- `q_proj` is twice as wide. Per head, its output splits into `[query | gate]` (view `(…, heads, 2·hd)`, `chunk(2, -1)`). After attention, the output is multiplied by `sigmoid(gate)` before `o_proj`.
- QK-norm over the head dim (zero-centred), as in Qwen3.
- **Partial RoPE**, factor 0.25 (64 of 256 dims), θ = 1e7, via the existing `PartialRotary` path. MRoPE (`mrope_section [11, 11, 10]`, interleaved) collapses to plain RoPE for text, because all three position streams carry the same ids. A text-only test will confirm that against transformers.

**Linear-attention layer** (`Qwen3_5GatedDeltaNet`), new. Per token, with `dk = 128`, `dv = 128`:
1. `in_proj_qkv` produces q, k, v concatenated (`2·Hk·dk + Hv·dv`).
2. A depthwise causal `conv1d`, kernel 4, no bias, then SiLU, over that concatenation.
3. Split the result into q, k, v.
4. `in_proj_z` gives the output gate `z` (`Hv·dv`).
5. `in_proj_b` gives β = `sigmoid(b)` (`Hv`).
6. `in_proj_a` gives the decay `g = −exp(A_log) · softplus(a + dt_bias)` (`Hv`, log space, ≤ 0).
7. When `Hv > Hk` (27B: 48 vs 16), q and k are repeated per group (`repeat_interleave`).
8. q and k are L2-normalised over the head dim, and q is scaled by `dk^-½`.
9. **Gated delta rule**, per head, with the state `S ∈ ℝ^{dk×dv}` starting at 0:
   ```
   S ← exp(g_t) · S
   S ← S + k_t ⊗ β_t (v_t − Sᵀk_t)        (delta-rule write)
   o_t = Sᵀ q_t
   ```
10. `norm(o, z)` = RMSNorm(o) · w · silu(z), the gated RMSNorm (plain weight here, not `1 + w`), then `out_proj`.

The delta rule runs in F32 whatever the activation dtype (`mamba_ssm_dtype: float32`).

**MTP head** (`mtp.*`, 15 tensors). Multi-token prediction, used only for speculative decoding. It's ignored on load. Vision tower (`model.visual.*`): ignored, text only.

**Weights** live under `model.language_model.` (multimodal wrapper, like Qwen3-ASR), e.g. `model.language_model.layers.0.linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}.weight`, `.conv1d.weight (C, 1, 4)`, `.A_log`, `.dt_bias`, `.norm.weight`.

## 2. Light design

### 2.1 Ops (new)

`ops.GatedDeltaRule(q, k, v, g, beta, state) → (out, state')`, with q, k `(B, T, H, dk)`, v `(B, T, H, dv)`, g and β `(B, T, H)`, and state `(B, H, dk, dv)` F32 or nil. Two forms, chosen by T:

- **Recurrent** (T = 1, decode). The loop above, one token. CPU: plain Go. CUDA: one block per (batch, head) holding `S` (128×128 F32 = 64 KB) in registers / shared memory, a few hundred FLOPs per element. This is the decode hot path for 3 of 4 layers.
- **Chunked** (prefill). transformers' `torch_chunk_gated_delta_rule`, chunk 64: per chunk, an intra-chunk lower-triangular system (`solve_triangular` with unit diagonal) plus the recurrent carry between chunks. CPU first.
  - CUDA, first version: the same algorithm composed from existing batched matmul ops, plus one new kernel for the 64×64 unit-lower-triangular solve (forward substitution, one block per (batch, head, chunk)).
  - A fused FLA-style kernel is a later optimisation. Running the recurrent kernel over T is the fallback and the test oracle.

Small helpers, checked to exist or added: depthwise causal `Conv1d` with a carried state (the last `kernel−1` inputs), L2-norm over the last axis, softplus, sigmoid / SiLU gate multiply, `repeat_interleave` on the head axis.

### 2.2 Model

`models.Qwen35` in `models/qwen3_5.go` rather than more flags on `models.Llama`. The layers are heterogeneous, and the cache holds two kinds of state. It reuses:
- the zero-centred RMSNorm (Gemma)
- the partial-RoPE table code (`applyRoPE` / `rotaryDim`, Phi-4-mini)
- `repeatKV`, SDPA / FlashAttention, `nn.Linear` (BF16 + GGUF / quantized weights later)

`Qwen35Cache`:
- **full layers:** the usual growing K/V
- **linear layers:** `conv` `(B, C, kernel−1)` and `S` `(B, Hv, dk, dv)` F32

So decode memory per linear layer is constant: about 3 MB per sequence on the 27B.

Loader `LoadQwen35FromHF(repo, dev, dtype)`: `text_config`, the `model.language_model.` prefix, `layer_types`; it skips `mtp.*` / `model.visual.*`. Route `qwen3_5` / `qwen3_5_text` in `LoadCausalLMFromHF`; add both to the engine catalog.

### 2.3 Verification (bar from MODELS.md: top-5 on a ≥15-token chat prompt)

1. **Op level:** golden tensors from transformers' own `torch_recurrent_gated_delta_rule` / `torch_chunk_gated_delta_rule` on random inputs (several T, including T not a multiple of 64, with and without an initial state), as `.npy` fixtures through `light/npy`. Chunked must equal recurrent within 1e-5 F32.
2. **Tiny-random model:** `scripts/verify/ref_qwen35.py --make-tiny`, the same pattern as `ref_falcon.py` / `ref_mellum.py`. Two of each layer type, Hv > Hk to cover the repeat, partial RoPE, vocab 512, a ~1 MB fixture. Top-5 + greedy + chunked-vs-single-shot on CPU, and a CUDA parity check.
3. **Qwen3.5-0.8B:** F32 and BF16 on CPU here vs transformers (1.75 GB download), then on CUDA (Colab A100 / the 5060 Ti).
4. **Qwen3.5-9B BF16** (18 GB) on the A100: first real-size check.
5. **Qwen3.8-27B:** 55 GB BF16 fits neither GPU, so it needs a weight-compressed path. Candidates:
   - **GPTQ-Int4** (`Qwen/Qwen3.5-27B-GPTQ-Int4`, ~15 GB): roadmap item 17, a GPTQ loader plus an int4 dequant-matmul.
   - **FP8 blockwise** (`Qwen/Qwen3.5-27B-FP8`, ~28 GB): needs `weight_scale_inv` block scales (not yet supported). On the A100 (no FP8 GEMM) it would also need a dequant-to-BF16 path.
   - **GGUF Q4_K_M** through the existing fused `MatMulQ`, if `qwen35` GGUFs exist.

   Decide when 1–4 are done.

## 3. Plan

| Step | Where | Verified by |
|---|---|---|
| 1. `GatedDeltaRule` recurrent + chunked, CPU, plus helpers | here | op fixtures |
| 2. `models.Qwen35` + cache + loader, CPU | here | tiny-random + 0.8B F32/BF16 |
| 3. CUDA: recurrent kernel, triangular-solve kernel, conv state | Colab | op parity + tiny + 0.8B on GPU |
| 4. Engine / catalog / chat template (Jinja, already done) | here | engine test on 0.8B |
| 5. 9B BF16 heavy test | Colab A100 | top-5 vs transformers |
| 6. 27B weight path (pick from §2.3.5) | both | 27B top-5 |

Steps 1–2 need no GPU and can be finished and verified entirely on the Windows machine.

## 4. Risks

- **Chunked-form numerics:** the triangular solve on 64×64 blocks in F32 is well conditioned for unit-diagonal systems with decays ≤ 1. Compare against the recurrent form on long random sequences to be sure.
- **MRoPE collapse to plain RoPE** is assumed for text. If the interleaved layout permutes dims even with equal positions, the tiny-model test will show it immediately.
- **Decode speed:** the recurrent kernel touches all of `S` (dk·dv per head) per token. That's fine for one sequence, but worth a benchmark on the 27B before tuning.
