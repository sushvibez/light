# CHECKPOINTS (desktop / CUDA session)

Work log of the desktop session (RTX 5060 Ti, branch `desktop/cuda`), kept
apart from `CHECKPOINTS.md` so the two parallel sessions never renumber each
other's rows. Numbering is `D.<n>`, append-only, newest at the bottom; the
laptop merges rows into `CHECKPOINTS.md` / `STATUS.md` and renumbers them
there. Same table format as `CHECKPOINTS.md`.

| Date | # | Checkpoint |
|---|---|---|
| 2026-11-09 | D.1 | **Moshi perf pass, step 0 (desktop, done 2026-11-06, recorded late; 7.32–7.35 built on it)** — one 80 ms Moshi frame from 260 ms to 60 ms on the RTX 5060 Ti: `matmul_q8_0_f32_mv` (M = 1 Q8_0 kernel: 34-byte blocks fetched as aligned 32-bit words + funnel shifts, float4 activations; LM step 135 → 61 ms), concat as one `cudaMemcpy2D` per input (KV append 64 row copies → 2), no per-op `cudaDeviceSynchronize` in any CUDA launcher (single legacy stream orders everything; host reads and frees still sync), strided-op metadata through a pinned-host/device ring with async uploads (`kernels/cuda/meta.go`), caching device allocator with size classes and a 2 GB cap (`storage/cuda.go` `cudaPool`, `DrainCUDAPool` on model load/unload). Profiled with `TestMoshikoProfile` (`LIGHT_MOSHI_FRAMES`): encode 7.9 / LM 28.1 / depformer 18.2 / decode 6.1 ms. |
| 2026-11-09 | D.2 | **Colab references for heavy checkpoints** — `scripts/verify/ref_json.py` (transformers reference → JSON: pre-tokenized chat prompt, top-k logits, greedy; bf16/fp16/f32/8bit/4bit loading; `--gguf` glob + `--tolerance`), `scripts/colab/light_reference.ipynb` (produce references for Llama-3.1-8B, Mistral-7B, Qwen3-8B, Gemma-3-12B on Colab, download; for the heavy tests themselves use `light_heavy_tests.ipynb`), `models/ref_json.go` (`RefJSON`, `LoadRefJSONs`), `models/heavy_ref_test.go` (`TestGGUFAgainstReferenceJSON`: every `testdata/refs/*.json` whose GGUF is in `~/.cache/light/gguf`; top-1 identical, rank-1 logit within the quant-level tolerance (Q8 0.5 / Q5-Q6 1.0 / Q4 2.0 / else 3.0), lower ranks within 2×, greedy agrees on ≥4 tokens; >4 GB files gated). Pipeline checked locally: Llama-3.2-1B-Instruct BF16 reference vs the Q4_K_M GGUF (top-4 ids identical, greedy 7/8). |
