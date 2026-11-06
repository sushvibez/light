# Model verification harness

Validates GoLight's model forward passes against real HuggingFace weights and a
golden reference (the `transformers` library, or a self-contained numpy
re-implementation).

This is how the Llama forward was validated end-to-end: load the **real**
`meta-llama/Llama-3.2-1B-Instruct` safetensors, run our Go forward, and confirm
the next-token prediction matches `transformers` exactly.

## Pieces

| File | What it does | Needs |
|------|--------------|-------|
| `../../cmd/hfvalidate` | Loads a checkpoint with our Go loader + tokenizer and greedily generates | Go |
| `ref_transformers.py` | Golden reference: top-k next token from `transformers` | `torch`, `transformers` |
| `ref_numpy.py` | Golden reference: same forward re-implemented in pure numpy (Llama-family) | `numpy` only |

## Usage

```bash
# point at any locally-cached HF model dir
LLAMA=$(echo ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/*)

# 1) our Go forward (prints the generated tokens)
go run ./cmd/hfvalidate "$LLAMA" "The capital of France is"

# 2) golden reference via transformers (the standard third-party impl)
python examples/verify/ref_transformers.py "$LLAMA" "The capital of France is"

# 3) golden reference via numpy (no torch needed); pass the token ids the Go
#    harness printed, e.g. 128000,791,6864,315,9822,374
python examples/verify/ref_numpy.py "$LLAMA" "" 128000,791,6864,315,9822,374
```

All three should agree on the top token. For "The capital of France is" the
expected next token is `12366` = `" Paris"` (logit ≈ 19.06).

## Bugs this harness caught

Running real weights through the forward (instead of only self-consistency
tests) surfaced two bugs that bf16-vs-f32 self-checks could never catch, because
both sides shared the same defect:

1. **Stale pre-transposed weights.** `nn.Linear` caches a pre-transposed
   `WeightT` for the matmul; loaders assigned `.Weight` directly and never
   refreshed it, so every loaded layer silently computed with its random-init
   weights. Fixed by `Linear.SetWeight`, which keeps both in sync.
2. **Tied-embedding LM head.** `Llama1B/3B` left `TieWeights` unset, so
   `NewLlama` created a **random** `LMHead`; the tied checkpoints have no
   `lm_head.weight`, so it was never overwritten and the final projection used
   random weights (different garbage each run). Fixed by setting `TieWeights` in
   the configs and forcing the tied path in `LoadLlama` when the checkpoint has
   no `lm_head.weight`.
