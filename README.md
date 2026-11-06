# Light

Light is a Go port of [Hugging Face Candle](https://github.com/huggingface/candle):
a tensor library and model runtime for machine learning with CPU and CUDA
backends, a PyTorch-style autograd, loaders for safetensors / GGUF / npz /
PyTorch checkpoints, HF-compatible tokenizers, and ports of the transformer,
vision, diffusion and speech models that Candle ships, each verified against
the reference implementation on real weights.

It also includes an inference engine with an OpenAI-compatible HTTP API
(`cmd/light-serve`) and a desktop app (`studio/`, Wails v3 + React) for
chat, speech-to-text, embeddings and full-duplex speech-to-speech (Moshi).

## What is in the box

| Area | Highlights |
|---|---|
| Tensors | 12 dtypes (F32/F64/F16/BF16/F8, ints, bool) plus GGUF block-quantized formats; views, broadcasting, reductions, indexing, sorting, scans |
| Ops | Every op has a CPU and a CUDA implementation with parity tests; cuBLAS / cuBLASLt (FP8) / cuDNN where they help, fused dequant-matmul kernels for Q4/Q5/Q6/Q8 weights |
| Autograd | Define-by-run graph (`autograd/`), optimizers and training loop examples (`training/`) |
| Models | Llama 3.x, Qwen2/2.5/3 (+MoE), Mistral, Phi-3/3.5/4-mini, Gemma 2/3/4, EmbeddingGemma 2, Falcon, GPT-2, BERT, Whisper, Qwen3-ASR, Mimi, Moshi, SDAR (diffusion LM), Stable Diffusion 1.5, ResNet / MobileNet / ViT / VGG / YOLOv5 — see [`MODELS.md`](MODELS.md) for what each was verified against |
| Formats | safetensors (sharded), GGUF (weights kept quantized, embedded tokenizer), `.npy`/`.npz`, PyTorch `.pth`/`.bin`, Hugging Face Hub downloads with resume |
| Tokenizers | HF `tokenizer.json` (BPE, byte-level, SentencePiece-BPE, Unigram), raw SentencePiece `.model`, GGUF-embedded vocabularies |
| Engine | Catalog of cached checkpoints, chat templates, streaming generation, `/v1/chat/completions`, `/v1/audio/transcriptions`, `/v1/embeddings`, `/v1/audio/conversations` |

Progress and design notes: [`FEATURES.md`](FEATURES.md), [`CHECKPOINTS.md`](CHECKPOINTS.md),
[`docs/PARITY-CANDLE.md`](docs/PARITY-CANDLE.md) (module-by-module parity with Candle),
[`docs/design/`](docs/design/) and [`docs/plans/`](docs/plans/).

## Requirements

- Go 1.26+
- For CUDA: an NVIDIA driver and the CUDA toolkit (cuBLAS, cuDNN) at
  `/usr/local/cuda` or `CUDA_PATH`. Kernels are shipped as PTX
  (`kernels/cuda/ptx/`, compiled for `compute_80`) and JIT-compiled by the
  driver on first use, so `nvcc` is only needed to rebuild them
  (`scripts/build_ptx.sh`). Everything builds without CUDA with `-tags nocuda`.
- Python with `transformers`/`torch` only for regenerating reference numbers
  (`scripts/verify/`), never at runtime.

The reference machine is a single 16 GB RTX 5060 Ti; every test that needs
more than a few GB is gated behind `LIGHT_HEAVY_TESTS=1`.

## Quickstart

```bash
go test ./tensor ./ops ./nn            # CPU + CUDA parity suites

go run ./examples/25_gpt2_pretrained   # real GPT-2 weights, greedy decode
go run ./examples/41_whisper           # speech-to-text on the test clip

# OpenAI-compatible server over everything in your HF cache / GGUF dir
go run ./cmd/light-serve -addr :8129
curl localhost:8129/v1/models

# Desktop app
cd studio && task dev                  # needs wails3 + task
```

Models are pulled from the Hugging Face Hub into the standard
`~/.cache/huggingface/hub` layout (`HF_TOKEN` is honoured for gated repos);
GGUF files are picked up from `~/.cache/light/gguf` or `LIGHT_GGUF_DIR`.

## Verification bar

A model counts as ported when its top-5 next-token logits match the
transformers (or upstream) implementation on a ≥15-token prompt with real
weights, greedy continuations agree, and the CPU and CUDA paths agree with
each other. Codecs and encoders are checked on codes, embeddings and
decoded signals; the per-model numbers and the scripts that produced them
live next to the tests.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). Third-party attributions,
including Candle (Apache-2.0 / MIT), are listed in [`NOTICE`](NOTICE).
Model weights are downloaded from their publishers under their own licenses.
