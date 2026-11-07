#!/usr/bin/env python3
"""Convert the tiny Qwen3.5 test checkpoint to GGUF with llama.cpp's own converter.

    python scripts/verify/qwen35_gguf_fixture.py <llama.cpp checkout> f32  models/testdata/qwen35/tiny-llamacpp-f32.gguf
    python scripts/verify/qwen35_gguf_fixture.py <llama.cpp checkout> q8_0 models/testdata/qwen35/tiny-llamacpp-q8_0.gguf

The checkout needs only convert_hf_to_gguf.py, conversion/ and gguf-py/
(`git clone --depth 1 --filter=blob:none --sparse https://github.com/ggml-org/llama.cpp
&& git sparse-checkout set conversion gguf-py`). Everything that touches the
weights is llama.cpp's code (Qwen3_5TextModel: the +1 on norms, −exp(A_log),
the grouped → tiled value-head reorder); the only patch is set_vocab, because
the tiny checkpoint has no tokenizer and LoadQwen35GGUF does not need one.
TestQwen35GGUFTiny (models/qwen3_5_gguf_test.go) loads these files when
present and checks them against the HF load of the same weights, next to
the fixtures it builds itself with a Go reimplementation of the same rules.
"""
import os, sys

llama, outtype, out = sys.argv[1], sys.argv[2], sys.argv[3]
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tiny = os.path.join(root, "models", "testdata", "qwen35", "tiny")
sys.path[:0] = [os.path.join(llama, "gguf-py"), llama]

import convert_hf_to_gguf  # noqa: E402  (registers the model classes)
from conversion import qwen  # noqa: E402

qwen.Qwen3_5TextModel.set_vocab = lambda self: None
sys.argv = ["convert_hf_to_gguf.py", tiny, "--outtype", outtype, "--outfile", out]
convert_hf_to_gguf.main()
