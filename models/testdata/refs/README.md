# Reference files for heavy checkpoints

Each `*.json` here is a transformers reference produced by
`scripts/verify/ref_json.py`, usually on Google Colab
(`scripts/colab/light_reference.ipynb`) for checkpoints that do not fit the
16 GB development card in BF16. `TestGGUFAgainstReferenceJSON`
(`models/heavy_ref_test.go`) compares the quantized GGUF named by the
file's `gguf` glob (looked up in `~/.cache/light/gguf` or `LIGHT_GGUF_DIR`)
against it: pre-tokenized prompt ids, top-5 next-token logits, greedy
continuation. The comparison needs no tokenizer and tolerates quantization
noise by quant level (`tolerance` in the file overrides).

| File | Reference | Verifies |
|---|---|---|
| `llama-3.2-1b-instruct.json` | BF16 on the local card (pipeline check) | `llama-3.2-1b-instruct-q4_k_m.gguf` |

Add a model: run the notebook (or the script locally), drop the JSON here,
put the GGUF in the cache directory, run `LIGHT_HEAVY_TESTS=1 go test
./models -run TestGGUFAgainstReferenceJSON -v`.
