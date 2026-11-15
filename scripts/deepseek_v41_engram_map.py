#!/usr/bin/env python3
"""Write engram_token_map.json for DeepSeek-V4.1 (models.LoadDeepseekV41).

    python -I scripts/deepseek_v41_engram_map.py <model_dir>

The engram layers hash n-grams over a compressed token id space in which
tokens that normalise alike (NFKC, NFD, accents stripped, lowercased,
whitespace collapsed) share one id. Light has no Unicode normaliser, so the
map is built here, as DeepSeek's engram.py build_compressed_token_map does,
from the checkpoint's tokenizer, and saved next to config.json. Needs
transformers and tokenizers.
"""
import json, os, sys

from tokenizers import Regex, normalizers
from transformers import AutoTokenizer

model_dir = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(model_dir)
sentinel = ""  # keeps a lone-space token from stripping to ""
normalizer = normalizers.Sequence([
    normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
    normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
    normalizers.Strip(), normalizers.Replace(sentinel, " "),
])
backend = tokenizer.backend_tokenizer
key_to_new, lookup = {}, [0] * len(tokenizer)
for token_id in range(len(tokenizer)):
    text = backend.decode([token_id], skip_special_tokens=False)
    if "�" in text:  # a partial UTF-8 byte token: keyed by its raw form
        key = backend.id_to_token(token_id)
    else:
        normalized = normalizer.normalize_str(text)
        key = normalized if normalized else text
    lookup[token_id] = key_to_new.setdefault(key, len(key_to_new))

cfg = json.load(open(os.path.join(model_dir, "config.json")))
want = cfg.get("text_config", cfg).get("engram_compressed_vocab_size")
if want is not None and want != len(key_to_new):
    sys.exit(f"compressed vocab {len(key_to_new)} != engram_compressed_vocab_size {want}")
json.dump({"token_map": lookup, "compressed_vocab_size": len(key_to_new)},
          open(os.path.join(model_dir, "engram_token_map.json"), "w"))
print(f"{len(lookup)} tokens -> {len(key_to_new)} compressed ids")
