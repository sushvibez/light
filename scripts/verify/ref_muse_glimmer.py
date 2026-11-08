#!/usr/bin/env python3
"""Muse Glimmer reference logits (transformers, eager) for models/muse_glimmer_test.go.

    .venv/bin/python scripts/verify/ref_muse_glimmer.py <repo_or_dir> [bf16|f32] [--ids 1,2,3 | --chat] [--cpu]
    .venv/bin/python scripts/verify/ref_muse_glimmer.py --make-tiny <dir>

--make-tiny writes a seeded tiny-random MuseGlimmerForConditionalGeneration
(text path exercised; a minimal vision tower is saved alongside and ignored)
covering what Muse Glimmer adds to Gemma 3: NoPE global layers
(layer_rope_theta 0), sliding layers with a window shorter than the test
prompt, weightless qk-norm with the query scale factor, the sigmoid output
gate, the normed embedding, centred (1 + w) sandwich norms with a separate
post-norm eps, a plain-weight final norm, the output multiplier and the
softcap, attention width (4 × 24) != hidden size (64). Norm weights are
perturbed away from their init so every convention matters.

--chat renders the validation-bar prompt (system + user, as ref_mellum.py)
with the repo's chat template and decodes 16 greedy tokens. Runs on CUDA
when the weights fit, else on the CPU (--cpu forces it).
"""
import sys, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from transformers.models.muse_glimmer.configuration_muse_glimmer import (
    MuseGlimmerConfig, MuseGlimmerTextConfig)
from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerForConditionalGeneration

if sys.argv[1] == "--make-tiny":
    out = sys.argv[2]
    text = MuseGlimmerTextConfig(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=24, max_position_embeddings=256,
        sliding_window=6, rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        bos_token_id=1, eos_token_id=2, pad_token_id=None)
    vision = dict(hidden_size=32, intermediate_size=64, num_attention_heads=2, num_hidden_layers=1)
    cfg = MuseGlimmerConfig(text_config=text.to_dict(), vision_config=vision,
                            out_hidden_size=64, projector_hidden_size=32,
                            image_token_id=500, video_token_id=501)
    torch.manual_seed(0)
    m = MuseGlimmerForConditionalGeneration(cfg).eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "language_model" in name and "norm" in name:
                p.add_(0.2 * torch.randn_like(p))
    print("layer types", m.config.text_config.layer_types, "rope", m.config.text_config.layer_rope_theta)
    m.save_pretrained(out, safe_serialization=True)
    print("wrote", out)
    sys.exit(0)

repo = sys.argv[1]
dt = torch.float32 if (len(sys.argv) > 2 and sys.argv[2] == "f32") else torch.bfloat16
chat = "--chat" in sys.argv
ids = None
if "--ids" in sys.argv:
    ids = [int(x) for x in sys.argv[sys.argv.index("--ids") + 1].split(",")]
tok = None if ids is not None else AutoTokenizer.from_pretrained(repo)
model = AutoModelForImageTextToText.from_pretrained(repo, dtype=dt, attn_implementation="eager").eval()
dev = "cpu"
if torch.cuda.is_available() and "--cpu" not in sys.argv:
    need = sum(p.numel() * p.element_size() for p in model.parameters())
    free, _ = torch.cuda.mem_get_info()
    if need < free - (2 << 30):
        dev = "cuda"
    else:
        print(f"weights {need / 2**30:.1f} GiB > free GPU {free / 2**30:.1f} GiB - 2: running on the CPU")
model = model.to(dev)
if chat:
    msgs = [{"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Explain in one sentence why the sky is blue."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
elif ids is None:
    ids = tok("The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is").input_ids
print("model class:", type(model).__name__)
print("prompt ids:", ids)
x = torch.tensor([ids], device=dev)
with torch.no_grad():
    logits = model(input_ids=x).logits[0, -1].float()
    topv, topi = logits.topk(5)
    print(f"dtype={dt} device={dev} top5:")
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"  id={i:<7} {(tok.decode([i]) if tok else '')!r:<14} logit={v:.4f}")
    out = model.generate(x, max_new_tokens=16 if chat else 8, do_sample=False)
    if tok:
        print("greedy:", repr(tok.decode(out[0, len(ids):], skip_special_tokens=False)))
    print("greedy ids:", out[0, len(ids):].tolist())
