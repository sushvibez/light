#!/usr/bin/env python3
"""Ministral 3 reference logits (transformers, eager, CPU) for models/ministral_test.go.

    .venv/bin/python scripts/verify/ref_ministral.py --make-tiny <dir>
    .venv/bin/python scripts/verify/ref_ministral.py <repo_or_dir> [bf16|f32] [--ids 1,2,3 | --chat]

--make-tiny writes a seeded tiny-random checkpoint laid out like
mistralai/Ministral-3-3B-Instruct-2512: a `mistral3` wrapper (tiny Pixtral
vision tower + projector) around a `ministral3` text model with YaRN RoPE
and llama_4_scaling_beta, whose original_max_position_embeddings is 8 so
the Llama 4 query scaling is active inside a short prompt. Language-model
linears are stored as per-tensor FP8 (F8E4M3 weight, BF16 scalar
weight_scale_inv and activation_scale) under the checkpoint's
`language_model.model.` / `vision_tower.` names, with an fp8
quantization_config. Without a GPU transformers dequantizes the FP8
weights at load (into the model dtype). Prints the ids, the top-5
last-position logits and the greedy continuation.
"""
import json, os, sys
import torch
from safetensors.torch import save_file
# The model class is reached through the Auto API: its 32-character name
# trips GitHub's secret scanning (it looks like a Mistral API key).
from transformers import AutoModelForImageTextToText, AutoTokenizer, Ministral3Config, Mistral3Config, PixtralVisionConfig

if sys.argv[1] == "--make-tiny":
    out = sys.argv[2]
    text = Ministral3Config(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
        rms_norm_eps=1e-5, tie_word_embeddings=True, sliding_window=None,
        rope_parameters={"rope_type": "yarn", "type": "yarn", "rope_theta": 1000000.0, "factor": 16.0,
                         "original_max_position_embeddings": 8, "beta_fast": 32.0, "beta_slow": 1.0,
                         "mscale": 1.0, "mscale_all_dim": 1.0, "llama_4_scaling_beta": 0.1},
    )
    vision = PixtralVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                                 head_dim=16, image_size=56, patch_size=14)
    cfg = Mistral3Config(text_config=text, vision_config=vision, image_token_index=10, spatial_merge_size=2,
                         projector_hidden_act="gelu", multimodal_projector_bias=False, vision_feature_layer=-1)
    torch.manual_seed(0)
    m = AutoModelForImageTextToText.from_config(cfg).eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "norm" in name:
                p.add_(0.2 * torch.randn_like(p))
    sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    outsd = {}
    lin = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    for k, v in sd.items():
        # Checkpoint names (as the real repo stores them).
        name = k
        if k.startswith("model.language_model."):
            name = "language_model.model." + k[len("model.language_model."):]
        elif k.startswith("lm_head."):
            name = "language_model." + k
        elif k.startswith("model.vision_tower."):
            name = k[len("model."):]
        elif k.startswith("model.multi_modal_projector."):
            name = k[len("model."):]
        if name.startswith("language_model.model.layers.") and name.endswith(".weight") and name.split(".")[-2] in lin:
            w = v.float()
            scale = (w.abs().max() / 448.0).to(torch.bfloat16).float()
            q = (w / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            base = name[: -len(".weight")]
            outsd[name] = q
            outsd[base + ".weight_scale_inv"] = scale.to(torch.bfloat16).reshape(())
            outsd[base + ".activation_scale"] = torch.tensor(1.0, dtype=torch.bfloat16)
        else:
            outsd[name] = v.to(torch.bfloat16).contiguous()
    if "language_model.lm_head.weight" in outsd and text.tie_word_embeddings:
        del outsd["language_model.lm_head.weight"]
    os.makedirs(out, exist_ok=True)
    save_file(outsd, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    c = cfg.to_dict()
    c["architectures"] = [type(m).__name__]
    c["quantization_config"] = {"quant_method": "fp8", "activation_scheme": "static", "weight_block_size": None,
                                "dequantize": False,
                                "modules_to_not_convert": ["model.vision_tower", "model.multi_modal_projector", "lm_head"]}
    json.dump(c, open(os.path.join(out, "config.json"), "w"), indent=2)
    print("wrote", out, sum(1 for k in outsd if k.endswith("weight_scale_inv")), "FP8 linears")
    sys.exit(0)

repo = sys.argv[1]
dt = torch.float32 if (len(sys.argv) > 2 and sys.argv[2] == "f32") else torch.bfloat16
ids = None
chat = "--chat" in sys.argv
if "--ids" in sys.argv:
    ids = [int(x) for x in sys.argv[sys.argv.index("--ids") + 1].split(",")]
tok = None if ids is not None else AutoTokenizer.from_pretrained(repo)
# Load in BF16 (transformers dequantizes FP8 weights to the scale dtype,
# BF16, whatever the model dtype), then widen for an F32 reference: exact F32
# arithmetic over the BF16-rounded dequantized weights.
model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16, attn_implementation="eager").eval()
if dt == torch.float32:
    model = model.float()
if chat:
    msgs = [{"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Explain in one sentence why the sky is blue."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
print("prompt ids:", ids)
x = torch.tensor([ids])
with torch.no_grad():
    logits = model(input_ids=x).logits[0, -1].float()
    topv, topi = logits.topk(5)
    print(f"dtype={dt} top5:")
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"  id={i:<7} {(tok.decode([i]) if tok else '')!r:<14} logit={v:.4f}")
    out = model.generate(x, max_new_tokens=16 if chat else 8, do_sample=False)
    if tok:
        print("greedy:", repr(tok.decode(out[0, len(ids):], skip_special_tokens=False)))
    print("greedy ids:", out[0, len(ids):].tolist())
