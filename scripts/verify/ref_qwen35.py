#!/usr/bin/env python3
"""Qwen3.5 reference logits (transformers, eager) for models/qwen3_5_test.go.

    .venv/bin/python scripts/verify/ref_qwen35.py <repo_or_dir> [bf16|f32] [--tiny-ids | --chat]
    .venv/bin/python scripts/verify/ref_qwen35.py --make-tiny <dir>

--make-tiny writes a seeded tiny-random Qwen3_5ForCausalLM (text only):
two Gated DeltaNet (linear_attention) and two gated full-attention layers,
more value than key heads in the linear layers (exercises the head
repeat), partial RoPE (1/4 of the head dim) and an untied LM head.
--tiny-ids uses the 70-token id sequence the Go test uses (ids[i] =
(37 i + 11) mod 512), so prefill crosses a 64-token chunk boundary.
--chat renders the validation-bar prompt (the same messages as
ref_phi4.py) with the repo's chat template.
Runs on CUDA when available, else on the CPU. Prints the ids, the top-5
last-position logits and the greedy continuation.
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

if sys.argv[1] == "--make-tiny":
    out = sys.argv[2]
    cfg = Qwen3_5TextConfig(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, max_position_embeddings=512, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 0], "mrope_interleaved": True},
    )
    torch.manual_seed(0)
    m = Qwen3_5ForCausalLM(cfg).eval()
    # The init leaves every (1 + w) norm at w = 0 and the gated norm at 1;
    # perturb them (and the decay parameters) so a norm applied with or
    # without the +1, or to the wrong tensor, changes the logits.
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "norm" in name or "A_log" in name or "dt_bias" in name:
                p.add_(0.2 * torch.randn_like(p))
    m.save_pretrained(out, safe_serialization=True)
    print("wrote", out)
    sys.exit(0)

repo = sys.argv[1]
dt = torch.float32 if (len(sys.argv) > 2 and sys.argv[2] == "f32") else torch.bfloat16
dev = "cuda" if torch.cuda.is_available() else "cpu"
chat = "--chat" in sys.argv
tok = None
if "--tiny-ids" in sys.argv:
    ids = [(37 * i + 11) % 512 for i in range(70)]
else:
    tok = AutoTokenizer.from_pretrained(repo)
    if chat:
        msgs = [{"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Explain in one sentence why the sky is blue."}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
    else:
        ids = tok("The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is").input_ids
model = AutoModelForCausalLM.from_pretrained(repo, dtype=dt, attn_implementation="eager").to(dev).eval()
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
