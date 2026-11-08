#!/usr/bin/env python3
"""GLM-4 reference logits (transformers, eager) for models/glm_test.go.

    .venv/bin/python scripts/verify/ref_glm.py <repo_or_dir> [bf16|f32] [--ids 1,2,3 | --chat] [--cpu]
    .venv/bin/python scripts/verify/ref_glm.py --make-tiny {glm|glm4} <dir>

--make-tiny writes a seeded tiny-random GlmForCausalLM ("glm", GLM-4-9B-chat
layout) or Glm4ForCausalLM ("glm4", GLM-4-0414: post-attention and post-MLP
norms): GQA 4 / 2 heads of 24 with partial interleaved RoPE over 12 dims,
Q/K/V biases, fused gate_up_proj. Norm weights and the (zero-initialised)
Q/K/V biases are perturbed, so the bias reordering and every norm matter.

--chat renders the validation-bar prompt (system + user) with the repo's
chat template and decodes 16 greedy tokens. Runs on CUDA when the weights
fit, else on the CPU (--cpu forces it).
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if sys.argv[1] == "--make-tiny":
    variant, out = sys.argv[2], sys.argv[3]
    if variant == "glm":
        from transformers import GlmConfig as Cfg, GlmForCausalLM as Model
    else:
        from transformers import Glm4Config as Cfg, Glm4ForCausalLM as Model
    cfg = Cfg(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
              num_attention_heads=4, num_key_value_heads=2, head_dim=24, partial_rotary_factor=0.5,
              attention_bias=True, max_position_embeddings=256, rms_norm_eps=1e-5,
              tie_word_embeddings=False, pad_token_id=0, eos_token_id=2, bos_token_id=1)
    torch.manual_seed(0)
    m = Model(cfg).eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "norm" in name:
                p.add_(0.2 * torch.randn_like(p))
            elif name.endswith(".bias"):
                p.add_(0.5 * torch.randn_like(p))
    m.save_pretrained(out, safe_serialization=True)
    print("wrote", out, type(m).__name__)
    sys.exit(0)

repo = sys.argv[1]
dt = torch.float32 if (len(sys.argv) > 2 and sys.argv[2] == "f32") else torch.bfloat16
chat = "--chat" in sys.argv
ids = None
if "--ids" in sys.argv:
    ids = [int(x) for x in sys.argv[sys.argv.index("--ids") + 1].split(",")]
tok = None if ids is not None else AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=dt, attn_implementation="eager").eval()
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
