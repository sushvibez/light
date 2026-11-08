#!/usr/bin/env python3
"""Nemotron reference logits (transformers, eager) for models/nemotron_test.go.

    .venv/bin/python scripts/verify/ref_nemotron.py <repo_or_dir> [bf16|f32] [--ids 1,2,3 | --chat] [--cpu]
    .venv/bin/python scripts/verify/ref_nemotron.py --make-tiny <dir>

--make-tiny writes a seeded tiny-random NemotronForCausalLM: GQA 4 / 2 heads
of 24, partial rotate-half RoPE over 12 dims, LayerNorm1P norms (weight + 1,
with bias), the non-gated relu² MLP. Norm weights and biases are perturbed so
the +1, the bias and the mean-centring all matter.

--chat renders the validation-bar prompt (system + user) with the repo's
chat template and decodes 16 greedy tokens. Runs on CUDA when the weights
fit, else on the CPU (--cpu forces it).
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, NemotronConfig, NemotronForCausalLM

if sys.argv[1] == "--make-tiny":
    out = sys.argv[2]
    cfg = NemotronConfig(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=24,
                         partial_rotary_factor=0.5, norm_eps=1e-5, max_position_embeddings=256,
                         tie_word_embeddings=False, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    torch.manual_seed(0)
    m = NemotronForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "norm" in name:
                p.add_(0.2 * torch.randn_like(p))
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
