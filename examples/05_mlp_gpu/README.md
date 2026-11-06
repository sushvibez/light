# Example 05 — MLP on GPU

Builds a small 3-layer MLP on CUDA and runs a forward pass:

```
Linear(8→32) → ReLU → Linear(32→32) → SiLU → Linear(32→4) → Softmax
```

First example that exercises `nn.Module`, `nn.Sequential`, `nn.Linear`, plus
activation modules — the foundation of all future real-model code.

## Run

```bash
bash scripts/build_ptx.sh       # once
go run -tags cuda ./examples/05_mlp_gpu
```

## Expected output

```
device=CUDA(0)
net: 5 modules, 6 trainable tensors, 1444 parameters
logits shape=(16, 4)
probs  shape=(16, 4)
probs[sample 0]: [… 4 values summing to 1]
probs[sample 1]: [… 4 values summing to 1]
all 16 rows sum to 1.0 ✓
```
