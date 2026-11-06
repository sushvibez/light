# Example 02 — Mini attention scores (CPU)

Computes a one-head attention score matrix from small `Q` and `K` tensors,
scales by `sqrt(d_k)`, applies softmax, and verifies row sums.

This exercises: `MatMul`, `Transpose`, `Div` (broadcast with scalar), `Softmax`,
`Sum`, `Max`.

## Run

```bash
go run ./examples/02_attention_scores
```

## Expected output

```
device=CPU
scores shape=(2, 3) data=[1 1 0 0 1 0]
scaled shape=(2, 3) data=[0.5 0.5 0 0 0.5 0]
attention shape=(2, 3) data=[0.384 0.384 0.233 0.295 0.487 0.217]
row sums (each must ≈ 1): [1 1]
max attention per query: [0.384... 0.487...]
```

(Numbers are approximate; your run should match to ~5 significant digits.)
