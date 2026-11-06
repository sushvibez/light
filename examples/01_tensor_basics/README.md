# Example 01 — Tensor basics (CPU)

Creates a few tensors, reshapes and transposes them, adds with broadcasting, and does a small matrix multiplication.

## Run

```bash
go run ./examples/01_tensor_basics
```

## Expected output

```
device=CPU
a shape=(2, 3) data=[0 1 2 3 4 5]
a+b shape=(2, 3) data=[1 2 3 4 5 6]
a @ a^T shape=(2, 2) data=[5 14 14 50]
```
