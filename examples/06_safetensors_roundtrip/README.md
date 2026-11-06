# Example 06 — safetensors round-trip onto GPU

Demonstrates the full loader pipeline:

1. Build an MLP on CPU with known weights.
2. Serialize to a safetensors buffer.
3. Load the buffer directly onto the GPU.
4. Swap the loaded tensors into a fresh `nn.Linear` on GPU.
5. Run forward (+ softmax) and verify the numerical result.

This is the smallest realistic sketch of "load a trained model and run
inference on GPU." Real model code in Phase 7 will extend the same pattern
to a full transformer.

## Run

```bash
bash scripts/build_ptx.sh       # once
go run -tags cuda ./examples/06_safetensors_roundtrip
```

## Expected output

```
device=CUDA(0)
wrote safetensors (~300 bytes)
loaded 4 tensors; framework="light"
  l1.weight  device=CUDA(0)  shape=(4, 4)
  l1.bias    device=CUDA(0)  shape=(4)
  l2.weight  device=CUDA(0)  shape=(2, 4)
  l2.bias    device=CUDA(0)  shape=(2)
forward output: [10 -10]
softmax(output): [0.99999997 2.061e-09] (rows sum to 1)
safetensors → GPU → nn.Module forward pass works.
```
