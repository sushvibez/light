// Package transformers provides reusable transformer components: rotary
// position embeddings (RoPE), scaled dot-product attention (SDPA),
// multi-head attention (MHA), grouped-query attention (GQA), and KV cache.
//
// Built on nn.Module and backed by CUDA fast paths (cuDNN attention where
// available) with pure-Go fallbacks for CPU.
package transformers
