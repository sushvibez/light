package main

import (
	"fmt"
	"light/gguf"
	"os"
)

func main() {
	f, _ := os.Open(os.Args[1])
	g, err := gguf.Open(f)
	if err != nil {
		panic(err)
	}
	for _, k := range []string{"general.architecture", "qwen2.block_count", "qwen2.embedding_length", "qwen2.attention.head_count", "qwen2.attention.head_count_kv", "qwen2.feed_forward_length", "qwen2.rope.freq_base", "qwen2.attention.layer_norm_rms_epsilon", "qwen2.context_length", "tokenizer.ggml.model", "tokenizer.ggml.pre"} {
		if s, ok := g.String(k); ok {
			fmt.Println(k, "=", s)
		} else if v, ok := g.Int64(k); ok {
			fmt.Println(k, "=", v)
		} else if v, ok := g.Float64(k); ok {
			fmt.Println(k, "=", v)
		} else {
			fmt.Println(k, "MISSING")
		}
	}
	for _, n := range []string{"token_embd.weight", "blk.0.attn_q.weight", "blk.0.attn_q.bias", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.attn_output.weight", "blk.0.attn_norm.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight", "blk.0.ffn_norm.weight", "output_norm.weight", "output.weight"} {
		if t, ok := g.Tensors[n]; ok {
			fmt.Println(n, t.DType, t.Dims)
		} else {
			fmt.Println(n, "MISSING")
		}
	}
	fmt.Println("ntensors", len(g.Tensors))
}
