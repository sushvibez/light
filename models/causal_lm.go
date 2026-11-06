package models

import "light/tensor"

// CausalLM is the decoder-only interface the chat engine and examples use,
// implemented by *Llama (Llama / Qwen2 / Mistral / Phi-3, GGUF) and
// *Gemma3.
type CausalLM interface {
	// ForwardWithCache runs tokens (B, T) I64 through the model and returns
	// F32 logits (B, T, V). A nil cache means a stateless forward.
	ForwardWithCache(tokens *tensor.Tensor, cache *KVCache) (*tensor.Tensor, error)
	NumLayers() int
	VocabSize() int
	MaxContext() int
	SlidingWindow() int
	ParamCount() int
}

func (m *Llama) NumLayers() int     { return m.Cfg.NumLayers }
func (m *Llama) VocabSize() int     { return m.Cfg.VocabSize }
func (m *Llama) MaxContext() int    { return m.Cfg.MaxPosition }
func (m *Llama) SlidingWindow() int { return m.Cfg.SlidingWindow }

var _ CausalLM = (*Llama)(nil)
