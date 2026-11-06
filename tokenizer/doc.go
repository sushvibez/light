// Package tokenizer provides pure-Go implementations of the three tokenizer
// families used across Hugging Face models: byte-level BPE (GPT-2, Llama3,
// Mistral), SentencePiece (Llama1/2, Gemma, T5), and WordPiece (BERT).
//
// Tokenizers load from the standard HF tokenizer.json format and expose
// Encode / Decode plus chat-template rendering (Jinja2 subset).
package tokenizer
