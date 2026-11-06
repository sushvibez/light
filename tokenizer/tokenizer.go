package tokenizer

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
)

// Tokenizer is the interface shared by every model type in this package:
// BPE (GPT-2 / Llama 3 / Qwen), WordPiece (BERT family) and Unigram
// (SentencePiece / T5). Ids are int32 to match the model embedding APIs.
type Tokenizer interface {
	// Encode converts text to ids without adding template specials.
	Encode(text string) []int32
	// Decode converts ids back to text.
	Decode(ids []int32) string
	// VocabSize is the number of distinct tokens, including added tokens.
	VocabSize() int
	// TokenToID returns -1 when the token is unknown.
	TokenToID(tok string) int32
	// IDToToken returns "" when the id is unknown.
	IDToToken(id int32) string
}

// SpecialRegistrar is implemented by every tokenizer here; loaders use it
// to register tokenizer.json `added_tokens` as atomic matches.
type SpecialRegistrar interface {
	RegisterSpecial(text string, id int32)
}

// TemplateEncoder is implemented by tokenizers that carry a post-processor
// template (WordPiece and Unigram): EncodeWithSpecial reproduces HF's
// `encode(text)` with add_special_tokens=True.
type TemplateEncoder interface {
	EncodeWithSpecial(text string) []int32
	EncodePair(a, b string) []int32
}

var (
	_ Tokenizer = (*BPE)(nil)
	_ Tokenizer = (*WordPiece)(nil)
	_ Tokenizer = (*Unigram)(nil)

	_ SpecialRegistrar = (*BPE)(nil)
	_ SpecialRegistrar = (*WordPiece)(nil)
	_ SpecialRegistrar = (*Unigram)(nil)

	_ TemplateEncoder = (*WordPiece)(nil)
	_ TemplateEncoder = (*Unigram)(nil)
)

// Load reads an HF tokenizer.json from disk and returns the tokenizer
// matching its model.type.
func Load(path string) (Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: read %q: %w", path, err)
	}
	return Parse(data)
}

// Parse builds a tokenizer from tokenizer.json bytes, dispatching on
// model.type ("BPE", "WordPiece" or "Unigram"). Every entry in
// `added_tokens` is registered as a special so it is matched atomically.
func Parse(data []byte) (Tokenizer, error) {
	var cfg hfTokenizerJSON
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("tokenizer: parse: %w", err)
	}
	switch cfg.Model.Type {
	case "BPE", "":
		return LoadBPE(bytes.NewReader(data))
	case "WordPiece":
		return buildWordPiece(&cfg)
	case "Unigram":
		return buildUnigram(&cfg)
	default:
		return nil, fmt.Errorf("tokenizer: unsupported model type %q (BPE, WordPiece, Unigram supported)", cfg.Model.Type)
	}
}

// buildWordPiece assembles a WordPiece from the parsed JSON, honoring the
// BertNormalizer block and the TemplateProcessing/BertProcessing
// post-processor.
func buildWordPiece(cfg *hfTokenizerJSON) (*WordPiece, error) {
	var vocab map[string]int32
	if err := json.Unmarshal(cfg.Model.Vocab, &vocab); err != nil {
		return nil, fmt.Errorf("tokenizer: WordPiece vocab: %w", err)
	}
	wcfg := WordPieceConfig{
		UnkToken:                cfg.Model.UnkToken,
		ContinuingSubwordPrefix: cfg.Model.ContinuingSubwordPrefix,
		MaxInputCharsPerWord:    cfg.Model.MaxInputCharsPerWord,
	}
	norm, err := parseBertNormalizer(cfg.Normalizer)
	if err != nil {
		return nil, err
	}
	wcfg.Normalizer = norm
	// added_tokens may carry the unk token with the same id; fold them in
	// first so NewWordPiece can find it even when the model vocab omits it.
	for _, at := range cfg.AddedTokens {
		if _, ok := vocab[at.Content]; !ok {
			vocab[at.Content] = at.ID
		}
	}
	wp, err := NewWordPiece(vocab, wcfg)
	if err != nil {
		return nil, err
	}
	for _, at := range cfg.AddedTokens {
		wp.RegisterSpecial(at.Content, at.ID)
	}
	post, err := compilePostProcessor(cfg.PostProcessor)
	if err != nil {
		return nil, err
	}
	wp.SetPostProcessor(post)
	return wp, nil
}

// parseBertNormalizer interprets the `normalizer` block for WordPiece.
// A BertNormalizer block is decoded field-for-field; a bare Lowercase
// normalizer maps to lowercase-only; null disables normalization.
func parseBertNormalizer(raw json.RawMessage) (*BertNormalizerConfig, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var probe struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		return nil, fmt.Errorf("tokenizer: normalizer: %w", err)
	}
	switch probe.Type {
	case "BertNormalizer":
		var n BertNormalizerConfig
		if err := json.Unmarshal(raw, &n); err != nil {
			return nil, fmt.Errorf("tokenizer: BertNormalizer: %w", err)
		}
		return &n, nil
	case "Lowercase":
		f := false
		return &BertNormalizerConfig{Lowercase: true, StripAccents: &f}, nil
	default:
		// Other normalizer chains (Sequence, Precompiled, …) are not modelled
		// for WordPiece; fall back to no normalization.
		return nil, nil
	}
}

// preTokenizerJSON is the subset of pre_tokenizer configs Unigram reads.
type preTokenizerJSON struct {
	Type           string             `json:"type"`
	Replacement    string             `json:"replacement"`
	AddPrefixSpace *bool              `json:"add_prefix_space"`
	PrependScheme  string             `json:"prepend_scheme"`
	Split          *bool              `json:"split"`
	Pretokenizers  []preTokenizerJSON `json:"pretokenizers"`
}

// findMetaspace returns the first Metaspace block in a (possibly nested)
// pre_tokenizer config, or nil.
func findMetaspace(pt *preTokenizerJSON) *preTokenizerJSON {
	if pt.Type == "Metaspace" {
		return pt
	}
	for i := range pt.Pretokenizers {
		if m := findMetaspace(&pt.Pretokenizers[i]); m != nil {
			return m
		}
	}
	return nil
}

// buildUnigram assembles a Unigram from the parsed JSON, honoring unk_id,
// byte_fallback, fuse_unk, the normalizer chain, the Metaspace
// pre-tokenizer and the post-processor template.
func buildUnigram(cfg *hfTokenizerJSON) (*Unigram, error) {
	var entries [][]json.RawMessage
	if err := json.Unmarshal(cfg.Model.Vocab, &entries); err != nil {
		return nil, fmt.Errorf("tokenizer: Unigram vocab: %w", err)
	}
	vocab := make([]UnigramPiece, len(entries))
	for i, e := range entries {
		if len(e) < 1 {
			return nil, fmt.Errorf("tokenizer: Unigram vocab entry %d is empty", i)
		}
		if err := json.Unmarshal(e[0], &vocab[i].Piece); err != nil {
			return nil, fmt.Errorf("tokenizer: Unigram piece %d: %w", i, err)
		}
		if len(e) > 1 {
			if err := json.Unmarshal(e[1], &vocab[i].Score); err != nil {
				return nil, fmt.Errorf("tokenizer: Unigram score %d: %w", i, err)
			}
		}
	}
	ucfg := DefaultUnigramConfig()
	ucfg.UnkID = -1
	if cfg.Model.UnkID != nil {
		ucfg.UnkID = *cfg.Model.UnkID
	}
	ucfg.ByteFallback = cfg.Model.ByteFallback
	if cfg.Model.FuseUnk != nil {
		ucfg.FuseUnk = *cfg.Model.FuseUnk
	}
	if len(cfg.PreTokenizer) > 0 && string(cfg.PreTokenizer) != "null" {
		var pt preTokenizerJSON
		if err := json.Unmarshal(cfg.PreTokenizer, &pt); err != nil {
			return nil, fmt.Errorf("tokenizer: pre_tokenizer: %w", err)
		}
		if m := findMetaspace(&pt); m != nil {
			if r := []rune(m.Replacement); len(r) > 0 {
				ucfg.Replacement = r[0]
			}
			switch {
			case m.PrependScheme != "":
				ucfg.PrependScheme = m.PrependScheme
			case m.AddPrefixSpace != nil && !*m.AddPrefixSpace:
				ucfg.PrependScheme = "never"
			}
			if m.Split != nil {
				ucfg.Split = *m.Split
			}
		}
	}
	u, err := NewUnigram(vocab, ucfg)
	if err != nil {
		return nil, err
	}
	norm, err := compileNormalizer(cfg.Normalizer)
	if err != nil {
		return nil, err
	}
	u.setNormalizer(norm)
	for _, at := range cfg.AddedTokens {
		u.RegisterSpecial(at.Content, at.ID)
	}
	post, err := compilePostProcessor(cfg.PostProcessor)
	if err != nil {
		return nil, err
	}
	u.SetPostProcessor(post)
	return u, nil
}
