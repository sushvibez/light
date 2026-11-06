package tokenizer

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// hfTokenizerJSON mirrors the subset of HuggingFace's tokenizer.json format
// shared by every model type. Vocab is left raw because BPE/WordPiece store
// a {token: id} map while Unigram stores a [[piece, score], ...] array.
type hfTokenizerJSON struct {
	Model struct {
		Type   string          `json:"type"`
		Vocab  json.RawMessage `json:"vocab"`
		Merges json.RawMessage `json:"merges"` // ["a b", ...] or [["a","b"], ...]
		// WordPiece
		UnkToken                string `json:"unk_token"`
		ContinuingSubwordPrefix string `json:"continuing_subword_prefix"`
		MaxInputCharsPerWord    int    `json:"max_input_chars_per_word"`
		// Unigram
		UnkID        *int32 `json:"unk_id"`
		ByteFallback bool   `json:"byte_fallback"`
		FuseUnk      *bool  `json:"fuse_unk"`
	} `json:"model"`
	Normalizer    json.RawMessage `json:"normalizer"`
	PreTokenizer  json.RawMessage `json:"pre_tokenizer"`
	Decoder       json.RawMessage `json:"decoder"`
	PostProcessor json.RawMessage `json:"post_processor"`
	AddedTokens   []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
		LStrip  bool   `json:"lstrip"`
		RStrip  bool   `json:"rstrip"`
	} `json:"added_tokens"`
}

// isSentencePieceBPE reports whether a BPE tokenizer.json is the
// SentencePiece flavour (Llama-2, Mistral, Phi-3): no pre_tokenizer block,
// so the whole segment is one BPE "word" over runes rather than GPT-2
// byte-level pre-tokens. GPT-2 / Llama-3 / Qwen always carry a ByteLevel
// pre_tokenizer.
func (h *hfTokenizerJSON) isSentencePieceBPE() bool {
	pt := strings.TrimSpace(string(h.PreTokenizer))
	norm := strings.TrimSpace(string(h.Normalizer))
	hasNorm := norm != "" && norm != "null"
	if pt != "" && pt != "null" {
		// Gemma 3: normalizer Replace " "→"▁" followed by a Split on " "
		// (MergedWithPrevious). After the Replace there are no spaces left,
		// so the split is a no-op and the tokenizer is plain SentencePiece.
		var split struct {
			Type    string `json:"type"`
			Pattern struct {
				String string `json:"String"`
			} `json:"pattern"`
		}
		if !hasNorm || json.Unmarshal(h.PreTokenizer, &split) != nil || split.Type != "Split" || split.Pattern.String != " " {
			return false
		}
		return h.Model.ByteFallback
	}
	return h.Model.ByteFallback || hasNorm
}

// decoderStripsLeadingSpace reports whether the decoder chain contains a
// Strip step (Llama-2 / Mistral / Phi-3 remove the space that Prepend "▁"
// added). Gemma's decoder has no Strip, so a leading space survives decode.
func (h *hfTokenizerJSON) decoderStripsLeadingSpace() bool {
	return strings.Contains(string(h.Decoder), `"Strip"`)
}

// mergeLines decodes the merges field, accepting both the legacy "a b"
// string form and the newer ["a","b"] pair form.
func (h *hfTokenizerJSON) mergeLines() ([]string, error) {
	if len(h.Model.Merges) == 0 || string(h.Model.Merges) == "null" {
		return nil, nil
	}
	var lines []string
	if err := json.Unmarshal(h.Model.Merges, &lines); err == nil {
		return lines, nil
	}
	var pairs [][]string
	if err := json.Unmarshal(h.Model.Merges, &pairs); err != nil {
		return nil, fmt.Errorf("merges: %w", err)
	}
	lines = make([]string, len(pairs))
	for i, p := range pairs {
		if len(p) != 2 {
			return nil, fmt.Errorf("merges: entry %d has %d elements, want 2", i, len(p))
		}
		lines[i] = p[0] + " " + p[1]
	}
	return lines, nil
}

// LoadBPE reads an HF-format `tokenizer.json` file and constructs a BPE.
// Only model.type == "BPE" is supported; use Load / Parse to dispatch on
// the model type.
func LoadBPE(r io.Reader) (*BPE, error) {
	var cfg hfTokenizerJSON
	if err := json.NewDecoder(r).Decode(&cfg); err != nil {
		return nil, fmt.Errorf("tokenizer.LoadBPE: json decode: %w", err)
	}
	if cfg.Model.Type != "BPE" && cfg.Model.Type != "" {
		return nil, fmt.Errorf("tokenizer.LoadBPE: unsupported model type %q", cfg.Model.Type)
	}
	bpe, err := buildBPE(&cfg)
	if err != nil {
		return nil, fmt.Errorf("tokenizer.LoadBPE: %w", err)
	}
	return bpe, nil
}

func buildBPE(cfg *hfTokenizerJSON) (*BPE, error) {
	var modelVocab map[string]int32
	if len(cfg.Model.Vocab) > 0 && string(cfg.Model.Vocab) != "null" {
		if err := json.Unmarshal(cfg.Model.Vocab, &modelVocab); err != nil {
			return nil, fmt.Errorf("vocab: %w", err)
		}
	}
	if len(modelVocab) == 0 {
		return nil, fmt.Errorf("empty vocab")
	}
	merges, err := cfg.mergeLines()
	if err != nil {
		return nil, err
	}
	// Fold added_tokens into vocab so the encoder can emit them too.
	vocab := make(map[string]int32, len(modelVocab)+len(cfg.AddedTokens))
	for tok, id := range modelVocab {
		vocab[tok] = id
	}
	for _, at := range cfg.AddedTokens {
		vocab[at.Content] = at.ID
	}
	bpe, err := NewBPE(vocab, merges)
	if err != nil {
		return nil, err
	}
	// Register every added_token as atomic so multi-character control tokens
	// (e.g. <|im_start|>) survive pre-tokenization.
	for _, at := range cfg.AddedTokens {
		bpe.RegisterSpecial(at.Content, at.ID)
		bpe.SetSpecialStrip(at.Content, at.LStrip, at.RStrip)
	}
	if cfg.isSentencePieceBPE() {
		norm, err := compileNormalizer(cfg.Normalizer)
		if err != nil {
			return nil, err
		}
		bpe.SetSentencePiece(norm, cfg.Model.ByteFallback, cfg.Model.UnkToken)
		bpe.spmStripLeading = cfg.decoderStripsLeadingSpace()
		return bpe, nil
	}
	// Byte-level BPE: honour the model's own Split regexes (Qwen2 splits
	// digits singly and glues a punctuation mark to a following word, Llama 3
	// groups digits in threes; the GPT-2 pattern does neither). A bare
	// ByteLevel pre-tokenizer keeps the GPT-2 regex.
	if pats := cfg.splitRegexes(); len(pats) > 0 {
		split, err := newRegexChain(pats)
		if err != nil {
			return nil, err
		}
		bpe.SetPreTokenizer(split)
	}
	return bpe, nil
}

// splitRegexes collects the `Split` regex patterns (behavior Isolated) of
// the pre_tokenizer block, in order, as a llama.cpp-style chain.
func (h *hfTokenizerJSON) splitRegexes() []string {
	type preTok struct {
		Type     string `json:"type"`
		Behavior string `json:"behavior"`
		Invert   bool   `json:"invert"`
		Pattern  struct {
			Regex  string `json:"Regex"`
			String string `json:"String"`
		} `json:"pattern"`
		PreTokenizers []json.RawMessage `json:"pretokenizers"`
	}
	var walk func(raw json.RawMessage, out []string) []string
	walk = func(raw json.RawMessage, out []string) []string {
		if len(raw) == 0 || string(raw) == "null" {
			return out
		}
		var pt preTok
		if json.Unmarshal(raw, &pt) != nil {
			return out
		}
		switch pt.Type {
		case "Sequence":
			for _, sub := range pt.PreTokenizers {
				out = walk(sub, out)
			}
		case "Split":
			// Isolated keeps the matches as tokens; Removed+invert (GPT-4o) is
			// the same thing written the other way round.
			if pt.Pattern.Regex != "" && (pt.Behavior == "" || pt.Behavior == "Isolated" || (pt.Behavior == "Removed" && pt.Invert)) {
				out = append(out, pt.Pattern.Regex)
			}
		}
		return out
	}
	return walk(h.PreTokenizer, nil)
}
