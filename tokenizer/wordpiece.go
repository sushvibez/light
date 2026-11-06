package tokenizer

import (
	"fmt"
	"strings"
	"unicode"
)

// WordPiece is the BERT-family tokenizer: BertNormalizer (clean / CJK
// padding / accent stripping / lowercasing) → BertPreTokenizer (split on
// whitespace, isolate punctuation) → greedy longest-match-first subword
// segmentation with a continuing-subword prefix (`##`).
//
// Encoding flow:
//  1. Registered special tokens are scanned out verbatim (longest-first).
//  2. Each plain segment is normalized per the tokenizer.json normalizer.
//  3. The segment is split into words; punctuation runes become words.
//  4. Each word is segmented greedily; a word with no valid segmentation,
//     or longer than MaxInputCharsPerWord, becomes one unk token.
//
// Encode never adds [CLS]/[SEP]; EncodeWithSpecial applies the configured
// post-processor so inputs match HF's `tokenizer.encode(text)` exactly.
type WordPiece struct {
	vocab        map[string]int32
	vocabReverse map[int32]string
	cfg          WordPieceConfig
	specials     specialSet
	unkID        int32
	post         *PostProcessor
}

// WordPieceConfig carries the model and normalizer settings from
// tokenizer.json. Zero values pick the BERT defaults.
type WordPieceConfig struct {
	UnkToken                string // default "[UNK]"
	ContinuingSubwordPrefix string // default "##"
	MaxInputCharsPerWord    int    // default 100
	// Normalizer is applied to every non-special segment. nil disables
	// normalization (equivalent to tokenizer.json `"normalizer": null`).
	Normalizer *BertNormalizerConfig
}

// NewWordPiece constructs a WordPiece from a vocab and config. The unk
// token must be present in the vocab.
func NewWordPiece(vocab map[string]int32, cfg WordPieceConfig) (*WordPiece, error) {
	if len(vocab) == 0 {
		return nil, fmt.Errorf("tokenizer: empty vocab")
	}
	if cfg.UnkToken == "" {
		cfg.UnkToken = "[UNK]"
	}
	if cfg.ContinuingSubwordPrefix == "" {
		cfg.ContinuingSubwordPrefix = "##"
	}
	if cfg.MaxInputCharsPerWord <= 0 {
		cfg.MaxInputCharsPerWord = 100
	}
	unkID, ok := vocab[cfg.UnkToken]
	if !ok {
		return nil, fmt.Errorf("tokenizer: unk token %q not in vocab", cfg.UnkToken)
	}
	rev := make(map[int32]string, len(vocab))
	for tok, id := range vocab {
		rev[id] = tok
	}
	return &WordPiece{
		vocab:        vocab,
		vocabReverse: rev,
		cfg:          cfg,
		specials:     newSpecialSet(),
		unkID:        unkID,
	}, nil
}

// RegisterSpecial marks text as an atomic token emitted as id whenever it
// appears verbatim in the input (before normalization, so "[CLS]" survives
// lowercasing).
func (w *WordPiece) RegisterSpecial(text string, id int32) {
	if text == "" {
		return
	}
	w.specials.add(text, id)
	w.vocab[text] = id
	w.vocabReverse[id] = text
}

// SetPostProcessor installs the template used by EncodeWithSpecial and
// EncodePair. nil makes both behave like Encode.
func (w *WordPiece) SetPostProcessor(p *PostProcessor) { w.post = p }

// PostProcessor returns the installed template, or nil.
func (w *WordPiece) PostProcessor() *PostProcessor { return w.post }

// UnkID returns the id of the unk token.
func (w *WordPiece) UnkID() int32 { return w.unkID }

// VocabSize returns the number of tokens in the vocabulary.
func (w *WordPiece) VocabSize() int { return len(w.vocab) }

// TokenToID returns the id for a token, or -1 if absent.
func (w *WordPiece) TokenToID(tok string) int32 {
	if id, ok := w.vocab[tok]; ok {
		return id
	}
	return -1
}

// IDToToken returns the token string for an id, or "" if absent.
func (w *WordPiece) IDToToken(id int32) string { return w.vocabReverse[id] }

// Encode tokenizes text into ids without adding [CLS]/[SEP].
func (w *WordPiece) Encode(text string) []int32 {
	if text == "" {
		return nil
	}
	var out []int32
	for _, seg := range w.specials.split(text) {
		if id, ok := seg.specialID(); ok {
			out = append(out, id)
			continue
		}
		plain := seg.text
		if w.cfg.Normalizer != nil {
			plain = w.cfg.Normalizer.Normalize(plain)
		}
		for _, word := range bertPreTokenize(plain) {
			out = append(out, w.encodeWord(word)...)
		}
	}
	return out
}

// EncodeWithSpecial encodes text and applies the post-processor template
// (for BERT: [CLS] ... [SEP]). Without a post-processor it equals Encode.
func (w *WordPiece) EncodeWithSpecial(text string) []int32 {
	return w.post.Apply(w.Encode(text))
}

// EncodePair encodes a sentence pair with the pair template
// (for BERT: [CLS] a [SEP] b [SEP]).
func (w *WordPiece) EncodePair(a, b string) []int32 {
	return w.post.ApplyPair(w.Encode(a), w.Encode(b))
}

// encodeWord greedily segments one pre-token into subword ids.
func (w *WordPiece) encodeWord(word string) []int32 {
	runes := []rune(word)
	if len(runes) > w.cfg.MaxInputCharsPerWord {
		return []int32{w.unkID}
	}
	var ids []int32
	start := 0
	for start < len(runes) {
		end := len(runes)
		found := false
		var cur int32
		for end > start {
			sub := string(runes[start:end])
			if start > 0 {
				sub = w.cfg.ContinuingSubwordPrefix + sub
			}
			if id, ok := w.vocab[sub]; ok {
				cur = id
				found = true
				break
			}
			end--
		}
		if !found {
			return []int32{w.unkID}
		}
		ids = append(ids, cur)
		start = end
	}
	return ids
}

// Decode converts ids back to text the way HF's WordPiece decoder does:
// pieces are joined with spaces, `##` continuations are glued to the
// previous piece, and the standard punctuation/contraction cleanup is
// applied. Registered special tokens and unknown ids are skipped.
func (w *WordPiece) Decode(ids []int32) string {
	var sb strings.Builder
	n := 0
	for _, id := range ids {
		tok, ok := w.vocabReverse[id]
		if !ok || w.specials.contains(tok) {
			continue
		}
		if n > 0 {
			if strings.HasPrefix(tok, w.cfg.ContinuingSubwordPrefix) {
				tok = tok[len(w.cfg.ContinuingSubwordPrefix):]
			} else {
				tok = " " + tok
			}
		}
		sb.WriteString(wordPieceCleanup(tok))
		n++
	}
	return sb.String()
}

// wordPieceCleaner is HF's per-token `cleanup` replacement set.
var wordPieceCleaner = strings.NewReplacer(
	" .", ".", " ?", "?", " !", "!", " ,", ",",
	" ' ", "'", " n't", "n't", " 'm", "'m",
	" do not", " don't", " 's", "'s", " 've", "'ve", " 're", "'re",
)

func wordPieceCleanup(tok string) string { return wordPieceCleaner.Replace(tok) }

// bertPreTokenize implements BertPreTokenizer: split on whitespace (dropped)
// and isolate every punctuation rune as its own word.
func bertPreTokenize(text string) []string {
	var out []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			out = append(out, cur.String())
			cur.Reset()
		}
	}
	for _, r := range text {
		switch {
		case unicode.IsSpace(r):
			flush()
		case isBertPunct(r):
			flush()
			out = append(out, string(r))
		default:
			cur.WriteRune(r)
		}
	}
	flush()
	return out
}

// isBertPunct matches upstream is_bert_punc: every printable ASCII
// non-alphanumeric character (so also $, ^, backtick and the other symbols
// Unicode does not class as punctuation) plus every Unicode P* rune.
func isBertPunct(r rune) bool {
	if (r >= '!' && r <= '/') || (r >= ':' && r <= '@') ||
		(r >= '[' && r <= '`') || (r >= '{' && r <= '~') {
		return true
	}
	return unicode.IsPunct(r)
}
