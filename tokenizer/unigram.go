package tokenizer

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Unigram is the SentencePiece Unigram tokenizer (T5, ALBERT, XLNet,
// mT5 and most `*.model`-derived tokenizer.json files).
//
// Encoding flow:
//  1. Registered special tokens are scanned out verbatim (longest-first).
//  2. Each plain segment goes through the tokenizer.json normalizer chain.
//  3. Metaspace pre-tokenization: spaces become Replacement (▁), a prefix
//     ▁ is prepended per PrependScheme, and (when Split) the segment is cut
//     before every ▁ so pieces never span word boundaries.
//  4. Each pre-token is segmented by Viterbi search for the maximum total
//     log-probability path through the vocabulary.
//  5. Unknown characters get the unk id (consecutive ones fused when
//     FuseUnk), or `<0xNN>` byte pieces when ByteFallback is on.
type Unigram struct {
	pieces       []string // id → piece for the model vocab (array index = id)
	scores       []float64
	vocab        map[string]int32
	vocabReverse map[int32]string
	maxPieceLen  int // longest piece in runes, bounds the Viterbi inner loop
	minScore     float64
	cfg          UnigramConfig
	specials     specialSet
	normalizer   *textNormalizer
	post         *PostProcessor
}

// UnigramPiece is one vocabulary entry: a piece and its log-probability.
type UnigramPiece struct {
	Piece string
	Score float64
}

// UnigramConfig carries the model and pre-tokenizer settings from
// tokenizer.json. Use DefaultUnigramConfig for the SentencePiece defaults.
type UnigramConfig struct {
	UnkID        int32 // -1 when the model has no unk piece
	ByteFallback bool  // emit <0xNN> pieces for unknown characters
	FuseUnk      bool  // merge runs of unknown characters into one unk id
	// Metaspace pre-tokenizer.
	Replacement   rune   // default '▁'
	PrependScheme string // "always" (default), "first" or "never"
	Split         bool   // cut pre-tokens before every Replacement (default true)
}

// DefaultUnigramConfig returns the SentencePiece conventions: unk id 0,
// fused unks, ▁ replacement, prefix always, split on ▁.
func DefaultUnigramConfig() UnigramConfig {
	return UnigramConfig{UnkID: 0, FuseUnk: true, Replacement: '▁', PrependScheme: "always", Split: true}
}

// unkPenalty mirrors SentencePiece's kUnkPenalty: unknown characters score
// this far below the worst known piece.
const unkPenalty = 10.0

// NewUnigram constructs a Unigram from the ordered vocab (index = id).
func NewUnigram(vocab []UnigramPiece, cfg UnigramConfig) (*Unigram, error) {
	if len(vocab) == 0 {
		return nil, fmt.Errorf("tokenizer: empty vocab")
	}
	if cfg.Replacement == 0 {
		cfg.Replacement = '▁'
	}
	switch cfg.PrependScheme {
	case "always", "first", "never":
	case "":
		cfg.PrependScheme = "always"
	default:
		return nil, fmt.Errorf("tokenizer: unknown prepend_scheme %q", cfg.PrependScheme)
	}
	if cfg.UnkID >= int32(len(vocab)) {
		return nil, fmt.Errorf("tokenizer: unk_id %d out of range for vocab of %d", cfg.UnkID, len(vocab))
	}
	u := &Unigram{
		pieces:       make([]string, len(vocab)),
		scores:       make([]float64, len(vocab)),
		vocab:        make(map[string]int32, len(vocab)),
		vocabReverse: make(map[int32]string, len(vocab)),
		minScore:     math.Inf(1),
		cfg:          cfg,
		specials:     newSpecialSet(),
		normalizer:   &textNormalizer{},
	}
	for i, p := range vocab {
		id := int32(i)
		u.pieces[i] = p.Piece
		u.scores[i] = p.Score
		if _, dup := u.vocab[p.Piece]; !dup {
			u.vocab[p.Piece] = id
		}
		u.vocabReverse[id] = p.Piece
		if n := len([]rune(p.Piece)); n > u.maxPieceLen {
			u.maxPieceLen = n
		}
		if p.Score < u.minScore {
			u.minScore = p.Score
		}
	}
	return u, nil
}

// RegisterSpecial marks text as an atomic token emitted as id whenever it
// appears verbatim in the input (before normalization and Metaspace).
func (u *Unigram) RegisterSpecial(text string, id int32) {
	if text == "" {
		return
	}
	u.specials.add(text, id)
	u.vocab[text] = id
	u.vocabReverse[id] = text
}

// SetNormalizer installs a normalizer chain parsed from tokenizer.json.
func (u *Unigram) setNormalizer(n *textNormalizer) {
	if n != nil {
		u.normalizer = n
	}
}

// SetPostProcessor installs the template used by EncodeWithSpecial and
// EncodePair (e.g. T5's trailing </s>). nil makes both behave like Encode.
func (u *Unigram) SetPostProcessor(p *PostProcessor) { u.post = p }

// PostProcessor returns the installed template, or nil.
func (u *Unigram) PostProcessor() *PostProcessor { return u.post }

// UnkID returns the unk id, or -1 if the model has none.
func (u *Unigram) UnkID() int32 { return u.cfg.UnkID }

// VocabSize returns the number of tokens in the vocabulary.
func (u *Unigram) VocabSize() int { return len(u.vocab) }

// TokenToID returns the id for a piece, or -1 if absent.
func (u *Unigram) TokenToID(tok string) int32 {
	if id, ok := u.vocab[tok]; ok {
		return id
	}
	return -1
}

// IDToToken returns the piece for an id, or "" if absent.
func (u *Unigram) IDToToken(id int32) string { return u.vocabReverse[id] }

// Encode tokenizes text into ids without applying the post-processor.
func (u *Unigram) Encode(text string) []int32 {
	if text == "" {
		return nil
	}
	var out []int32
	for i, seg := range u.specials.split(text) {
		if id, ok := seg.specialID(); ok {
			out = append(out, id)
			continue
		}
		plain := u.normalizer.apply(seg.text)
		for _, word := range u.metaspace(plain, i == 0) {
			out = append(out, u.encodeWord(word)...)
		}
	}
	return out
}

// EncodeWithSpecial encodes text and applies the post-processor template.
func (u *Unigram) EncodeWithSpecial(text string) []int32 {
	return u.post.Apply(u.Encode(text))
}

// EncodePair encodes a sentence pair with the pair template.
func (u *Unigram) EncodePair(a, b string) []int32 {
	return u.post.ApplyPair(u.Encode(a), u.Encode(b))
}

// metaspace applies the Metaspace pre-tokenizer to one segment. first says
// whether this is the first segment of the whole input, which matters for
// PrependScheme "first".
func (u *Unigram) metaspace(text string, first bool) []string {
	rep := string(u.cfg.Replacement)
	text = strings.ReplaceAll(text, " ", rep)
	prepend := u.cfg.PrependScheme == "always" || (u.cfg.PrependScheme == "first" && first)
	if prepend && !strings.HasPrefix(text, rep) {
		text = rep + text
	}
	if text == "" {
		return nil
	}
	if !u.cfg.Split {
		return []string{text}
	}
	// Split before every replacement char ("MergedWithNext"): "▁a▁b" →
	// ["▁a", "▁b"]; a run of replacements yields one-char pre-tokens.
	var words []string
	start := 0
	for i := range text {
		if i > 0 && strings.HasPrefix(text[i:], rep) {
			words = append(words, text[start:i])
			start = i
		}
	}
	words = append(words, text[start:])
	return words
}

// encodeWord Viterbi-segments one pre-token and maps pieces to ids,
// applying byte fallback and unk fusing.
func (u *Unigram) encodeWord(word string) []int32 {
	pieces, known := u.viterbi(word)
	var out []int32
	unkRun := false
	for i, p := range pieces {
		if known[i] {
			out = append(out, u.vocab[p])
			unkRun = false
			continue
		}
		if u.cfg.ByteFallback {
			if ids, ok := u.byteFallbackIDs(p); ok {
				out = append(out, ids...)
				unkRun = false
				continue
			}
		}
		if u.cfg.UnkID < 0 {
			continue
		}
		if u.cfg.FuseUnk && unkRun {
			continue
		}
		out = append(out, u.cfg.UnkID)
		unkRun = true
	}
	return out
}

// byteFallbackIDs returns the <0xNN> ids for every UTF-8 byte of s, or
// false if any byte piece is missing from the vocab.
func (u *Unigram) byteFallbackIDs(s string) ([]int32, bool) {
	ids := make([]int32, 0, len(s))
	for i := 0; i < len(s); i++ {
		id, ok := u.vocab[bytePiece(s[i])]
		if !ok {
			return nil, false
		}
		ids = append(ids, id)
	}
	return ids, true
}

func bytePiece(b byte) string {
	const hex = "0123456789ABCDEF"
	return "<0x" + string(hex[b>>4]) + string(hex[b&0xF]) + ">"
}

// viterbi finds the maximum-log-prob segmentation of s. Positions where no
// vocab piece starts get a single-character unknown node scored
// minScore-unkPenalty, so a path always exists. known[i] reports whether
// pieces[i] is a real vocab piece.
func (u *Unigram) viterbi(s string) (pieces []string, known []bool) {
	runes := []rune(s)
	n := len(runes)
	if n == 0 {
		return nil, nil
	}
	unkScore := u.minScore - unkPenalty
	best := make([]float64, n+1)
	back := make([]int, n+1)
	isUnk := make([]bool, n+1)
	for i := 1; i <= n; i++ {
		best[i] = math.Inf(-1)
	}
	for start := 0; start < n; start++ {
		if math.IsInf(best[start], -1) {
			continue
		}
		hasSingle := false
		limit := n
		if u.maxPieceLen > 0 && start+u.maxPieceLen < n {
			limit = start + u.maxPieceLen
		}
		for end := start + 1; end <= limit; end++ {
			id, ok := u.vocab[string(runes[start:end])]
			if !ok || int(id) >= len(u.scores) {
				continue
			}
			if end == start+1 {
				hasSingle = true
			}
			if sc := best[start] + u.scores[id]; sc > best[end] {
				best[end] = sc
				back[end] = start
				isUnk[end] = false
			}
		}
		if !hasSingle {
			if sc := best[start] + unkScore; sc > best[start+1] {
				best[start+1] = sc
				back[start+1] = start
				isUnk[start+1] = true
			}
		}
	}
	for i := n; i > 0; {
		j := back[i]
		pieces = append(pieces, string(runes[j:i]))
		known = append(known, !isUnk[i])
		i = j
	}
	for l, r := 0, len(pieces)-1; l < r; l, r = l+1, r-1 {
		pieces[l], pieces[r] = pieces[r], pieces[l]
		known[l], known[r] = known[r], known[l]
	}
	return pieces, known
}

// Decode converts ids back to text: pieces are concatenated, <0xNN> byte
// pieces are reassembled into UTF-8 (when ByteFallback is on), ▁ becomes a
// space and the leading prefix space is dropped unless PrependScheme is
// "never". Registered special tokens and unknown ids are skipped.
func (u *Unigram) Decode(ids []int32) string {
	var sb strings.Builder
	var pending []byte
	flush := func() {
		if len(pending) > 0 {
			sb.Write(pending)
			pending = pending[:0]
		}
	}
	for _, id := range ids {
		tok, ok := u.vocabReverse[id]
		if !ok || u.specials.contains(tok) {
			continue
		}
		if u.cfg.ByteFallback {
			if b, ok := parseBytePiece(tok); ok {
				pending = append(pending, b)
				continue
			}
		}
		flush()
		sb.WriteString(tok)
	}
	flush()
	out := strings.ReplaceAll(sb.String(), string(u.cfg.Replacement), " ")
	if u.cfg.PrependScheme != "never" {
		out = strings.TrimPrefix(out, " ")
	}
	return out
}

// parseBytePiece recognises "<0xNN>".
func parseBytePiece(tok string) (byte, bool) {
	if len(tok) != 6 || !strings.HasPrefix(tok, "<0x") || tok[5] != '>' {
		return 0, false
	}
	v, err := strconv.ParseUint(tok[3:5], 16, 8)
	if err != nil {
		return 0, false
	}
	return byte(v), true
}
