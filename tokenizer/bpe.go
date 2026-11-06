package tokenizer

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"
)

// BPE is a byte-level Byte-Pair-Encoding tokenizer compatible with the
// Hugging Face / GPT-2 convention.
//
// Encoding flow:
//  1. Special tokens (e.g. `<|im_start|>`) are scanned out as atomic
//     matches, longest-first. Their IDs are emitted directly.
//  2. The remaining text is pre-tokenized with the GPT-2 regex
//     (contractions, words, numbers, punctuation, whitespace).
//  3. Each pre-token is byte-encoded via the GPT-2 byte-to-unicode table.
//  4. Greedy-lowest-rank BPE merges are applied until none remain.
//  5. Each final piece is looked up in the vocabulary.
//
// Decoding reverses steps 4 and 3 and reconstructs the original UTF-8 bytes.
type BPE struct {
	vocab         map[string]int32
	vocabReverse  map[int32]string
	merges        map[mergeKey]int32 // rank; lower = preferred
	byteEncoder   [256]rune
	byteDecoder   map[rune]byte
	preTokenRE    *regexp.Regexp
	preTokenize   func(string) []string // overrides preTokenRE when set (GGUF / llama.cpp regex chains)
	ignoreMerges  bool                  // llama.cpp ignore_merges: a pre-token that is itself a vocab entry skips BPE
	specials      map[string]int32      // surface-form → id, atomic at encode time
	specialsByLen []string              // specials sorted longest first for greedy match
	specialRStrip map[string]bool       // special swallows the whitespace after it
	specialLStrip map[string]bool       // special swallows the whitespace before it

	// SentencePiece mode (Llama-2 / Mistral / Phi-3 `tokenizer.json`: model
	// BPE with byte_fallback and no pre_tokenizer). Each special-free segment
	// is normalized (Prepend "▁", Replace " "→"▁"), BPE runs over runes
	// rather than GPT-2 byte-encoded text, and runes missing from the vocab
	// expand to `<0xNN>` byte pieces.
	spm             bool
	spmStripLeading bool // decoder has a Strip step: drop one leading space on decode
	byteFallback    bool
	normalizer      *textNormalizer
	unkID           int32 // -1 when absent
}

type mergeKey struct {
	A, B string
}

// DefaultPreTokenPattern is an RE2-compatible approximation of GPT-2's
// pre-tokenization regex. GPT-2 upstream uses a lookahead `\s+(?!\S)` which
// Go's regexp engine doesn't support, so trailing-whitespace-vs-inner-
// whitespace tokenize identically here — matters only for exact token-ID
// parity on strings ending in whitespace.
const DefaultPreTokenPattern = `'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+`

// NewBPE constructs a BPE from a vocab and merge list. Merges are given in
// the form "A B" (space-separated); their position in the slice is their
// rank (0 = highest priority).
func NewBPE(vocab map[string]int32, merges []string) (*BPE, error) {
	if len(vocab) == 0 {
		return nil, fmt.Errorf("tokenizer: empty vocab")
	}
	mergeMap := make(map[mergeKey]int32, len(merges))
	for rank, line := range merges {
		parts := strings.SplitN(line, " ", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("tokenizer: malformed merge line %q at rank %d", line, rank)
		}
		mergeMap[mergeKey{parts[0], parts[1]}] = int32(rank)
	}
	rev := make(map[int32]string, len(vocab))
	for tok, id := range vocab {
		rev[id] = tok
	}
	re, err := regexp.Compile(DefaultPreTokenPattern)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: pre-token regex: %w", err)
	}
	b := &BPE{
		vocab:         vocab,
		vocabReverse:  rev,
		merges:        mergeMap,
		preTokenRE:    re,
		byteDecoder:   make(map[rune]byte, 256),
		specials:      map[string]int32{},
		specialRStrip: map[string]bool{},
		specialLStrip: map[string]bool{},
		unkID:         -1,
	}
	buildByteEncoder(&b.byteEncoder, b.byteDecoder)
	return b, nil
}

// SetSentencePiece switches the encoder to SentencePiece-style BPE (see the
// struct comment). norm may be nil; byteFallback enables `<0xNN>` pieces for
// runes outside the vocab, otherwise they map to unk (or are dropped when
// unk is empty).
func (b *BPE) SetSentencePiece(norm *textNormalizer, byteFallback bool, unk string) {
	b.spm = true
	b.byteFallback = byteFallback
	b.normalizer = norm
	if id, ok := b.vocab[unk]; ok && unk != "" {
		b.unkID = id
	}
}

// SetPreTokenizer replaces the GPT-2 regex pre-tokenizer with fn, which
// must split a special-free text segment into the pieces BPE runs over.
func (b *BPE) SetPreTokenizer(fn func(string) []string) { b.preTokenize = fn }

// SetIgnoreMerges mirrors llama.cpp's `ignore_merges` (Llama 3 family): a
// pre-token that is already a vocabulary entry is emitted without merging.
func (b *BPE) SetIgnoreMerges(v bool) { b.ignoreMerges = v }

// IsSentencePiece reports whether the SentencePiece encoding path is active.
func (b *BPE) IsSentencePiece() bool { return b.spm }

// SetSpecialStrip records HF added_tokens `lstrip` / `rstrip`: a special
// with rstrip also consumes the whitespace that follows it (Phi-3's
// `<|end|>\n`), lstrip the whitespace before it.
func (b *BPE) SetSpecialStrip(text string, lstrip, rstrip bool) {
	if lstrip {
		b.specialLStrip[text] = true
	}
	if rstrip {
		b.specialRStrip[text] = true
	}
}

// RegisterSpecial marks `text` as an atomic token: the encoder will emit
// `id` whenever it sees `text` verbatim, without running BPE on the bytes.
// Loaders use this for `added_tokens` in HF `tokenizer.json` files.
func (b *BPE) RegisterSpecial(text string, id int32) {
	if text == "" {
		return
	}
	if _, exists := b.specials[text]; !exists {
		// Insert sorted by length descending so the greedy match is correct.
		b.specialsByLen = append(b.specialsByLen, text)
		// Bubble-sort up — n is tiny (a handful of tokens), so a one-liner
		// is fine and easier to read than sort.Slice.
		for i := len(b.specialsByLen) - 1; i > 0; i-- {
			if len(b.specialsByLen[i]) > len(b.specialsByLen[i-1]) {
				b.specialsByLen[i], b.specialsByLen[i-1] = b.specialsByLen[i-1], b.specialsByLen[i]
			} else {
				break
			}
		}
	}
	b.specials[text] = id
	b.vocab[text] = id
	b.vocabReverse[id] = text
}

// VocabSize returns the number of tokens in the vocabulary.
func (b *BPE) VocabSize() int { return len(b.vocab) }

// Encode tokenizes text into a list of token IDs. Special tokens (registered
// via `added_tokens` in tokenizer.json) are matched atomically before BPE.
// Unknown pieces are silently dropped.
func (b *BPE) Encode(text string) []int32 {
	if text == "" {
		return nil
	}
	var out []int32
	for _, segment := range b.splitOnSpecials(text) {
		if id, ok := segment.specialID(); ok {
			out = append(out, id)
			continue
		}
		out = append(out, b.encodePlain(segment.text)...)
	}
	return out
}

// encodePlain runs the GPT-2 BPE pipeline on a special-token-free segment.
func (b *BPE) encodePlain(text string) []int32 {
	if text == "" {
		return nil
	}
	if b.spm {
		return b.encodeSPM(text)
	}
	var out []int32
	var pieces []string
	if b.preTokenize != nil {
		pieces = b.preTokenize(text)
	} else {
		pieces = b.preTokenRE.FindAllString(text, -1)
	}
	for _, match := range pieces {
		var piece strings.Builder
		for i := 0; i < len(match); i++ {
			piece.WriteRune(b.byteEncoder[match[i]])
		}
		if b.ignoreMerges {
			if id, ok := b.vocab[piece.String()]; ok {
				out = append(out, id)
				continue
			}
		}
		tokens := b.applyBPE(piece.String())
		for _, t := range tokens {
			if id, ok := b.vocab[t]; ok {
				out = append(out, id)
			}
		}
	}
	return out
}

// encodeSPM is the SentencePiece-style path: normalize the segment, seed the
// symbol list with one entry per rune (expanding runes outside the vocab to
// `<0xNN>` byte pieces when byte fallback is on, as HF's BPE model does
// before merging), then merge by rank.
func (b *BPE) encodeSPM(text string) []int32 {
	if b.normalizer != nil {
		text = b.normalizer.apply(text)
	}
	if text == "" {
		return nil
	}
	var syms []string
	for _, r := range text {
		s := string(r)
		if _, ok := b.vocab[s]; ok {
			syms = append(syms, s)
			continue
		}
		if b.byteFallback {
			for _, by := range []byte(s) {
				syms = append(syms, fmt.Sprintf("<0x%02X>", by))
			}
			continue
		}
		syms = append(syms, s) // unknown; resolved to unk below
	}
	var out []int32
	for _, t := range b.mergeSymbols(syms) {
		if id, ok := b.vocab[t]; ok {
			out = append(out, id)
		} else if b.unkID >= 0 {
			out = append(out, b.unkID)
		}
	}
	return out
}

// segment is a contiguous chunk of text emitted by splitOnSpecials. Either
// `id` is set (an exact special-token match) or `text` carries plain text
// to be BPE-encoded.
type segment struct {
	text string
	id   int32
	hit  bool
}

func (s segment) specialID() (int32, bool) { return s.id, s.hit }

// splitOnSpecials walks `text` left-to-right; at each position, the longest
// registered special-token prefix wins. Anything between matches is emitted
// as plain text. With no specials registered this returns one segment for
// the whole input.
func (b *BPE) splitOnSpecials(text string) []segment {
	if len(b.specialsByLen) == 0 {
		return []segment{{text: text}}
	}
	var out []segment
	var buf strings.Builder
	i := 0
	for i < len(text) {
		matched := ""
		var matchedID int32
		for _, s := range b.specialsByLen {
			if strings.HasPrefix(text[i:], s) {
				matched = s
				matchedID = b.specials[s]
				break
			}
		}
		if matched != "" {
			if buf.Len() > 0 {
				txt := buf.String()
				if b.specialLStrip[matched] {
					txt = strings.TrimRightFunc(txt, unicode.IsSpace)
				}
				if txt != "" {
					out = append(out, segment{text: txt})
				}
				buf.Reset()
			}
			out = append(out, segment{id: matchedID, hit: true})
			i += len(matched)
			if b.specialRStrip[matched] {
				for i < len(text) && (text[i] == ' ' || text[i] == '\n' || text[i] == '\t' || text[i] == '\r') {
					i++
				}
			}
			continue
		}
		buf.WriteByte(text[i])
		i++
	}
	if buf.Len() > 0 {
		out = append(out, segment{text: buf.String()})
	}
	return out
}

// applyBPE greedily merges the highest-ranked adjacent pair until none remain.
func (b *BPE) applyBPE(piece string) []string {
	if piece == "" {
		return nil
	}
	// Start with one "word" per rune.
	runes := []rune(piece)
	tokens := make([]string, len(runes))
	for i, r := range runes {
		tokens[i] = string(r)
	}
	return b.mergeSymbols(tokens)
}

// mergeSymbols applies the merge table to an initial symbol list.
func (b *BPE) mergeSymbols(tokens []string) []string {
	if len(tokens) < 2 {
		return tokens
	}
	for {
		bestRank := int32(-1)
		bestIdx := -1
		for i := 0; i < len(tokens)-1; i++ {
			if rank, ok := b.merges[mergeKey{tokens[i], tokens[i+1]}]; ok {
				if bestIdx == -1 || rank < bestRank {
					bestRank = rank
					bestIdx = i
				}
			}
		}
		if bestIdx == -1 {
			break
		}
		merged := tokens[bestIdx] + tokens[bestIdx+1]
		next := make([]string, 0, len(tokens)-1)
		next = append(next, tokens[:bestIdx]...)
		next = append(next, merged)
		next = append(next, tokens[bestIdx+2:]...)
		tokens = next
	}
	return tokens
}

// Decode converts token IDs back to text. Unknown IDs are skipped.
func (b *BPE) Decode(ids []int32) string {
	if len(ids) == 0 {
		return ""
	}
	if b.spm {
		return b.decodeSPM(ids)
	}
	// Concatenate token strings and reverse the byte-to-unicode mapping.
	// Registered specials / added tokens are stored verbatim (GGUF
	// user-defined tokens such as GPT-NeoX's whitespace runs are raw text,
	// not byte-encoded), so they bypass the byte decoder.
	var out []byte
	var pending strings.Builder
	flush := func() {
		for _, r := range pending.String() {
			if by, ok := b.byteDecoder[r]; ok {
				out = append(out, by)
			}
		}
		pending.Reset()
	}
	for _, id := range ids {
		tok, ok := b.vocabReverse[id]
		if !ok {
			continue
		}
		if _, special := b.specials[tok]; special {
			flush()
			out = append(out, tok...)
			continue
		}
		pending.WriteString(tok)
	}
	flush()
	return string(out)
}

// decodeSPM mirrors the HF decoder chain for SentencePiece BPE: `<0xNN>`
// pieces become raw bytes, "▁" becomes a space, specials pass through
// verbatim, and one leading space is stripped (Strip{start:1}).
func (b *BPE) decodeSPM(ids []int32) string {
	var out []byte
	for _, id := range ids {
		tok, ok := b.vocabReverse[id]
		if !ok {
			continue
		}
		// Added tokens go through the same Replace "▁"→" " decoder step as
		// vocab pieces (Gemma registers "▁▁" & co. as added tokens); only
		// the byte-fallback rule is reserved for vocab pieces.
		if _, special := b.specials[tok]; !special && len(tok) == 6 && strings.HasPrefix(tok, "<0x") && tok[5] == '>' {
			var by byte
			if _, err := fmt.Sscanf(tok[3:5], "%02X", &by); err == nil {
				out = append(out, by)
				continue
			}
		}
		out = append(out, strings.ReplaceAll(tok, "▁", " ")...)
	}
	s := string(out)
	if b.spmStripLeading {
		return strings.TrimPrefix(s, " ")
	}
	return s
}

// TokenToID returns the id for a given token (or -1 if not found).
func (b *BPE) TokenToID(tok string) int32 {
	if id, ok := b.vocab[tok]; ok {
		return id
	}
	return -1
}

// IDToToken returns the token string for an id (or "" if not found).
func (b *BPE) IDToToken(id int32) string {
	return b.vocabReverse[id]
}

// buildByteEncoder populates the GPT-2 byte-to-unicode mapping. The 188
// "printable" bytes map to themselves; the remaining 68 map to codepoints
// 256..323 so every byte round-trips through a valid Unicode rune.
func buildByteEncoder(enc *[256]rune, dec map[rune]byte) {
	n := 0
	for b := range 256 {
		inPrintable := (b >= 33 && b <= 126) ||
			(b >= 161 && b <= 172) ||
			(b >= 174 && b <= 255)
		var r rune
		if inPrintable {
			r = rune(b)
		} else {
			r = rune(256 + n)
			n++
		}
		enc[b] = r
		dec[r] = byte(b)
	}
}
