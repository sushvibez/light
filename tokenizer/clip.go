package tokenizer

import (
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"strings"
	"unicode"
)

// CLIPTokenizer is the BPE tokenizer used by OpenAI CLIP and Stable
// Diffusion 1.x's text encoder. Compared to our byte-level GPT-2 BPE this
// one:
//
//   - lowercases input;
//   - splits on whitespace and a small regex (letters / digits /
//     non-space-non-letter-non-digit punctuation);
//   - represents each "word" with a trailing `</w>` marker before
//     BPE-encoding;
//   - is *not* byte-level — CLIP ships a vocab of UTF-8 strings and you're
//     supposed to character-fall-back when a word doesn't fit any merge.
//
// CLIP's checkpoint repos (`openai/clip-vit-large-patch14`,
// `runwayml/stable-diffusion-v1-5/tokenizer`) expose a `vocab.json` and
// `merges.txt`. `LoadCLIPTokenizer` consumes those.
type CLIPTokenizer struct {
	vocab        map[string]int32
	vocabReverse map[int32]string
	merges       map[mergeKey]int32 // rank
	bos          int32              // <|startoftext|>
	eos          int32              // <|endoftext|>
	pat          *regexp.Regexp
}

// LoadCLIPTokenizer reads CLIP's two tokenizer files. `vocabReader` is the
// JSON `{token: id}` and `mergesReader` is the line-oriented merges.txt
// (lines starting with `#version:` are ignored).
func LoadCLIPTokenizer(vocabReader, mergesReader io.Reader) (*CLIPTokenizer, error) {
	var vocab map[string]int32
	if err := json.NewDecoder(vocabReader).Decode(&vocab); err != nil {
		return nil, fmt.Errorf("CLIP tokenizer: vocab.json: %w", err)
	}
	if len(vocab) == 0 {
		return nil, fmt.Errorf("CLIP tokenizer: empty vocab")
	}
	mergesBytes, err := io.ReadAll(mergesReader)
	if err != nil {
		return nil, err
	}
	mergeMap := make(map[mergeKey]int32)
	rank := int32(0)
	for _, line := range strings.Split(string(mergesBytes), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, " ", 2)
		if len(parts) != 2 {
			continue
		}
		mergeMap[mergeKey{parts[0], parts[1]}] = rank
		rank++
	}
	rev := make(map[int32]string, len(vocab))
	for tok, id := range vocab {
		rev[id] = tok
	}
	bos, ok := vocab["<|startoftext|>"]
	if !ok {
		return nil, fmt.Errorf("CLIP tokenizer: vocab missing <|startoftext|>")
	}
	eos, ok := vocab["<|endoftext|>"]
	if !ok {
		return nil, fmt.Errorf("CLIP tokenizer: vocab missing <|endoftext|>")
	}
	// Pre-tokenization regex (CLIP uses this exact form):
	// \p{L}+ | \p{N} | [^\s\p{L}\p{N}]+
	pat := regexp.MustCompile(`(\p{L}+)|(\p{N})|([^\s\p{L}\p{N}]+)`)
	return &CLIPTokenizer{
		vocab: vocab, vocabReverse: rev, merges: mergeMap,
		bos: bos, eos: eos, pat: pat,
	}, nil
}

// Encode tokenizes `text` into a length-`maxLen` slice (typically 77 for SD)
// padded with EOS. The output starts with BOS, contains the BPE pieces of
// the prompt (truncated if too long), and ends with EOS + EOS-padding.
func (t *CLIPTokenizer) Encode(text string, maxLen int) []int32 {
	return t.EncodeWithPad(text, maxLen, t.eos)
}

// EncodeWithPad is Encode with an explicit padding id: SDXL's second
// tokenizer pads with "!" (id 0) rather than EOS, and the UNet
// cross-attends to the padded positions, so the id matters.
func (t *CLIPTokenizer) EncodeWithPad(text string, maxLen int, pad int32) []int32 {
	out := []int32{t.bos}
	// Whitespace-collapse + lowercase mirrors HF's `whitespace_clean`.
	text = collapseWhitespace(strings.ToLower(strings.TrimSpace(text)))
	for _, m := range t.pat.FindAllString(text, -1) {
		// Each token is the word + the `</w>` end-of-word marker, which
		// then gets greedy-BPE-merged.
		pieces := t.bpe(m + "</w>")
		for _, p := range pieces {
			if id, ok := t.vocab[p]; ok {
				out = append(out, id)
				if len(out) >= maxLen-1 {
					break
				}
			}
		}
		if len(out) >= maxLen-1 {
			break
		}
	}
	out = append(out, t.eos)
	for len(out) < maxLen {
		out = append(out, pad)
	}
	if len(out) > maxLen {
		out = out[:maxLen]
	}
	return out
}

// Decode is the inverse of Encode for already-tokenised IDs. Reconstructs
// the surface text by concatenating piece strings and stripping the `</w>`
// markers (replacing each with a space). BOS / EOS are skipped.
func (t *CLIPTokenizer) Decode(ids []int32) string {
	var b strings.Builder
	for _, id := range ids {
		if id == t.bos || id == t.eos {
			continue
		}
		piece := t.vocabReverse[id]
		if strings.HasSuffix(piece, "</w>") {
			b.WriteString(strings.TrimSuffix(piece, "</w>"))
			b.WriteByte(' ')
		} else {
			b.WriteString(piece)
		}
	}
	return strings.TrimRight(b.String(), " ")
}

// VocabSize returns the number of entries in CLIP's vocabulary.
func (t *CLIPTokenizer) VocabSize() int { return len(t.vocab) }

// EOS / BOS getters for callers that need to construct padding manually.
func (t *CLIPTokenizer) EOS() int32 { return t.eos }
func (t *CLIPTokenizer) BOS() int32 { return t.bos }

// bpe applies greedy merges to a word. The word always ends with `</w>`;
// initial pieces are individual characters.
func (t *CLIPTokenizer) bpe(word string) []string {
	if word == "" {
		return nil
	}
	// Initial pieces: split off the `</w>` marker as a single piece on the
	// last character. CLIP's reference implementation makes the *last*
	// character carry the marker, e.g. "cat</w>" → ["c", "a", "t</w>"].
	core := strings.TrimSuffix(word, "</w>")
	if core == "" {
		return []string{word}
	}
	runes := []rune(core)
	tokens := make([]string, len(runes))
	for i, r := range runes {
		tokens[i] = string(r)
	}
	tokens[len(tokens)-1] = tokens[len(tokens)-1] + "</w>"
	// Greedy lowest-rank merges.
	for len(tokens) >= 2 {
		bestRank := int32(-1)
		bestIdx := -1
		for i := 0; i < len(tokens)-1; i++ {
			if r, ok := t.merges[mergeKey{tokens[i], tokens[i+1]}]; ok {
				if bestIdx == -1 || r < bestRank {
					bestRank = r
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

func collapseWhitespace(s string) string {
	var b strings.Builder
	prevSpace := false
	for _, r := range s {
		if unicode.IsSpace(r) {
			if !prevSpace {
				b.WriteByte(' ')
				prevSpace = true
			}
			continue
		}
		prevSpace = false
		b.WriteRune(r)
	}
	return b.String()
}
