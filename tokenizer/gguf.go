package tokenizer

import (
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	"light/gguf"
)

// GGUFVocab carries the non-vocabulary tokenizer metadata of a GGUF file.
type GGUFVocab struct {
	Model          string // "llama" (SentencePiece) or "gpt2" (byte-level BPE)
	Pre            string // tokenizer.ggml.pre, pre-tokenizer family (BPE only)
	BOS, EOS       int32  // -1 when absent
	UNK, PAD       int32  // -1 when absent
	AddBOS, AddEOS bool
	AddSpacePrefix bool
	ChatTemplate   string
}

// Token-type values of tokenizer.ggml.token_type (llama_token_attr).
const (
	ggufTokenNormal      = 1
	ggufTokenUnknown     = 2
	ggufTokenControl     = 3
	ggufTokenUserDefined = 4
	ggufTokenUnused      = 5
	ggufTokenByte        = 6
)

// LoadGGUF builds a tokenizer from the `tokenizer.ggml.*` metadata of a GGUF
// file, so quantized checkpoints need no sidecar tokenizer.json. Encoding
// follows llama.cpp (src/llama-vocab.cpp): SentencePiece vocabularies merge
// by piece score, byte-level BPE vocabularies by merge rank after the
// family-specific pre-tokenizer regex chain selected by tokenizer.ggml.pre.
func LoadGGUF(path string) (*BPE, *GGUFVocab, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	g, err := gguf.Open(f)
	if err != nil {
		return nil, nil, fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	return FromGGUFMeta(g.Meta)
}

// FromGGUFMeta is LoadGGUF on an already-parsed metadata map.
func FromGGUFMeta(meta map[string]any) (*BPE, *GGUFVocab, error) {
	info := &GGUFVocab{BOS: -1, EOS: -1, UNK: -1, PAD: -1}
	info.Model, _ = meta["tokenizer.ggml.model"].(string)
	info.Pre, _ = meta["tokenizer.ggml.pre"].(string)
	info.ChatTemplate, _ = meta["tokenizer.chat_template"].(string)
	info.BOS = metaInt(meta, "tokenizer.ggml.bos_token_id", -1)
	info.EOS = metaInt(meta, "tokenizer.ggml.eos_token_id", -1)
	info.UNK = metaInt(meta, "tokenizer.ggml.unknown_token_id", -1)
	info.PAD = metaInt(meta, "tokenizer.ggml.padding_token_id", -1)

	tokens, err := metaStrings(meta, "tokenizer.ggml.tokens")
	if err != nil {
		return nil, nil, err
	}
	types := metaInts(meta, "tokenizer.ggml.token_type", len(tokens))
	vocab := make(map[string]int32, len(tokens))
	for i, t := range tokens {
		vocab[t] = int32(i)
	}

	var bpe *BPE
	switch info.Model {
	case "llama":
		info.AddBOS = metaBool(meta, "tokenizer.ggml.add_bos_token", true)
		info.AddEOS = metaBool(meta, "tokenizer.ggml.add_eos_token", false)
		info.AddSpacePrefix = metaBool(meta, "tokenizer.ggml.add_space_prefix", true)
		scores := metaFloats(meta, "tokenizer.ggml.scores", len(tokens))
		lines, ranks := spmMerges(tokens, scores, types)
		bpe, err = NewBPE(vocab, lines)
		if err != nil {
			return nil, nil, err
		}
		for i, l := range lines { // dense score ranks (NewBPE assigned positions)
			ab := strings.SplitN(l, " ", 2)
			bpe.merges[mergeKey{ab[0], ab[1]}] = ranks[i]
		}
		norm := &textNormalizer{}
		if info.AddSpacePrefix {
			norm.steps = append(norm.steps, func(s string) string { return "▁" + s })
		}
		norm.steps = append(norm.steps, func(s string) string { return strings.ReplaceAll(s, " ", "▁") })
		unk := ""
		if info.UNK >= 0 && int(info.UNK) < len(tokens) {
			unk = tokens[info.UNK]
		}
		bpe.SetSentencePiece(norm, true, unk)
		bpe.spmStripLeading = info.AddSpacePrefix
	case "gpt2":
		info.AddBOS = metaBool(meta, "tokenizer.ggml.add_bos_token", false)
		info.AddEOS = metaBool(meta, "tokenizer.ggml.add_eos_token", false)
		merges, err := metaStrings(meta, "tokenizer.ggml.merges")
		if err != nil {
			return nil, nil, err
		}
		bpe, err = NewBPE(vocab, merges)
		if err != nil {
			return nil, nil, err
		}
		pre := info.Pre
		if pre == "" {
			pre = "default"
		}
		fam, ok := ggufPreFamilies[pre]
		if !ok {
			return nil, nil, fmt.Errorf("tokenizer: unsupported GGUF pre-tokenizer %q", pre)
		}
		split, err := newRegexChain(fam.regexes)
		if err != nil {
			return nil, nil, err
		}
		bpe.SetPreTokenizer(split)
		bpe.SetIgnoreMerges(fam.ignoreMerges)
		if fam.addBOS {
			info.AddBOS = metaBool(meta, "tokenizer.ggml.add_bos_token", true)
		}
	case "":
		return nil, nil, fmt.Errorf("tokenizer: GGUF has no tokenizer.ggml.model")
	default:
		return nil, nil, fmt.Errorf("tokenizer: unsupported GGUF tokenizer model %q", info.Model)
	}
	for i, t := range tokens {
		if types[i] == ggufTokenControl || types[i] == ggufTokenUserDefined {
			bpe.RegisterSpecial(t, int32(i))
		}
	}
	return bpe, info, nil
}

// spmMerges derives ranked merges from SentencePiece scores: llama.cpp
// repeatedly merges the adjacent pair whose concatenation is the vocabulary
// piece with the highest score (ties: leftmost). Every split of every
// normal piece into two vocabulary pieces becomes a merge; ranks are the
// dense rank of the piece's score so that equal scores share a rank and
// mergeSymbols' leftmost preference reproduces the tie-break.
func spmMerges(tokens []string, scores []float32, types []int32) (lines []string, ranks []int32) {
	vocab := make(map[string]struct{}, len(tokens))
	for _, t := range tokens {
		vocab[t] = struct{}{}
	}
	type cand struct {
		a, b  string
		score float32
	}
	var cands []cand
	for i, t := range tokens {
		if types[i] != ggufTokenNormal || utf8.RuneCountInString(t) < 2 {
			continue
		}
		for j := 1; j < len(t); j++ {
			if !utf8.RuneStart(t[j]) {
				continue
			}
			a, b := t[:j], t[j:]
			if _, ok := vocab[a]; !ok {
				continue
			}
			if _, ok := vocab[b]; !ok {
				continue
			}
			cands = append(cands, cand{a, b, scores[i]})
		}
	}
	sort.SliceStable(cands, func(i, j int) bool { return cands[i].score > cands[j].score })
	lines = make([]string, len(cands))
	ranks = make([]int32, len(cands))
	rank := int32(-1)
	for i, c := range cands {
		if i == 0 || c.score != cands[i-1].score {
			rank++
		}
		lines[i] = c.a + " " + c.b
		ranks[i] = rank
	}
	return lines, ranks
}

func metaStrings(meta map[string]any, key string) ([]string, error) {
	arr, ok := meta[key].([]any)
	if !ok {
		return nil, fmt.Errorf("tokenizer: GGUF metadata %s missing", key)
	}
	out := make([]string, len(arr))
	for i, v := range arr {
		s, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("tokenizer: GGUF %s[%d] is %T, want string", key, i, v)
		}
		out[i] = s
	}
	return out, nil
}

func metaInts(meta map[string]any, key string, n int) []int32 {
	out := make([]int32, n)
	for i := range out {
		out[i] = ggufTokenNormal
	}
	if arr, ok := meta[key].([]any); ok {
		for i, v := range arr {
			if i < n {
				out[i] = anyInt(v, ggufTokenNormal)
			}
		}
	}
	return out
}

func metaFloats(meta map[string]any, key string, n int) []float32 {
	out := make([]float32, n)
	if arr, ok := meta[key].([]any); ok {
		for i, v := range arr {
			if i < n {
				switch f := v.(type) {
				case float32:
					out[i] = f
				case float64:
					out[i] = float32(f)
				}
			}
		}
	}
	return out
}

func metaInt(meta map[string]any, key string, def int32) int32 {
	v, ok := meta[key]
	if !ok {
		return def
	}
	return anyInt(v, def)
}

func metaBool(meta map[string]any, key string, def bool) bool {
	if v, ok := meta[key].(bool); ok {
		return v
	}
	return def
}

func anyInt(v any, def int32) int32 {
	switch x := v.(type) {
	case int32:
		return x
	case uint32:
		return int32(x)
	case int64:
		return int32(x)
	case uint64:
		return int32(x)
	case int8:
		return int32(x)
	case uint8:
		return int32(x)
	case int16:
		return int32(x)
	case uint16:
		return int32(x)
	case int:
		return int32(x)
	}
	return def
}

// ---- pre-tokenizer regex chains (llama.cpp llm_tokenizer_bpe) ----

type ggufPreFamily struct {
	regexes      []string
	ignoreMerges bool
	addBOS       bool
}

// The patterns are llama.cpp's, written for a lookahead-capable engine; see
// newRegexChain for how `\s+(?!\S)` and Unicode whitespace are handled on RE2.
const (
	reGPT2   = `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)`
	reLlama3 = `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	reQwen2  = `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	reQwen35 = `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	rePoro   = ` ?[^(\s|.,!?…。，、।۔،)]+`
	reDSLat  = `\s?[A-Za-z\x{B5}\x{C0}-\x{D6}\x{D8}-\x{F6}\x{F8}-\x{1BA}\x{1BC}-\x{1BF}\x{1C4}-\x{293}\x{295}-\x{2AF}\x{370}-\x{373}\x{376}\x{377}\x{37B}-\x{37D}\x{37F}\x{386}\x{388}-\x{38A}\x{38C}\x{38E}-\x{3A1}\x{3A3}-\x{3F5}\x{3F7}-\x{481}\x{48A}-\x{52F}\x{531}-\x{556}\x{10A0}-\x{10C5}\x{13A0}-\x{13F5}\x{13F8}-\x{13FD}\x{1C90}-\x{1CBA}\x{1CBD}-\x{1CBF}\x{1D00}-\x{1D2B}\x{1D6B}-\x{1D77}\x{1D79}-\x{1D9A}\x{1E00}-\x{1F15}\x{1F18}-\x{1F1D}\x{1F20}-\x{1F45}\x{1F48}-\x{1F4D}\x{1F50}-\x{1F57}\x{1F59}\x{1F5B}\x{1F5D}\x{1F5F}-\x{1F7D}\x{1F80}-\x{1FB4}\x{1FB6}-\x{1FBC}\x{1FBE}\x{1FC2}-\x{1FC4}\x{1FC6}-\x{1FCC}\x{1FD0}-\x{1FD3}\x{1FD6}-\x{1FDB}\x{1FE0}-\x{1FEC}\x{1FF2}-\x{1FF4}\x{1FF6}-\x{1FFC}\x{2102}\x{2107}\x{210A}-\x{2113}\x{2115}\x{2119}-\x{211D}\x{2124}\x{2126}\x{2128}\x{212A}-\x{212D}\x{212F}-\x{2134}\x{2139}\x{213C}-\x{213F}\x{2145}-\x{2149}\x{214E}\x{2183}\x{2184}\x{2C00}-\x{2C7B}\x{2C7E}-\x{2CE4}\x{2CEB}-\x{2CEE}\x{2CF2}\x{2CF3}\x{A640}-\x{A66D}\x{A680}-\x{A69B}\x{A722}-\x{A76F}\x{A771}-\x{A787}\x{A78B}-\x{A78E}\x{AB70}-\x{ABBF}\x{FB00}-\x{FB06}\x{FB13}-\x{FB17}\x{FF21}-\x{FF3A}\x{FF41}-\x{FF5A}\x{10400}-\x{1044F}\x{104B0}-\x{104D3}\x{104D8}-\x{104FB}\x{10C80}-\x{10CB2}\x{10CC0}-\x{10CF2}\x{118A0}-\x{118DF}\x{1E900}-\x{1E943}]+`
)

var ggufPreFamilies = func() map[string]ggufPreFamily {
	m := map[string]ggufPreFamily{}
	add := func(fam ggufPreFamily, names ...string) {
		for _, n := range names {
			m[n] = fam
		}
	}
	add(ggufPreFamily{regexes: []string{`[\p{P}\$\+<=>\^~\|]+`, reGPT2, `\p{N}+`, `[0-9][0-9][0-9]`}}, "default")
	add(ggufPreFamily{regexes: []string{reLlama3}, ignoreMerges: true, addBOS: true},
		"llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1", "pixtral", "midm-2.0", "lfm2", "jina-v5-nano")
	add(ggufPreFamily{regexes: []string{reLlama3}}, "dbrx", "smaug-bpe", "glm4", "chatglm-bpe")
	add(ggufPreFamily{regexes: []string{`[\r\n]`, reDSLat, `\s?[!-/:-~\x{FF01}-\x{FF0F}\x{FF1A}-\x{FF5E}\x{2018}-\x{201F}\x{3000}-\x{3002}]+`, `\s+$`, `[\x{4E00}-\x{9FA5}\x{800}-\x{4E00}\x{AC00}-\x{D7FF}]+`, `\p{N}+`}}, "deepseek-llm")
	add(ggufPreFamily{regexes: []string{`[\r\n]`, `\s?\p{L}+`, `\s?\p{P}+`, `[\x{4E00}-\x{9FA5}\x{800}-\x{4E00}\x{AC00}-\x{D7FF}]+`, `\p{N}`}}, "deepseek-coder")
	add(ggufPreFamily{regexes: []string{`[\p{P}\$\+<=>\^~\|` + "`" + `]+`, reGPT2, `[0-9][0-9][0-9]`}}, "falcon")
	add(ggufPreFamily{regexes: []string{`\p{N}`, reGPT2}},
		"starcoder", "refact", "command-r", "smollm", "codeshell", "exaone", "minerva-7b", "mellum2")
	add(ggufPreFamily{regexes: []string{reGPT2}},
		"gpt-2", "phi-2", "jina-es", "jina-de", "gigachat", "jina-v2-es", "jina-v2-de", "a.x-4.0", "mellum", "modern-bert",
		"jina-v1-en", "jina-v2-code", "roberta-bpe", "mpt", "olmo", "jais")
	add(ggufPreFamily{regexes: []string{reQwen2}}, "qwen2", "deepseek-r1-qwen", "kormo", "f2llmv2", "stablelm2", "hunyuan", "solar-open")
	add(ggufPreFamily{regexes: []string{reQwen35}}, "qwen35")
	add(ggufPreFamily{regexes: []string{rePoro}}, "poro-chat", "bloom", "gpt3-finnish")
	add(ggufPreFamily{regexes: []string{rePoro, `\p{N}`}}, "viking")
	return m
}()

// Unicode White_Space for RE2, whose `\s` is ASCII-only. `\p{Z}` covers the
// separator categories (NBSP, en/em spaces, ideographic space, line/para
// separators); NEL and VT are added by hand.
const reWS = `\s\p{Z}\x{85}\x{0B}`

// chainStep is one regex of a llama.cpp pre-tokenizer chain.
type chainStep struct {
	re     *regexp.Regexp // pattern with `\s+(?!\S)` rewritten to `\s+`
	before *regexp.Regexp // anchored alternatives preceding `\s+(?!\S)`, nil when absent
}

// newRegexChain compiles a llama.cpp regex chain for RE2 and returns the
// splitter: each regex is applied to every piece produced so far; matches
// and the unmatched gaps between them both become pieces
// (unicode_regex_split_stl). `\s+(?!\S)` ("whitespace not followed by
// non-whitespace") has no RE2 equivalent: it is compiled as `\s+` and a
// whitespace-only match of two or more runes that is followed by a
// non-space is shortened by one rune, unless an earlier alternative of the
// same pattern already produced that match.
func newRegexChain(patterns []string) (func(string) []string, error) {
	steps := make([]chainStep, len(patterns))
	for i, p := range patterns {
		var st chainStep
		const la = `\s+(?!\S)`
		if k := strings.Index(p, la); k >= 0 {
			if k > 0 {
				before, err := regexp.Compile(`^(?:` + re2Whitespace(strings.TrimSuffix(p[:k], "|")) + `)`)
				if err != nil {
					return nil, fmt.Errorf("tokenizer: pre-tokenizer regex %q: %w", p, err)
				}
				st.before = before
			}
			p = strings.Replace(p, la, `\s+`, 1)
		}
		re, err := regexp.Compile(re2Whitespace(p))
		if err != nil {
			return nil, fmt.Errorf("tokenizer: pre-tokenizer regex %q: %w", p, err)
		}
		st.re = re
		steps[i] = st
	}
	return func(text string) []string {
		pieces := []string{text}
		for i := range steps {
			var next []string
			for _, piece := range pieces {
				next = steps[i].split(piece, next)
			}
			pieces = next
		}
		return pieces
	}, nil
}

func (st chainStep) split(piece string, out []string) []string {
	pos := 0
	for pos < len(piece) {
		loc := st.re.FindStringIndex(piece[pos:])
		if loc == nil {
			break
		}
		s, e := pos+loc[0], pos+loc[1]
		if e == s { // empty match: advance one rune to avoid looping
			_, w := utf8.DecodeRuneInString(piece[s:])
			if s+w > len(piece) {
				break
			}
			pos = s + w
			continue
		}
		if st.before != nil && e < len(piece) && isAllSpace(piece[s:e]) && utf8.RuneCountInString(piece[s:e]) >= 2 {
			r, _ := utf8.DecodeRuneInString(piece[e:])
			if !isSpaceRune(r) {
				if m := st.before.FindStringIndex(piece[s:]); m == nil || m[1] != e-s {
					_, w := utf8.DecodeLastRuneInString(piece[s:e])
					e -= w
				}
			}
		}
		if s > pos {
			out = append(out, piece[pos:s])
		}
		out = append(out, piece[s:e])
		pos = e
	}
	if pos < len(piece) {
		out = append(out, piece[pos:])
	}
	return out
}

func isSpaceRune(r rune) bool {
	switch r {
	case ' ', '\t', '\n', '\v', '\f', '\r', 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000:
		return true
	}
	return r >= 0x2000 && r <= 0x200A
}

func isAllSpace(s string) bool {
	for _, r := range s {
		if !isSpaceRune(r) {
			return false
		}
	}
	return true
}

// re2Whitespace rewrites `\s` to the Unicode White_Space set, inside and
// outside character classes.
func re2Whitespace(p string) string {
	var b strings.Builder
	inClass := false
	for i := 0; i < len(p); i++ {
		c := p[i]
		switch {
		case c == '\\' && i+1 < len(p):
			if p[i+1] == 's' {
				if inClass {
					b.WriteString(reWS)
				} else {
					b.WriteString(`[` + reWS + `]`)
				}
			} else {
				b.WriteByte(c)
				b.WriteByte(p[i+1])
			}
			i++
		case c == '[' && !inClass:
			inClass = true
			b.WriteByte(c)
		case c == ']' && inClass:
			inClass = false
			b.WriteByte(c)
		default:
			b.WriteByte(c)
		}
	}
	return b.String()
}
