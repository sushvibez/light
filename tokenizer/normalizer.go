package tokenizer

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"unicode"
)

// BertNormalizerConfig mirrors HF's `BertNormalizer` block in tokenizer.json.
//
// StripAccents is a tri-state like upstream: nil means "follow Lowercase"
// (HF: `strip_accents.unwrap_or(lowercase)`).
type BertNormalizerConfig struct {
	CleanText          bool  `json:"clean_text"`
	HandleChineseChars bool  `json:"handle_chinese_chars"`
	StripAccents       *bool `json:"strip_accents"`
	Lowercase          bool  `json:"lowercase"`
}

// DefaultBertNormalizer is what HF uses for BERT uncased checkpoints.
func DefaultBertNormalizer() BertNormalizerConfig {
	return BertNormalizerConfig{CleanText: true, HandleChineseChars: true, Lowercase: true}
}

// Normalize applies the four BertNormalizer steps in upstream order:
// clean_text → handle_chinese_chars → strip_accents → lowercase.
func (c BertNormalizerConfig) Normalize(text string) string {
	if c.CleanText {
		text = bertCleanText(text)
	}
	if c.HandleChineseChars {
		text = bertPadChineseChars(text)
	}
	strip := c.Lowercase
	if c.StripAccents != nil {
		strip = *c.StripAccents
	}
	if strip {
		text = stripAccents(text)
	}
	if c.Lowercase {
		text = strings.ToLower(text)
	}
	return text
}

// bertCleanText drops NUL, U+FFFD and control characters, and folds every
// whitespace rune to a plain space.
func bertCleanText(text string) string {
	var sb strings.Builder
	sb.Grow(len(text))
	for _, r := range text {
		switch {
		case r == 0 || r == 0xFFFD || isBertControl(r):
			continue
		case isBertWhitespace(r):
			sb.WriteByte(' ')
		default:
			sb.WriteRune(r)
		}
	}
	return sb.String()
}

func isBertWhitespace(r rune) bool {
	switch r {
	case ' ', '\t', '\n', '\r':
		return true
	}
	return unicode.Is(unicode.Zs, r)
}

func isBertControl(r rune) bool {
	switch r {
	case '\t', '\n', '\r':
		return false
	}
	return unicode.Is(unicode.Cc, r) || unicode.Is(unicode.Cf, r)
}

// bertPadChineseChars surrounds every CJK ideograph with spaces so the
// pre-tokenizer emits it as its own word.
func bertPadChineseChars(text string) string {
	var sb strings.Builder
	sb.Grow(len(text))
	for _, r := range text {
		if isChineseChar(r) {
			sb.WriteByte(' ')
			sb.WriteRune(r)
			sb.WriteByte(' ')
		} else {
			sb.WriteRune(r)
		}
	}
	return sb.String()
}

// isChineseChar reports whether r is in one of the CJK Unified Ideograph
// blocks BERT treats specially (same ranges as upstream `is_chinese_char`).
func isChineseChar(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) ||
		(r >= 0x3400 && r <= 0x4DBF) ||
		(r >= 0x20000 && r <= 0x2A6DF) ||
		(r >= 0x2A700 && r <= 0x2B73F) ||
		(r >= 0x2B740 && r <= 0x2B81F) ||
		(r >= 0x2B920 && r <= 0x2CEAF) ||
		(r >= 0xF900 && r <= 0xFAFF) ||
		(r >= 0x2F800 && r <= 0x2FA1F)
}

var (
	accentOnce sync.Once
	accentMap  map[rune]rune
)

func accentTable() map[rune]rune {
	accentOnce.Do(func() {
		pairs := []rune(accentPairs)
		accentMap = make(map[rune]rune, len(pairs)/2)
		for i := 0; i+1 < len(pairs); i += 2 {
			accentMap[pairs[i]] = pairs[i+1]
		}
	})
	return accentMap
}

// stripAccents emulates NFD + removal of combining marks (Mn): precomposed
// letters are replaced by their base letter via accentPairs and any
// free-standing combining marks are dropped.
func stripAccents(text string) string {
	table := accentTable()
	var sb strings.Builder
	sb.Grow(len(text))
	for _, r := range text {
		if base, ok := table[r]; ok {
			sb.WriteRune(base)
			continue
		}
		if unicode.Is(unicode.Mn, r) {
			continue
		}
		sb.WriteRune(r)
	}
	return sb.String()
}

// ---- generic normalizer chain (used by Unigram) ----

// normalizerJSON is the subset of HF normalizer configs we can apply.
// Unsupported types (Precompiled, NFKC, NFC, NFD, NFKD, …) are skipped;
// they require Unicode normalization tables we do not ship.
type normalizerJSON struct {
	Type        string           `json:"type"`
	Normalizers []normalizerJSON `json:"normalizers"`
	// Replace
	Pattern struct {
		String string `json:"String"`
		Regex  string `json:"Regex"`
	} `json:"pattern"`
	Content string `json:"content"`
	// Prepend
	Prepend string `json:"prepend"`
	// Strip
	StripLeft  bool `json:"strip_left"`
	StripRight bool `json:"strip_right"`
	// BertNormalizer
	CleanText          bool  `json:"clean_text"`
	HandleChineseChars bool  `json:"handle_chinese_chars"`
	StripAccents       *bool `json:"strip_accents"`
	Lowercase          bool  `json:"lowercase"`
}

// textNormalizer is a compiled chain of simple normalization steps.
type textNormalizer struct {
	steps []func(string) string
}

func (n *textNormalizer) apply(text string) string {
	for _, step := range n.steps {
		text = step(text)
	}
	return text
}

// compileNormalizer builds a textNormalizer from a tokenizer.json
// `normalizer` block (which may be null). It returns an empty chain for
// unknown types rather than failing, so loading stays permissive.
func compileNormalizer(raw json.RawMessage) (*textNormalizer, error) {
	n := &textNormalizer{}
	if len(raw) == 0 || string(raw) == "null" {
		return n, nil
	}
	var cfg normalizerJSON
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("tokenizer: normalizer: %w", err)
	}
	if err := n.addStep(&cfg); err != nil {
		return nil, err
	}
	return n, nil
}

func (n *textNormalizer) addStep(cfg *normalizerJSON) error {
	switch cfg.Type {
	case "Sequence":
		for i := range cfg.Normalizers {
			if err := n.addStep(&cfg.Normalizers[i]); err != nil {
				return err
			}
		}
	case "Lowercase":
		n.steps = append(n.steps, strings.ToLower)
	case "Prepend":
		p := cfg.Prepend
		n.steps = append(n.steps, func(s string) string {
			if s == "" || strings.HasPrefix(s, p) {
				return s
			}
			return p + s
		})
	case "Strip":
		left, right := cfg.StripLeft, cfg.StripRight
		n.steps = append(n.steps, func(s string) string {
			if left {
				s = strings.TrimLeftFunc(s, unicode.IsSpace)
			}
			if right {
				s = strings.TrimRightFunc(s, unicode.IsSpace)
			}
			return s
		})
	case "Replace":
		content := cfg.Content
		if cfg.Pattern.Regex != "" {
			re, err := regexp.Compile(cfg.Pattern.Regex)
			if err != nil {
				return fmt.Errorf("tokenizer: Replace normalizer regex %q: %w", cfg.Pattern.Regex, err)
			}
			n.steps = append(n.steps, func(s string) string { return re.ReplaceAllLiteralString(s, content) })
		} else if cfg.Pattern.String != "" {
			old := cfg.Pattern.String
			n.steps = append(n.steps, func(s string) string { return strings.ReplaceAll(s, old, content) })
		}
	case "BertNormalizer":
		b := BertNormalizerConfig{
			CleanText:          cfg.CleanText,
			HandleChineseChars: cfg.HandleChineseChars,
			StripAccents:       cfg.StripAccents,
			Lowercase:          cfg.Lowercase,
		}
		n.steps = append(n.steps, b.Normalize)
	}
	return nil
}
