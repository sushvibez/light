package tokenizer

import "strings"

// specialSet is the registry of atomic ("added") tokens shared by the
// WordPiece and Unigram tokenizers. A registered surface form is matched
// verbatim in the raw input, before any normalization or pre-tokenization,
// exactly like HF's added-token splitting with normalized=false.
type specialSet struct {
	ids   map[string]int32
	byLen []string // surface forms sorted longest first for greedy matching
}

func newSpecialSet() specialSet {
	return specialSet{ids: map[string]int32{}}
}

// add registers text → id, keeping byLen sorted longest-first.
func (s *specialSet) add(text string, id int32) {
	if text == "" {
		return
	}
	if _, exists := s.ids[text]; !exists {
		s.byLen = append(s.byLen, text)
		for i := len(s.byLen) - 1; i > 0; i-- {
			if len(s.byLen[i]) > len(s.byLen[i-1]) {
				s.byLen[i], s.byLen[i-1] = s.byLen[i-1], s.byLen[i]
			} else {
				break
			}
		}
	}
	s.ids[text] = id
}

func (s *specialSet) contains(text string) bool {
	_, ok := s.ids[text]
	return ok
}

// split walks text left-to-right; at each position the longest registered
// special-token prefix wins. Text between matches is emitted as plain
// segments. With no specials registered this returns one plain segment.
func (s *specialSet) split(text string) []segment {
	if len(s.byLen) == 0 {
		return []segment{{text: text}}
	}
	var out []segment
	var buf strings.Builder
	i := 0
	for i < len(text) {
		matched := ""
		for _, sp := range s.byLen {
			if strings.HasPrefix(text[i:], sp) {
				matched = sp
				break
			}
		}
		if matched != "" {
			if buf.Len() > 0 {
				out = append(out, segment{text: buf.String()})
				buf.Reset()
			}
			out = append(out, segment{id: s.ids[matched], hit: true})
			i += len(matched)
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
