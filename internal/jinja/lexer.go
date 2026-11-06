package jinja

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

type tokKind int

const (
	tEOF tokKind = iota
	tData
	tVarBegin
	tVarEnd
	tBlockBegin
	tBlockEnd
	tName
	tString
	tInt
	tFloat
	tOp
)

func (k tokKind) String() string {
	return [...]string{"end of template", "template data", "'{{'", "'}}'", "'{%'", "'%}'", "name", "string", "integer", "float", "operator"}[k]
}

type token struct {
	kind tokKind
	val  string // data / name / op / raw string value (unescaped)
	num  any    // int64 or float64 for tInt / tFloat
	line int
}

func (t token) String() string {
	switch t.kind {
	case tName, tOp:
		return "'" + t.val + "'"
	case tString:
		return strconv.Quote(t.val)
	case tInt, tFloat:
		return fmt.Sprint(t.num)
	}
	return t.kind.String()
}

// isPySpace reports Python's str.isspace for one rune (Go's unicode.IsSpace
// plus the ASCII file/group/record/unit separators).
func isPySpace(r rune) bool { return unicode.IsSpace(r) || (r >= 0x1c && r <= 0x1f) }

func pyRStrip(s string) string { return strings.TrimRightFunc(s, isPySpace) }

func pyLStrip(s string) string { return strings.TrimLeftFunc(s, isPySpace) }

// lexer mirrors jinja2.lexer.Lexer.tokeniter for the default delimiters,
// including trim_blocks / lstrip_blocks / '-' and '+' whitespace control.
type lexer struct {
	src  string
	pos  int
	line int
	opt  Options
	toks []token
	// lineStarting is jinja's flag: the previous match ended with "\n".
	lineStarting bool
}

func tokenize(src string, opt Options) ([]token, error) {
	// Newline normalisation and the single-trailing-newline drop happen on
	// the whole source first, exactly like jinja.
	src = strings.ReplaceAll(src, "\r\n", "\n")
	src = strings.ReplaceAll(src, "\r", "\n")
	if !opt.KeepTrailingNewline && strings.HasSuffix(src, "\n") {
		src = src[:len(src)-1]
	}
	lx := &lexer{src: src, line: 1, opt: opt, lineStarting: true}
	if err := lx.run(); err != nil {
		return nil, err
	}
	lx.toks = append(lx.toks, token{kind: tEOF, line: lx.line})
	return lx.toks, nil
}

func (lx *lexer) errf(format string, a ...any) error {
	return &Error{Line: lx.line, Msg: fmt.Sprintf(format, a...)}
}

func (lx *lexer) emit(k tokKind, v string) {
	lx.toks = append(lx.toks, token{kind: k, val: v, line: lx.line})
}

// nextTag finds the next "{{", "{%" or "{#" at or after pos.
func (lx *lexer) nextTag() (int, byte) {
	s := lx.src
	for i := lx.pos; i+1 < len(s); i++ {
		if s[i] == '{' {
			switch s[i+1] {
			case '{', '%', '#':
				return i, s[i+1]
			}
		}
	}
	return -1, 0
}

// stripData applies '-' / lstrip_blocks to the data before a tag.
func (lx *lexer) stripData(text string, sign byte, isVar bool) string {
	if sign == '-' {
		return pyRStrip(text)
	}
	if sign != '+' && lx.opt.LstripBlocks && !isVar {
		lpos := strings.LastIndexByte(text, '\n') + 1
		if lpos > 0 || lx.lineStarting {
			rest := text[lpos:]
			if rest != "" && strings.TrimFunc(rest, isPySpace) == "" {
				return text[:lpos]
			}
		}
	}
	return text
}

func (lx *lexer) run() error {
	for lx.pos < len(lx.src) {
		start, kind := lx.nextTag()
		if start < 0 {
			lx.emitData(lx.src[lx.pos:])
			lx.pos = len(lx.src)
			return nil
		}
		text := lx.src[lx.pos:start]
		p := start + 2
		var sign byte
		if p < len(lx.src) && (lx.src[p] == '-' || lx.src[p] == '+') {
			sign = lx.src[p]
			p++
		}
		if kind == '%' {
			if end, body, ok := lx.matchRaw(p); ok {
				lx.emitData(lx.stripData(text, sign, false))
				lx.pos = end
				lx.lineStarting = false
				if err := lx.lexRaw(body); err != nil {
					return err
				}
				continue
			}
		}
		lx.emitData(lx.stripData(text, sign, kind == '{'))
		lx.pos = p
		lx.lineStarting = false
		var err error
		switch kind {
		case '#':
			err = lx.lexComment()
		case '%':
			lx.emit(tBlockBegin, "")
			err = lx.lexTag(false)
		case '{':
			lx.emit(tVarBegin, "")
			err = lx.lexTag(true)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

func (lx *lexer) emitData(s string) {
	if s != "" {
		lx.emit(tData, s)
		lx.line += strings.Count(s, "\n")
	}
}

func (lx *lexer) advance(n int) {
	lx.line += strings.Count(lx.src[lx.pos:lx.pos+n], "\n")
	lx.pos += n
}

func skipSpace(s string, i int) int {
	for i < len(s) {
		r, n := utf8.DecodeRuneInString(s[i:])
		if !isPySpace(r) {
			break
		}
		i += n
	}
	return i
}

// matchRaw recognises `{% raw %}` (p is just past "{%" and its sign);
// it returns the end of the opening tag.
func (lx *lexer) matchRaw(p int) (int, int, bool) {
	s := lx.src
	i := skipSpace(s, p)
	if !strings.HasPrefix(s[i:], "raw") {
		return 0, 0, false
	}
	i += 3
	j := skipSpace(s, i)
	switch {
	case strings.HasPrefix(s[j:], "-%}"):
		k := skipSpace(s, j+3)
		return k, k, true
	case strings.HasPrefix(s[j:], "%}"):
		return j + 2, j + 2, true
	}
	return 0, 0, false
}

func (lx *lexer) lexRaw(body int) error {
	s := lx.src
	for i := body; i+1 < len(s); i++ {
		if s[i] != '{' || s[i+1] != '%' {
			continue
		}
		p := i + 2
		var sign byte
		if p < len(s) && (s[p] == '-' || s[p] == '+') {
			sign = s[p]
			p++
		}
		q := skipSpace(s, p)
		if !strings.HasPrefix(s[q:], "endraw") {
			continue
		}
		q = skipSpace(s, q+6)
		var end int
		switch {
		case strings.HasPrefix(s[q:], "+%}"):
			end = q + 3
		case strings.HasPrefix(s[q:], "-%}"):
			end = skipSpace(s, q+3)
		case strings.HasPrefix(s[q:], "%}"):
			end = q + 2
			if lx.opt.TrimBlocks && end < len(s) && s[end] == '\n' {
				end++
			}
		default:
			continue
		}
		lx.emitData(lx.stripData(s[body:i], sign, false))
		lx.line += strings.Count(s[i:end], "\n")
		lx.lineStarting = strings.HasSuffix(s[lx.pos:end], "\n")
		lx.pos = end
		return nil
	}
	return lx.errf("missing end of raw directive")
}

func (lx *lexer) lexComment() error {
	s := lx.src
	for i := lx.pos; i+1 < len(s); i++ {
		var end int
		switch {
		case strings.HasPrefix(s[i:], "+#}"):
			end = i + 3
		case strings.HasPrefix(s[i:], "-#}"):
			end = skipSpace(s, i+3)
		case strings.HasPrefix(s[i:], "#}"):
			end = i + 2
			if lx.opt.TrimBlocks && end < len(s) && s[end] == '\n' {
				end++
			}
		default:
			continue
		}
		lx.lineStarting = strings.HasSuffix(s[i:end], "\n")
		lx.advance(end - lx.pos)
		return nil
	}
	return lx.errf("missing end of comment tag")
}

var operators = []string{"//", "**", "==", "!=", "<=", ">=", "+", "-", "*", "/", "%", "~", "[", "]", "(", ")", "{", "}", "<", ">", "=", ".", ":", "|", ",", ";"}

func (lx *lexer) lexTag(isVar bool) error {
	s := lx.src
	var balance []byte
	for {
		if lx.pos >= len(s) {
			if isVar {
				return lx.errf("unexpected end of template, expected '}}'")
			}
			return lx.errf("unexpected end of template, expected '%%}'")
		}
		// End of tag (only when brackets are balanced).
		if len(balance) == 0 {
			if end, ok := lx.matchTagEnd(isVar); ok {
				matched := s[lx.pos:end]
				lx.lineStarting = strings.HasSuffix(matched, "\n")
				if isVar {
					lx.emit(tVarEnd, "")
				} else {
					lx.emit(tBlockEnd, "")
				}
				lx.advance(end - lx.pos)
				return nil
			}
		}
		c := s[lx.pos]
		r, rn := utf8.DecodeRuneInString(s[lx.pos:])
		switch {
		case isPySpace(r):
			lx.advance(rn)
		case c >= '0' && c <= '9':
			if err := lx.lexNumber(); err != nil {
				return err
			}
		case c == '_' || unicode.IsLetter(r):
			j := lx.pos
			for j < len(s) {
				r2, n2 := utf8.DecodeRuneInString(s[j:])
				if r2 != '_' && !unicode.IsLetter(r2) && !unicode.IsDigit(r2) {
					break
				}
				j += n2
			}
			lx.emit(tName, s[lx.pos:j])
			lx.pos = j
		case c == '\'' || c == '"':
			if err := lx.lexString(c); err != nil {
				return err
			}
		default:
			op := ""
			for _, o := range operators {
				if strings.HasPrefix(s[lx.pos:], o) {
					op = o
					break
				}
			}
			if op == "" {
				return lx.errf("unexpected char %q", r)
			}
			switch op {
			case "(":
				balance = append(balance, ')')
			case "[":
				balance = append(balance, ']')
			case "{":
				balance = append(balance, '}')
			case ")", "]", "}":
				if len(balance) == 0 {
					return lx.errf("unexpected '%s'", op)
				}
				if want := balance[len(balance)-1]; want != op[0] {
					return lx.errf("unexpected '%s', expected '%c'", op, want)
				}
				balance = balance[:len(balance)-1]
			}
			lx.emit(tOp, op)
			lx.pos += len(op)
		}
	}
}

func (lx *lexer) matchTagEnd(isVar bool) (int, bool) {
	s := lx.src[lx.pos:]
	if isVar {
		switch {
		case strings.HasPrefix(s, "-}}"):
			return skipSpace(lx.src, lx.pos+3), true
		case strings.HasPrefix(s, "}}"):
			return lx.pos + 2, true
		}
		return 0, false
	}
	switch {
	case strings.HasPrefix(s, "+%}"):
		return lx.pos + 3, true
	case strings.HasPrefix(s, "-%}"):
		return skipSpace(lx.src, lx.pos+3), true
	case strings.HasPrefix(s, "%}"):
		end := lx.pos + 2
		if lx.opt.TrimBlocks && end < len(lx.src) && lx.src[end] == '\n' {
			end++
		}
		return end, true
	}
	return 0, false
}

func (lx *lexer) lexNumber() error {
	s := lx.src
	i := lx.pos
	digits := func() {
		for i < len(s) && (s[i] >= '0' && s[i] <= '9' || s[i] == '_') {
			i++
		}
	}
	digits()
	isFloat := false
	// jinja's float regex has a (?<!\.) guard: "x.0.1" style attribute
	// access stays integer.
	afterDot := lx.pos > 0 && s[lx.pos-1] == '.'
	if !afterDot && i+1 < len(s) && s[i] == '.' && s[i+1] >= '0' && s[i+1] <= '9' {
		isFloat = true
		i++
		digits()
	}
	if !afterDot && i < len(s) && (s[i] == 'e' || s[i] == 'E') {
		j := i + 1
		if j < len(s) && (s[j] == '+' || s[j] == '-') {
			j++
		}
		if j < len(s) && s[j] >= '0' && s[j] <= '9' {
			isFloat = true
			i = j
			digits()
		}
	}
	lit := strings.ReplaceAll(s[lx.pos:i], "_", "")
	if isFloat {
		f, err := strconv.ParseFloat(lit, 64)
		if err != nil {
			return lx.errf("bad float literal %q", lit)
		}
		lx.toks = append(lx.toks, token{kind: tFloat, val: lit, num: f, line: lx.line})
	} else {
		n, err := strconv.ParseInt(lit, 10, 64)
		if err != nil {
			return lx.errf("bad integer literal %q", lit)
		}
		lx.toks = append(lx.toks, token{kind: tInt, val: lit, num: n, line: lx.line})
	}
	lx.pos = i
	return nil
}

func (lx *lexer) lexString(q byte) error {
	s := lx.src
	i := lx.pos + 1
	for i < len(s) && s[i] != q {
		if s[i] == '\\' {
			i++
		}
		i++
	}
	if i >= len(s) {
		return lx.errf("unterminated string")
	}
	raw := s[lx.pos+1 : i]
	val, err := pyUnescape(raw)
	if err != nil {
		return lx.errf("%v", err)
	}
	lx.toks = append(lx.toks, token{kind: tString, val: val, line: lx.line})
	lx.advance(i + 1 - lx.pos)
	return nil
}

// pyUnescape decodes Python "unicode-escape" sequences (what jinja applies
// to string literals). Unknown escapes keep their backslash.
func pyUnescape(s string) (string, error) {
	if !strings.Contains(s, `\`) {
		return s, nil
	}
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c != '\\' || i+1 >= len(s) {
			b.WriteByte(c)
			continue
		}
		i++
		switch e := s[i]; e {
		case '\n':
		case '\\', '\'', '"':
			b.WriteByte(e)
		case 'a':
			b.WriteByte('\a')
		case 'b':
			b.WriteByte('\b')
		case 'f':
			b.WriteByte('\f')
		case 'n':
			b.WriteByte('\n')
		case 'r':
			b.WriteByte('\r')
		case 't':
			b.WriteByte('\t')
		case 'v':
			b.WriteByte('\v')
		case '0', '1', '2', '3', '4', '5', '6', '7':
			j := i
			for j < len(s) && j < i+3 && s[j] >= '0' && s[j] <= '7' {
				j++
			}
			n, _ := strconv.ParseUint(s[i:j], 8, 32)
			b.WriteRune(rune(n))
			i = j - 1
		case 'x', 'u', 'U':
			w := map[byte]int{'x': 2, 'u': 4, 'U': 8}[e]
			if i+1+w > len(s) {
				return "", fmt.Errorf("truncated \\%c escape", e)
			}
			n, err := strconv.ParseUint(s[i+1:i+1+w], 16, 32)
			if err != nil {
				return "", fmt.Errorf("truncated \\%c escape", e)
			}
			b.WriteRune(rune(n))
			i += w
		case 'N':
			return "", fmt.Errorf(`\N{...} escapes are not supported`)
		default:
			b.WriteByte('\\')
			b.WriteByte(e)
		}
	}
	return b.String(), nil
}
