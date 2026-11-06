package jinja

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Python methods reachable from templates as obj.method(...). Mutating
// methods are refused the way jinja2's ImmutableSandboxedEnvironment
// (what transformers uses) refuses them.

func unsafeMethod(obj any, name string) Func {
	return func([]any, *Dict) (any, error) {
		return nil, fmt.Errorf("access to attribute '%s' of '%s' object is unsafe", name, typeName(obj))
	}
}

func argStr(args []any, i int, fn string) (string, bool, error) {
	if i >= len(args) || args[i] == nil {
		return "", false, nil
	}
	s, ok := args[i].(string)
	if !ok {
		return "", false, fmt.Errorf("%s arg must be None or str, not %s", fn, typeName(args[i]))
	}
	return s, true, nil
}

func argInt(args []any, i int, def int64, fn string) (int64, error) {
	if i >= len(args) || args[i] == nil {
		return def, nil
	}
	switch v := args[i].(type) {
	case int64:
		return v, nil
	case bool:
		if v {
			return 1, nil
		}
		return 0, nil
	}
	return 0, fmt.Errorf("%s: an integer is required, not %s", fn, typeName(args[i]))
}

// kwToArgs moves keyword arguments into positional slots by name.
func kwToArgs(fn string, args []any, kw *Dict, names ...string) ([]any, error) {
	if kw == nil || kw.Len() == 0 {
		if len(args) > len(names) {
			return nil, fmt.Errorf("%s() takes at most %d arguments (%d given)", fn, len(names), len(args))
		}
		return args, nil
	}
	out := make([]any, len(names))
	copy(out, args)
	if len(args) > len(names) {
		return nil, fmt.Errorf("%s() takes at most %d arguments (%d given)", fn, len(names), len(args))
	}
	for _, k := range kw.Keys() {
		ks, _ := k.(string)
		idx := -1
		for i, n := range names {
			if n == ks {
				idx = i
			}
		}
		if idx < 0 {
			return nil, fmt.Errorf("%s() got an unexpected keyword argument '%s'", fn, ks)
		}
		if idx < len(args) {
			return nil, fmt.Errorf("%s() got multiple values for argument '%s'", fn, ks)
		}
		out[idx], _ = kw.Get(k)
	}
	// Trailing unset slots are nil (= default).
	return out, nil
}

func pyStrip(s string, chars *string, left, right bool) string {
	f := isPySpace
	if chars != nil {
		set := *chars
		f = func(r rune) bool { return strings.ContainsRune(set, r) }
	}
	if left {
		s = strings.TrimLeftFunc(s, f)
	}
	if right {
		s = strings.TrimRightFunc(s, f)
	}
	return s
}

func pySplitWhitespace(s string, maxsplit int) []any {
	var out []any
	i := 0
	for {
		for i < len(s) {
			r, n := utf8.DecodeRuneInString(s[i:])
			if !isPySpace(r) {
				break
			}
			i += n
		}
		if i >= len(s) {
			return out
		}
		if maxsplit >= 0 && len(out) == maxsplit {
			return append(out, s[i:])
		}
		j := i
		for j < len(s) {
			r, n := utf8.DecodeRuneInString(s[j:])
			if isPySpace(r) {
				break
			}
			j += n
		}
		out = append(out, s[i:j])
		i = j
	}
}

func pyRSplitWhitespace(s string, maxsplit int) []any {
	var rev []any
	j := len(s)
	for {
		for j > 0 {
			r, n := utf8.DecodeLastRuneInString(s[:j])
			if !isPySpace(r) {
				break
			}
			j -= n
		}
		if j <= 0 {
			break
		}
		if maxsplit >= 0 && len(rev) == maxsplit {
			rev = append(rev, s[:j])
			break
		}
		i := j
		for i > 0 {
			r, n := utf8.DecodeLastRuneInString(s[:i])
			if isPySpace(r) {
				break
			}
			i -= n
		}
		rev = append(rev, s[i:j])
		j = i
	}
	out := make([]any, len(rev))
	for i := range rev {
		out[i] = rev[len(rev)-1-i]
	}
	return out
}

func strList(ss []string) *List {
	l := &List{Items: make([]any, len(ss))}
	for i, s := range ss {
		l.Items[i] = s
	}
	return l
}

func isCased(r rune) bool { return unicode.IsUpper(r) || unicode.IsLower(r) || unicode.IsTitle(r) }

func pyTitle(s string) string {
	var b strings.Builder
	prev := false
	for _, r := range s {
		if prev {
			b.WriteRune(unicode.ToLower(r))
		} else {
			b.WriteRune(unicode.ToTitle(r))
		}
		prev = isCased(r)
	}
	return b.String()
}

func pyCapitalize(s string) string {
	r, n := utf8.DecodeRuneInString(s)
	if n == 0 {
		return s
	}
	return string(unicode.ToTitle(r)) + strings.ToLower(s[n:])
}

// pySplitlines splits on Python's line boundaries.
func pySplitlines(s string, keepends bool) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); {
		r, n := utf8.DecodeRuneInString(s[i:])
		switch r {
		case '\n', '\r', '\v', '\f', 0x1c, 0x1d, 0x1e, 0x85, 0x2028, 0x2029:
			end := i + n
			if r == '\r' && end < len(s) && s[end] == '\n' {
				end++
			}
			if keepends {
				out = append(out, s[start:end])
			} else {
				out = append(out, s[start:i])
			}
			start = end
			i = end
			continue
		}
		i += n
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}

func prefixArgs(args []any, fn string) ([]string, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("%s() takes exactly one argument (%d given)", fn, len(args))
	}
	switch a := args[0].(type) {
	case string:
		return []string{a}, nil
	case *List:
		if a.Tuple {
			var out []string
			for _, it := range a.Items {
				s, ok := it.(string)
				if !ok {
					return nil, fmt.Errorf("tuple for %s must only contain str", fn)
				}
				out = append(out, s)
			}
			return out, nil
		}
	}
	return nil, fmt.Errorf("%s first arg must be str or a tuple of str, not %s", fn, typeName(args[0]))
}

func runeIndex(s string, byteIdx int) int64 {
	if byteIdx < 0 {
		return -1
	}
	return int64(utf8.RuneCountInString(s[:byteIdx]))
}

func strMethod(s, name string) (Func, bool) {
	var f Func
	switch name {
	case "strip", "lstrip", "rstrip":
		f = func(args []any, kw *Dict) (any, error) {
			args, err := kwToArgs(name, args, kw, "chars")
			if err != nil {
				return nil, err
			}
			chars, ok, err := argStr(args, 0, name)
			if err != nil {
				return nil, err
			}
			var cp *string
			if ok {
				cp = &chars
			}
			return pyStrip(s, cp, name != "rstrip", name != "lstrip"), nil
		}
	case "split", "rsplit":
		f = func(args []any, kw *Dict) (any, error) {
			args, err := kwToArgs(name, args, kw, "sep", "maxsplit")
			if err != nil {
				return nil, err
			}
			sep, hasSep, err := argStr(args, 0, name)
			if err != nil {
				return nil, err
			}
			maxsplit, err := argInt(args, 1, -1, name)
			if err != nil {
				return nil, err
			}
			if !hasSep {
				if name == "split" {
					return &List{Items: pySplitWhitespace(s, int(maxsplit))}, nil
				}
				return &List{Items: pyRSplitWhitespace(s, int(maxsplit))}, nil
			}
			if sep == "" {
				return nil, fmt.Errorf("empty separator")
			}
			if maxsplit < 0 {
				return strList(strings.Split(s, sep)), nil
			}
			if name == "split" {
				return strList(strings.SplitN(s, sep, int(maxsplit)+1)), nil
			}
			var parts []string
			rest := s
			for i := int64(0); i < maxsplit; i++ {
				j := strings.LastIndex(rest, sep)
				if j < 0 {
					break
				}
				parts = append([]string{rest[j+len(sep):]}, parts...)
				rest = rest[:j]
			}
			return strList(append([]string{rest}, parts...)), nil
		}
	case "startswith", "endswith":
		f = func(args []any, kw *Dict) (any, error) {
			ps, err := prefixArgs(args, name)
			if err != nil {
				return nil, err
			}
			for _, p := range ps {
				if (name == "startswith" && strings.HasPrefix(s, p)) || (name == "endswith" && strings.HasSuffix(s, p)) {
					return true, nil
				}
			}
			return false, nil
		}
	case "upper":
		f = func([]any, *Dict) (any, error) { return strings.ToUpper(s), nil }
	case "lower":
		f = func([]any, *Dict) (any, error) { return strings.ToLower(s), nil }
	case "title":
		f = func([]any, *Dict) (any, error) { return pyTitle(s), nil }
	case "capitalize":
		f = func([]any, *Dict) (any, error) { return pyCapitalize(s), nil }
	case "replace":
		f = func(args []any, kw *Dict) (any, error) {
			args, err := kwToArgs(name, args, kw, "old", "new", "count")
			if err != nil {
				return nil, err
			}
			if len(args) < 2 {
				return nil, fmt.Errorf("replace expected at least 2 arguments, got %d", len(args))
			}
			o, ok1 := args[0].(string)
			n, ok2 := args[1].(string)
			if !ok1 || !ok2 {
				return nil, fmt.Errorf("replace() arguments must be str")
			}
			cnt, err := argInt(args, 2, -1, name)
			if err != nil {
				return nil, err
			}
			return strings.Replace(s, o, n, int(cnt)), nil
		}
	case "find", "rfind", "index", "rindex", "count":
		f = func(args []any, kw *Dict) (any, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("%s() with start/end arguments is not supported", name)
			}
			sub, ok := args[0].(string)
			if !ok {
				return nil, fmt.Errorf("must be str, not %s", typeName(args[0]))
			}
			var i int
			switch name {
			case "count":
				if sub == "" {
					return int64(utf8.RuneCountInString(s) + 1), nil
				}
				return int64(strings.Count(s, sub)), nil
			case "find", "index":
				i = strings.Index(s, sub)
			default:
				i = strings.LastIndex(s, sub)
			}
			if i < 0 && (name == "index" || name == "rindex") {
				return nil, fmt.Errorf("substring not found")
			}
			return runeIndex(s, i), nil
		}
	case "join":
		f = func(args []any, kw *Dict) (any, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("join() takes exactly one argument")
			}
			items, err := iterate(args[0])
			if err != nil {
				return nil, err
			}
			parts := make([]string, len(items))
			for i, it := range items {
				p, ok := it.(string)
				if !ok {
					return nil, fmt.Errorf("sequence item %d: expected str instance, %s found", i, typeName(it))
				}
				parts[i] = p
			}
			return strings.Join(parts, s), nil
		}
	case "splitlines":
		f = func(args []any, kw *Dict) (any, error) {
			args, err := kwToArgs(name, args, kw, "keepends")
			if err != nil {
				return nil, err
			}
			keep := len(args) > 0 && truthy(args[0])
			return strList(pySplitlines(s, keep)), nil
		}
	case "removeprefix":
		f = func(args []any, kw *Dict) (any, error) {
			p, _, err := argStr(args, 0, name)
			return strings.TrimPrefix(s, p), err
		}
	case "removesuffix":
		f = func(args []any, kw *Dict) (any, error) {
			p, _, err := argStr(args, 0, name)
			return strings.TrimSuffix(s, p), err
		}
	case "isdigit", "isnumeric", "isdecimal", "isalpha", "isalnum", "isspace", "isupper", "islower":
		f = func([]any, *Dict) (any, error) {
			if s == "" {
				return false, nil
			}
			cased := false
			for _, r := range s {
				var ok bool
				switch name {
				case "isdigit", "isnumeric", "isdecimal":
					ok = unicode.IsDigit(r)
				case "isalpha":
					ok = unicode.IsLetter(r)
				case "isalnum":
					ok = unicode.IsLetter(r) || unicode.IsDigit(r)
				case "isspace":
					ok = isPySpace(r)
				case "isupper":
					ok = !unicode.IsLower(r)
					cased = cased || unicode.IsUpper(r)
				case "islower":
					ok = !unicode.IsUpper(r)
					cased = cased || unicode.IsLower(r)
				}
				if !ok {
					return false, nil
				}
			}
			if name == "isupper" || name == "islower" {
				return cased, nil
			}
			return true, nil
		}
	case "format":
		f = func(args []any, kw *Dict) (any, error) { return pyStrFormat(s, args, kw) }
	default:
		return nil, false
	}
	return f, true
}

// pyStrFormat supports "{}", "{0}" and "{name}" fields without format specs.
func pyStrFormat(s string, args []any, kw *Dict) (any, error) {
	var b strings.Builder
	auto := 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == '}' {
			if i+1 < len(s) && s[i+1] == '}' {
				i++
			}
			b.WriteByte('}')
			continue
		}
		if c != '{' {
			b.WriteByte(c)
			continue
		}
		if i+1 < len(s) && s[i+1] == '{' {
			b.WriteByte('{')
			i++
			continue
		}
		j := strings.IndexByte(s[i:], '}')
		if j < 0 {
			return nil, fmt.Errorf("single '{' encountered in format string")
		}
		field := s[i+1 : i+j]
		i += j
		if strings.ContainsAny(field, ":!.[") {
			return nil, fmt.Errorf("str.format field %q is not supported", field)
		}
		var v any
		switch {
		case field == "":
			if auto >= len(args) {
				return nil, fmt.Errorf("Replacement index %d out of range", auto)
			}
			v = args[auto]
			auto++
		case field[0] >= '0' && field[0] <= '9':
			var n int
			fmt.Sscanf(field, "%d", &n)
			if n >= len(args) {
				return nil, fmt.Errorf("Replacement index %d out of range", n)
			}
			v = args[n]
		default:
			var ok bool
			if v, ok = kw.Get(field); !ok {
				return nil, fmt.Errorf("KeyError: '%s'", field)
			}
		}
		b.WriteString(toStr(v))
	}
	return b.String(), nil
}

func dictMethod(d *Dict, name string) (Func, bool) {
	switch name {
	case "items":
		return func([]any, *Dict) (any, error) {
			l := &List{Items: make([]any, 0, d.Len())}
			for _, k := range d.Keys() {
				v, _ := d.Get(k)
				l.Items = append(l.Items, &List{Items: []any{k, v}, Tuple: true})
			}
			return l, nil
		}, true
	case "keys":
		return func([]any, *Dict) (any, error) {
			return &List{Items: append([]any{}, d.Keys()...)}, nil
		}, true
	case "values":
		return func([]any, *Dict) (any, error) {
			l := &List{Items: make([]any, 0, d.Len())}
			for _, k := range d.Keys() {
				v, _ := d.Get(k)
				l.Items = append(l.Items, v)
			}
			return l, nil
		}, true
	case "get":
		return func(args []any, kw *Dict) (any, error) {
			if len(args) < 1 || len(args) > 2 {
				return nil, fmt.Errorf("get expected 1 or 2 arguments, got %d", len(args))
			}
			if v, ok := d.Get(args[0]); ok {
				return v, nil
			}
			if len(args) == 2 {
				return args[1], nil
			}
			return nil, nil
		}, true
	case "copy":
		return func([]any, *Dict) (any, error) {
			c := NewDict()
			for _, k := range d.Keys() {
				v, _ := d.Get(k)
				c.Set(k, v)
			}
			return c, nil
		}, true
	case "pop", "popitem", "setdefault", "update", "clear":
		return unsafeMethod(d, name), true
	}
	return nil, false
}

func listMethod(l *List, name string) (Func, bool) {
	switch name {
	case "index", "count":
		return func(args []any, kw *Dict) (any, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("%s() takes exactly one argument", name)
			}
			n := int64(0)
			for i, it := range l.Items {
				if equal(it, args[0]) {
					if name == "index" {
						return int64(i), nil
					}
					n++
				}
			}
			if name == "index" {
				return nil, fmt.Errorf("%s is not in list", repr(args[0]))
			}
			return n, nil
		}, true
	case "append", "extend", "insert", "pop", "remove", "reverse", "sort", "clear":
		if l.Tuple {
			return nil, false
		}
		return unsafeMethod(l, name), true
	}
	return nil, false
}
