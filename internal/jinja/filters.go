package jinja

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

type filterFn func(r *renderer, v any, args []any, kw *Dict) (any, error)

// filters holds jinja2's builtin filters that chat templates use, with the
// transformers override of tojson. Anything else is a parse error.
var filters map[string]filterFn

func init() {
	filters = map[string]filterFn{
		"abs":        fAbs,
		"attr":       fAttr,
		"capitalize": strFilter(pyCapitalize),
		"count":      fLength,
		"d":          fDefault,
		"default":    fDefault,
		"dictsort":   fDictsort,
		"e":          fEscape,
		"escape":     fEscape,
		"first":      fFirst,
		"float":      fFloat,
		"format":     fFormat,
		"indent":     fIndent,
		"int":        fInt,
		"items":      fItems,
		"join":       fJoin,
		"last":       fLast,
		"length":     fLength,
		"list":       fList,
		"lower":      strFilter(strings.ToLower),
		"map":        fMap,
		"max":        fMinMax(1),
		"min":        fMinMax(-1),
		"reject":     fSelect(false, false),
		"rejectattr": fSelect(false, true),
		"replace":    fReplace,
		"reverse":    fReverse,
		"round":      fRound,
		"safe":       func(_ *renderer, v any, _ []any, _ *Dict) (any, error) { return v, nil },
		"select":     fSelect(true, false),
		"selectattr": fSelect(true, true),
		"sort":       fSort,
		"string":     func(_ *renderer, v any, _ []any, _ *Dict) (any, error) { return toStr(v), nil },
		"sum":        fSum,
		"title":      strFilter(jinjaTitle),
		"tojson":     fToJSON,
		"trim":       fTrim,
		"unique":     fUnique,
		"upper":      strFilter(strings.ToUpper),
		"wordcount":  fWordcount,
	}
}

func (r *renderer) callFilter(f *filterExpr, v any, sc *scope) (any, error) {
	args, kw, err := r.evalArgs(f.args, f.kwargs, sc)
	if err != nil {
		return nil, err
	}
	return filters[f.name](r, v, args, kw)
}

// bindArgs maps positional + keyword filter arguments onto named
// parameters; missing ones are nil.
func bindArgs(name string, args []any, kw *Dict, params ...string) ([]any, error) {
	if len(args) > len(params) {
		return nil, fmt.Errorf("filter '%s' takes at most %d arguments", name, len(params))
	}
	out := make([]any, len(params))
	copy(out, args)
	set := make([]bool, len(params))
	for i := range args {
		set[i] = true
	}
	for _, k := range kw.Keys() {
		ks, _ := k.(string)
		found := false
		for i, p := range params {
			if p == ks {
				if set[i] {
					return nil, fmt.Errorf("filter '%s' got multiple values for '%s'", name, ks)
				}
				out[i], _ = kw.Get(k)
				set[i] = true
				found = true
			}
		}
		if !found {
			return nil, fmt.Errorf("filter '%s' got an unexpected keyword argument '%s'", name, ks)
		}
	}
	return out, nil
}

func strFilter(fn func(string) string) filterFn {
	return func(_ *renderer, v any, _ []any, _ *Dict) (any, error) { return fn(toStr(v)), nil }
}

func fAbs(_ *renderer, v any, _ []any, _ *Dict) (any, error) {
	switch v := v.(type) {
	case int64:
		if v < 0 {
			return -v, nil
		}
		return v, nil
	case float64:
		return math.Abs(v), nil
	case bool:
		if v {
			return int64(1), nil
		}
		return int64(0), nil
	}
	return nil, fmt.Errorf("bad operand type for abs(): '%s'", typeName(v))
}

func fAttr(_ *renderer, v any, args []any, _ *Dict) (any, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("attr filter takes one argument")
	}
	name, ok := args[0].(string)
	if !ok {
		return nil, fmt.Errorf("attribute name must be a string")
	}
	switch o := v.(type) {
	case *Dict:
		// attr() is getattr only: dict keys are not attributes.
		if m, ok := dictMethod(o, name); ok {
			return m, nil
		}
		return Undefined{Name: name, Obj: "dict object"}, nil
	}
	return getattr(v, name)
}

func fLength(_ *renderer, v any, _ []any, _ *Dict) (any, error) {
	n, err := length(v)
	return int64(n), err
}

func fDefault(_ *renderer, v any, args []any, kw *Dict) (any, error) {
	a, err := bindArgs("default", args, kw, "default_value", "boolean")
	if err != nil {
		return nil, err
	}
	def := a[0]
	if def == nil && len(args) == 0 {
		if _, ok := kw.Get("default_value"); !ok {
			def = ""
		}
	}
	_, undef := v.(Undefined)
	if undef || (truthy(a[1]) && !truthy(v)) {
		return def, nil
	}
	return v, nil
}

func sortKey(v any, caseSensitive bool) any {
	if s, ok := v.(string); ok && !caseSensitive {
		return strings.ToLower(s)
	}
	return v
}

func fDictsort(_ *renderer, v any, args []any, kw *Dict) (any, error) {
	a, err := bindArgs("dictsort", args, kw, "case_sensitive", "by", "reverse")
	if err != nil {
		return nil, err
	}
	d, ok := v.(*Dict)
	if !ok {
		return nil, fmt.Errorf("dictsort filter needs a mapping, got %s", typeName(v))
	}
	pos := 0
	if by, _ := a[1].(string); by == "value" {
		pos = 1
	} else if a[1] != nil && by != "key" {
		return nil, fmt.Errorf("You can only sort by either \"key\" or \"value\"")
	}
	items, _ := dictMethod(d, "items")
	lv, _ := items(nil, nil)
	pairs := append([]any(nil), lv.(*List).Items...)
	var cerr error
	sort.SliceStable(pairs, func(i, j int) bool {
		x := sortKey(pairs[i].(*List).Items[pos], truthy(a[0]))
		y := sortKey(pairs[j].(*List).Items[pos], truthy(a[0]))
		c, err := compare(x, y)
		if err != nil {
			cerr = err
		}
		if truthy(a[2]) {
			return c > 0
		}
		return c < 0
	})
	return &List{Items: pairs}, cerr
}

var htmlEscaper = strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&#34;", "'", "&#39;")

func fEscape(_ *renderer, v any, _ []any, _ *Dict) (any, error) {
	return htmlEscaper.Replace(toStr(v)), nil
}

func fFirst(_ *renderer, v any, _ []any, _ *Dict) (any, error) {
	items, err := iterate(v)
	if err != nil {
		return nil, err
	}
	if len(items) == 0 {
		return Undefined{Name: "No first item, sequence was empty."}, nil
	}
	return items[0], nil
}

func fLast(_ *renderer, v any, _ []any, _ *Dict) (any, error) {
	items, err := iterate(v)
	if err != nil {
		return nil, err
	}
	if len(items) == 0 {
		return Undefined{Name: "No last item, sequence was empty."}, nil
	}
	return items[len(items)-1], nil
}

func fFloat(_ *renderer, v any, args []any, kw *Dict) (any, error) {
	a, err := bindArgs("float", args, kw, "default")
	if err != nil {
		return nil, err
	}
	def := a[0]
	if def == nil {
		def = 0.0
	}
	switch x := v.(type) {
	case float64:
		return x, nil
	case int64:
		return float64(x), nil
	case bool:
		if x {
			return 1.0, nil
		}
		return 0.0, nil
	case string:
		if f, err := strconv.ParseFloat(strings.TrimFunc(x, isPySpace), 64); err == nil {
			return f, nil
		}
	}
	return def, nil
}

func fInt(_ *renderer, v any, args []any, kw *Dict) (any, error) {
	a, err := bindArgs("int", args, kw, "default", "base")
	if err != nil {
		return nil, err
	}
	def := a[0]
	if def == nil {
		def = int64(0)
	}
	base := int64(10)
	if b, ok := a[1].(int64); ok {
		base = b
	}
	switch x := v.(type) {
	case int64:
		return x, nil
	case bool:
		if x {
			return int64(1), nil
		}
		return int64(0), nil
	case float64:
		return int64(x), nil
	case string:
		s := strings.ReplaceAll(strings.TrimFunc(x, isPySpace), "_", "")
		if n, err := strconv.ParseInt(s, int(base), 64); err == nil {
			return n, nil
		}
		if f, err := strconv.ParseFloat(s, 64); err == nil {
			return int64(f), nil
		}
	}
	return def, nil
}

func fFormat(_ *renderer, v any, args []any, kw *Dict) (any, error) {
	if kw.Len() > 0 {
		return nil, fmt.Errorf("format filter with keyword arguments is not supported")
	}
	return pyPercentFormat(toStr(v), &List{Items: args, Tuple: true})
}

func fIndent(_ *renderer, v any, args []any, kw *Dict) (any, error) {
	a, err := bindArgs("indent", args, kw, "width", "first", "blank")
	if err != nil {
		return nil, err
	}
	ind := "    "
	switch w := a[0].(type) {
	case int64:
		ind = strings.Repeat(" ", int(w))
	case string:
		ind = w
	}
	s := toStr(v) + "\n"
	var rv string
	if truthy(a[2]) {
		rv = strings.Join(pySplitlines(s, false), "\n"+ind)
	} else {
		lines := pySplitlines(s, false)
		rv = lines[0]
		if len(lines) > 1 {
			rest := make([]string, len(lines)-1)
			for i, l := range lines[1:] {
				if l != "" {
					l = ind + l
				}
				rest[i] = l
			}
			rv += "\n" + strings.Join(rest, "\n")
		}
	}
	if truthy(a[1]) {
		rv = ind + rv
	}
	return rv, nil
}

func fItems(_ *renderer, v any, _ []any, _ *Dict) (any, error) {
	switch d := v.(type) {
	case Undefined:
		return &List{}, nil
	case *Dict:
		m, _ := dictMethod(d, "items")
		return m(nil, nil)
	}
	return nil, fmt.Errorf("Can only get item pairs from a mapping.")
}

// attrGetter resolves jinja's dotted "attribute" arguments ("a.b", "0").
func attrGetter(attr any) func(any) (any, error) {
	if attr == nil {
		return func(v any) (any, error) { return v, nil }
	}
	var parts []any
	if s, ok := attr.(string); ok {
		for _, p := range strings.Split(s, ".") {
			if n, err := strconv.ParseInt(p, 10, 64); err == nil {
				parts = append(parts, n)
			} else {
				parts = append(parts, p)
			}
		}
	} else {
		parts = []any{attr}
	}
	return func(v any) (any, error) {
		for _, p := range parts {
			var err error
			if v, err = getitem(v, p); err != nil {
				return nil, err
			}
		}
		return v, nil
	}
}

func fJoin(_ *renderer, v any, args []any, kw *Dict) (any, error) {
	a, err := bindArgs("join", args, kw, "d", "attribute")
	if err != nil {
		return nil, err
	}
	sep := ""
	if a[0] != nil {
		sep = toStr(a[0])
	}
	items, err := iterate(v)
	if err != nil {
		return nil, err
	}
	get := attrGetter(a[1])
	parts := make([]string, len(items))
	for i, it := range items {
		x, err := get(it)
		if err != nil {
			return nil, err
		}
		parts[i] = toStr(x)
	}
	return strings.Join(parts, sep), nil
}

func fList(_ *renderer, v any, _ []any, _ *Dict) (any, error) {
	items, err := iterate(v)
	if err != nil {
		return nil, err
	}
	return &List{Items: append([]any{}, items...)}, nil
}

func fMap(r *renderer, v any, args []any, kw *Dict) (any, error) {
	items, err := iterate(v)
	if err != nil {
		return nil, err
	}
	out := &List{Items: make([]any, 0, len(items))}
	if len(args) == 0 {
		attr, ok := kw.Get("attribute")
		if !ok {
			return nil, fmt.Errorf("map requires a filter name or attribute=")
		}
		def, hasDef := kw.Get("default")
		get := attrGetter(attr)
		for _, it := range items {
			x, err := get(it)
			if err != nil {
				return nil, err
			}
			if _, undef := x.(Undefined); undef && hasDef {
				x = def
			}
			out.Items = append(out.Items, x)
		}
		return out, nil
	}
	name, ok := args[0].(string)
	if !ok {
		return nil, fmt.Errorf("map filter name must be a string")
	}
	fn, ok := filters[name]
	if !ok {
		return nil, fmt.Errorf("no filter named '%s'", name)
	}
	for _, it := range items {
		x, err := fn(r, it, args[1:], kw)
		if err != nil {
			return nil, err
		}
		out.Items = append(out.Items, x)
	}
	return out, nil
}

func fMinMax(sign int) filterFn {
	return func(_ *renderer, v any, args []any, kw *Dict) (any, error) {
		a, err := bindArgs("min/max", args, kw, "case_sensitive", "attribute")
		if err != nil {
			return nil, err
		}
		items, err := iterate(v)
		if err != nil {
			return nil, err
		}
		if len(items) == 0 {
			return Undefined{Name: "No aggregated item, sequence was empty."}, nil
		}
		get := attrGetter(a[1])
		best := items[0]
		bk, err := get(best)
		if err != nil {
			return nil, err
		}
		bk = sortKey(bk, truthy(a[0]))
		for _, it := range items[1:] {
			k, err := get(it)
			if err != nil {
				return nil, err
			}
			k = sortKey(k, truthy(a[0]))
			c, err := compare(k, bk)
			if err != nil {
				return nil, err
			}
			if c*sign > 0 {
				best, bk = it, k
			}
		}
		return best, nil
	}
}

func fSelect(keep, byAttr bool) filterFn {
	return func(_ *renderer, v any, args []any, kw *Dict) (any, error) {
		if kw.Len() > 0 {
			return nil, fmt.Errorf("select/reject filters take no keyword arguments")
		}
		items, err := iterate(v)
		if err != nil {
			return nil, err
		}
		get := attrGetter(nil)
		if byAttr {
			if len(args) == 0 {
				return nil, fmt.Errorf("Missing parameter for attribute name")
			}
			get = attrGetter(args[0])
			args = args[1:]
		}
		test := ""
		if len(args) > 0 {
			s, ok := args[0].(string)
			if !ok {
				return nil, fmt.Errorf("test name must be a string")
			}
			if _, ok := tests[s]; !ok {
				return nil, fmt.Errorf("no test named '%s'", s)
			}
			test, args = s, args[1:]
		}
		out := &List{}
		for _, it := range items {
			x, err := get(it)
			if err != nil {
				return nil, err
			}
			var ok bool
			if test == "" {
				ok = truthy(x)
			} else if ok, err = runTest(test, x, args); err != nil {
				return nil, err
			}
			if ok == keep {
				out.Items = append(out.Items, it)
			}
		}
		return out, nil
	}
}

func fReplace(_ *renderer, v any, args []any, kw *Dict) (any, error) {
	a, err := bindArgs("replace", args, kw, "old", "new", "count")
	if err != nil {
		return nil, err
	}
	n := -1
	if c, ok := a[2].(int64); ok {
		n = int(c)
	}
	return strings.Replace(toStr(v), toStr(a[0]), toStr(a[1]), n), nil
}

func fReverse(_ *renderer, v any, _ []any, _ *Dict) (any, error) {
	if s, ok := v.(string); ok {
		rs := []rune(s)
		for i, j := 0, len(rs)-1; i < j; i, j = i+1, j-1 {
			rs[i], rs[j] = rs[j], rs[i]
		}
		return string(rs), nil
	}
	items, err := iterate(v)
	if err != nil {
		return nil, err
	}
	out := &List{Items: make([]any, len(items))}
	for i, it := range items {
		out.Items[len(items)-1-i] = it
	}
	return out, nil
}

func fRound(_ *renderer, v any, args []any, kw *Dict) (any, error) {
	a, err := bindArgs("round", args, kw, "precision", "method")
	if err != nil {
		return nil, err
	}
	prec := int64(0)
	if p, ok := a[0].(int64); ok {
		prec = p
	}
	_, f, _, ok := numeric(v)
	if !ok {
		return nil, fmt.Errorf("round filter needs a number, got %s", typeName(v))
	}
	scale := math.Pow(10, float64(prec))
	method, _ := a[1].(string)
	switch method {
	case "", "common":
		return math.RoundToEven(f*scale) / scale, nil
	case "ceil":
		return math.Ceil(f*scale) / scale, nil
	case "floor":
		return math.Floor(f*scale) / scale, nil
	}
	return nil, fmt.Errorf("method must be common, ceil or floor")
}

func fSort(_ *renderer, v any, args []any, kw *Dict) (any, error) {
	a, err := bindArgs("sort", args, kw, "reverse", "case_sensitive", "attribute")
	if err != nil {
		return nil, err
	}
	items, err := iterate(v)
	if err != nil {
		return nil, err
	}
	out := append([]any{}, items...)
	get := attrGetter(a[2])
	keys := make([]any, len(out))
	for i, it := range out {
		k, err := get(it)
		if err != nil {
			return nil, err
		}
		keys[i] = sortKey(k, truthy(a[1]))
	}
	idx := make([]int, len(out))
	for i := range idx {
		idx[i] = i
	}
	var cerr error
	sort.SliceStable(idx, func(i, j int) bool {
		c, err := compare(keys[idx[i]], keys[idx[j]])
		if err != nil {
			cerr = err
		}
		if truthy(a[0]) {
			return c > 0
		}
		return c < 0
	})
	res := &List{Items: make([]any, len(out))}
	for i, j := range idx {
		res.Items[i] = out[j]
	}
	return res, cerr
}

func fSum(_ *renderer, v any, args []any, kw *Dict) (any, error) {
	a, err := bindArgs("sum", args, kw, "attribute", "start")
	if err != nil {
		return nil, err
	}
	var total any = int64(0)
	if a[1] != nil {
		total = a[1]
	}
	items, err := iterate(v)
	if err != nil {
		return nil, err
	}
	get := attrGetter(a[0])
	for _, it := range items {
		x, err := get(it)
		if err != nil {
			return nil, err
		}
		if total, err = binop("+", total, x); err != nil {
			return nil, err
		}
	}
	return total, nil
}

var wordBeginningSplit = regexp.MustCompile(`[-\s({\[<]+`)

// jinjaTitle is jinja2's do_title (not Python's str.title).
func jinjaTitle(s string) string {
	var b strings.Builder
	last := 0
	emit := func(item string) {
		if item == "" {
			return
		}
		r, n := utf8.DecodeRuneInString(item)
		b.WriteString(strings.ToUpper(string(r)))
		b.WriteString(strings.ToLower(item[n:]))
	}
	for _, m := range wordBeginningSplit.FindAllStringIndex(s, -1) {
		emit(s[last:m[0]])
		emit(s[m[0]:m[1]])
		last = m[1]
	}
	emit(s[last:])
	return b.String()
}

// fToJSON is transformers' tojson override:
// json.dumps(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False).
func fToJSON(_ *renderer, v any, args []any, kw *Dict) (any, error) {
	a, err := bindArgs("tojson", args, kw, "ensure_ascii", "indent", "separators", "sort_keys")
	if err != nil {
		return nil, err
	}
	o := jsonOpts{ensureASCII: truthy(a[0]), sortKeys: truthy(a[3]), itemSep: ", ", keySep: ": "}
	switch ind := a[1].(type) {
	case nil:
	case int64:
		o.hasIndent, o.indent, o.itemSep = true, strings.Repeat(" ", int(max(0, ind))), ","
	case string:
		o.hasIndent, o.indent, o.itemSep = true, ind, ","
	default:
		return nil, fmt.Errorf("tojson indent must be an int or str")
	}
	if a[2] != nil {
		l, ok := a[2].(*List)
		if !ok || len(l.Items) != 2 {
			return nil, fmt.Errorf("tojson separators must be a pair of strings")
		}
		s1, ok1 := l.Items[0].(string)
		s2, ok2 := l.Items[1].(string)
		if !ok1 || !ok2 {
			return nil, fmt.Errorf("tojson separators must be a pair of strings")
		}
		o.itemSep, o.keySep = s1, s2
	}
	return toJSON(v, o)
}

func fTrim(_ *renderer, v any, args []any, kw *Dict) (any, error) {
	a, err := bindArgs("trim", args, kw, "chars")
	if err != nil {
		return nil, err
	}
	var cp *string
	if s, ok := a[0].(string); ok {
		cp = &s
	}
	return pyStrip(toStr(v), cp, true, true), nil
}

func fUnique(_ *renderer, v any, args []any, kw *Dict) (any, error) {
	a, err := bindArgs("unique", args, kw, "case_sensitive", "attribute")
	if err != nil {
		return nil, err
	}
	items, err := iterate(v)
	if err != nil {
		return nil, err
	}
	get := attrGetter(a[1])
	out := &List{}
	var seen []any
outer:
	for _, it := range items {
		k, err := get(it)
		if err != nil {
			return nil, err
		}
		k = sortKey(k, truthy(a[0]))
		for _, s := range seen {
			if equal(s, k) {
				continue outer
			}
		}
		seen = append(seen, k)
		out.Items = append(out.Items, it)
	}
	return out, nil
}

var wordRe = regexp.MustCompile(`\w+`)

func fWordcount(_ *renderer, v any, _ []any, _ *Dict) (any, error) {
	return int64(len(wordRe.FindAllString(toStr(v), -1))), nil
}

// ---- tests -----------------------------------------------------------------

type testFn func(v any, args []any) (bool, error)

var tests map[string]testFn

func init() {
	cmp := func(op string) testFn {
		return func(v any, args []any) (bool, error) {
			if len(args) != 1 {
				return false, fmt.Errorf("test takes one argument")
			}
			return compareOp(op, v, args[0])
		}
	}
	isInt := func(v any) bool { _, ok := v.(int64); return ok }
	tests = map[string]testFn{
		"boolean":  func(v any, _ []any) (bool, error) { _, ok := v.(bool); return ok, nil },
		"callable": func(v any, _ []any) (bool, error) { _, f := v.(Func); _, m := v.(*macroValue); return f || m, nil },
		"defined":  func(v any, _ []any) (bool, error) { _, u := v.(Undefined); return !u, nil },
		"divisibleby": func(v any, args []any) (bool, error) {
			if len(args) != 1 {
				return false, fmt.Errorf("divisibleby takes one argument")
			}
			m, err := binop("%", v, args[0])
			return err == nil && equal(m, int64(0)), err
		},
		"eq":      cmp("=="),
		"equalto": cmp("=="),
		"==":      cmp("=="),
		"escaped": func(any, []any) (bool, error) { return false, nil },
		"even": func(v any, _ []any) (bool, error) {
			n, ok := v.(int64)
			if !ok {
				return false, fmt.Errorf("even test needs an integer")
			}
			return n%2 == 0, nil
		},
		"odd": func(v any, _ []any) (bool, error) {
			n, ok := v.(int64)
			if !ok {
				return false, fmt.Errorf("odd test needs an integer")
			}
			return n%2 != 0, nil
		},
		"false":       func(v any, _ []any) (bool, error) { b, ok := v.(bool); return ok && !b, nil },
		"true":        func(v any, _ []any) (bool, error) { b, ok := v.(bool); return ok && b, nil },
		"float":       func(v any, _ []any) (bool, error) { _, ok := v.(float64); return ok, nil },
		"integer":     func(v any, _ []any) (bool, error) { return isInt(v), nil },
		"number":      func(v any, _ []any) (bool, error) { _, _, _, ok := numeric(v); return ok, nil },
		"ge":          cmp(">="),
		">=":          cmp(">="),
		"gt":          cmp(">"),
		">":           cmp(">"),
		"greaterthan": cmp(">"),
		"le":          cmp("<="),
		"<=":          cmp("<="),
		"lt":          cmp("<"),
		"<":           cmp("<"),
		"lessthan":    cmp("<"),
		"ne":          cmp("!="),
		"!=":          cmp("!="),
		"in": func(v any, args []any) (bool, error) {
			if len(args) != 1 {
				return false, fmt.Errorf("in test takes one argument")
			}
			return contains(args[0], v)
		},
		"iterable": func(v any, _ []any) (bool, error) {
			switch v.(type) {
			case string, *List, *Dict, Undefined:
				return true, nil
			}
			return false, nil
		},
		"sequence": func(v any, _ []any) (bool, error) {
			switch v.(type) {
			case string, *List, *Dict, Undefined:
				return true, nil
			}
			return false, nil
		},
		"mapping":   func(v any, _ []any) (bool, error) { _, ok := v.(*Dict); return ok, nil },
		"none":      func(v any, _ []any) (bool, error) { return v == nil, nil },
		"string":    func(v any, _ []any) (bool, error) { _, ok := v.(string); return ok, nil },
		"undefined": func(v any, _ []any) (bool, error) { _, u := v.(Undefined); return u, nil },
		"lower": func(v any, _ []any) (bool, error) {
			f, _ := strMethod(toStr(v), "islower")
			b, err := f(nil, nil)
			return b.(bool), err
		},
		"upper": func(v any, _ []any) (bool, error) {
			f, _ := strMethod(toStr(v), "isupper")
			b, err := f(nil, nil)
			return b.(bool), err
		},
		"sameas": func(v any, args []any) (bool, error) {
			if len(args) != 1 {
				return false, fmt.Errorf("sameas takes one argument")
			}
			o := args[0]
			switch a := v.(type) {
			case nil:
				return o == nil, nil
			case bool:
				b, ok := o.(bool)
				return ok && a == b, nil
			case int64, float64, string:
				return a == o, nil
			case *List:
				b, ok := o.(*List)
				return ok && a == b, nil
			case *Dict:
				b, ok := o.(*Dict)
				return ok && a == b, nil
			case *Namespace:
				b, ok := o.(*Namespace)
				return ok && a == b, nil
			}
			return false, nil
		},
		"filter": func(v any, _ []any) (bool, error) {
			s, ok := v.(string)
			_, exists := filters[s]
			return ok && exists, nil
		},
	}
	tests["test"] = func(v any, _ []any) (bool, error) {
		s, ok := v.(string)
		_, exists := tests[s]
		return ok && exists, nil
	}
}

func runTest(name string, v any, args []any) (bool, error) {
	return tests[name](v, args)
}
