package jinja

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Values are plain Go values with Python semantics:
//
//	nil        None
//	bool       bool
//	int64      int
//	float64    float
//	string     str
//	*List      list (Tuple set: tuple)
//	*Dict      dict (insertion ordered)
//	*Namespace jinja2 namespace()
//	Undefined  jinja2.Undefined
//	Func       a callable (globals, bound methods, macros)

// Undefined is a missing variable / attribute. Printing it gives "", it is
// falsy and iterates as empty; any other use is an error (jinja's default
// Undefined, which is what transformers uses).
type Undefined struct {
	Name string // variable name, or attribute name with Obj set
	Obj  string // type of the object the attribute was missing on
}

func (u Undefined) message() string {
	if u.Obj != "" {
		return fmt.Sprintf("'%s' has no attribute '%s'", u.Obj, u.Name)
	}
	if u.Name != "" {
		return fmt.Sprintf("'%s' is undefined", u.Name)
	}
	return "value is undefined"
}

// List is a Python list or tuple.
type List struct {
	Items []any
	Tuple bool
}

// NewList wraps items as a list value.
func NewList(items ...any) *List { return &List{Items: items} }

// Dict is an insertion-ordered dict. Keys are strings, int64, bool, float64
// or nil (the hashable scalars).
type Dict struct {
	keys []any
	m    map[any]any
}

// NewDict returns an empty dict.
func NewDict() *Dict { return &Dict{m: map[any]any{}} }

func dictKey(k any) any {
	switch v := k.(type) {
	case bool:
		if v {
			return int64(1)
		}
		return int64(0)
	case float64:
		if v == math.Trunc(v) && math.Abs(v) < 1<<62 {
			return int64(v)
		}
	}
	return k
}

// Set inserts or replaces k.
func (d *Dict) Set(k, v any) {
	hk := dictKey(k)
	if _, ok := d.m[hk]; !ok {
		d.keys = append(d.keys, k)
	}
	d.m[hk] = v
}

// Get looks k up.
func (d *Dict) Get(k any) (any, bool) {
	if !hashable(k) {
		return nil, false
	}
	v, ok := d.m[dictKey(k)]
	return v, ok
}

// Len is the number of keys.
func (d *Dict) Len() int { return len(d.keys) }

// Keys returns the keys in insertion order.
func (d *Dict) Keys() []any { return d.keys }

func hashable(k any) bool {
	switch k.(type) {
	case nil, bool, int64, float64, string:
		return true
	}
	return false
}

// Namespace is the mutable object returned by namespace().
type Namespace struct{ attrs *Dict }

// Func is a callable value. kw is never nil.
type Func func(args []any, kw *Dict) (any, error)

// ---- conversions ----------------------------------------------------------

// FromJSON decodes JSON into values, keeping object key order (which
// matters for tojson / item iteration).
func FromJSON(data []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	v, err := decodeJSON(dec)
	if err != nil {
		return nil, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, fmt.Errorf("jinja: trailing data after JSON value")
	}
	return v, nil
}

func decodeJSON(dec *json.Decoder) (any, error) {
	t, err := dec.Token()
	if err != nil {
		return nil, err
	}
	switch t := t.(type) {
	case json.Delim:
		switch t {
		case '[':
			l := &List{Items: []any{}}
			for dec.More() {
				v, err := decodeJSON(dec)
				if err != nil {
					return nil, err
				}
				l.Items = append(l.Items, v)
			}
			_, err := dec.Token()
			return l, err
		case '{':
			d := NewDict()
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return nil, err
				}
				v, err := decodeJSON(dec)
				if err != nil {
					return nil, err
				}
				d.Set(kt.(string), v)
			}
			_, err := dec.Token()
			return d, err
		}
	case json.Number:
		s := string(t)
		if !strings.ContainsAny(s, ".eE") {
			if n, err := strconv.ParseInt(s, 10, 64); err == nil {
				return n, nil
			}
		}
		return strconv.ParseFloat(s, 64)
	case string, bool, nil:
		return t, nil
	}
	return nil, fmt.Errorf("jinja: unexpected JSON token %v", t)
}

// FromGo converts ordinary Go data (maps, slices, numbers, strings,
// json.RawMessage) into values. map[string]any keys are sorted, since Go
// maps have no order; pass a *Dict or JSON to control key order.
func FromGo(v any) (any, error) {
	switch v := v.(type) {
	case nil, bool, int64, float64, string, *List, *Dict, *Namespace, Undefined, Func:
		return v, nil
	case int:
		return int64(v), nil
	case int32:
		return int64(v), nil
	case float32:
		return float64(v), nil
	case json.RawMessage:
		return FromJSON(v)
	case func(args []any, kw *Dict) (any, error):
		return Func(v), nil
	case []string:
		l := &List{Items: make([]any, len(v))}
		for i, s := range v {
			l.Items[i] = s
		}
		return l, nil
	case []any:
		l := &List{Items: make([]any, len(v))}
		for i, x := range v {
			c, err := FromGo(x)
			if err != nil {
				return nil, err
			}
			l.Items[i] = c
		}
		return l, nil
	case map[string]any:
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		d := NewDict()
		for _, k := range keys {
			c, err := FromGo(v[k])
			if err != nil {
				return nil, err
			}
			d.Set(k, c)
		}
		return d, nil
	case map[string]string:
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		d := NewDict()
		for _, k := range keys {
			d.Set(k, v[k])
		}
		return d, nil
	}
	return nil, fmt.Errorf("jinja: cannot convert %T to a template value", v)
}

// ---- type names, truthiness ------------------------------------------------

func typeName(v any) string {
	switch v := v.(type) {
	case nil:
		return "NoneType"
	case bool:
		return "bool"
	case int64:
		return "int"
	case float64:
		return "float"
	case string:
		return "str"
	case *List:
		if v.Tuple {
			return "tuple"
		}
		return "list"
	case *Dict:
		return "dict"
	case *Namespace:
		return "Namespace"
	case Undefined:
		return "Undefined"
	case Func, *macroValue:
		return "function"
	case *loopValue:
		return "LoopContext"
	}
	return fmt.Sprintf("%T", v)
}

func truthy(v any) bool {
	switch v := v.(type) {
	case nil:
		return false
	case bool:
		return v
	case int64:
		return v != 0
	case float64:
		return v != 0
	case string:
		return v != ""
	case *List:
		return len(v.Items) > 0
	case *Dict:
		return v.Len() > 0
	case Undefined:
		return false
	}
	return true
}

// ---- str() / repr() --------------------------------------------------------

func pyFloatRepr(f float64) string {
	switch {
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	case math.IsNaN(f):
		return "nan"
	}
	// Shortest round-trip digits, then Python's layout rules: scientific
	// when the decimal exponent is < -4 or >= 16.
	e := strconv.FormatFloat(f, 'e', -1, 64)
	mant, expS, _ := strings.Cut(e, "e")
	exp, _ := strconv.Atoi(expS)
	neg := strings.HasPrefix(mant, "-")
	mant = strings.TrimPrefix(mant, "-")
	digits := strings.Replace(mant, ".", "", 1)
	sign := ""
	if neg {
		sign = "-"
	}
	if exp < -4 || exp >= 16 {
		m := digits[:1]
		if len(digits) > 1 {
			m += "." + digits[1:]
		}
		es := fmt.Sprintf("%+03d", exp)
		return sign + m + "e" + es
	}
	if exp < 0 {
		return sign + "0." + strings.Repeat("0", -exp-1) + digits
	}
	if len(digits) <= exp+1 {
		return sign + digits + strings.Repeat("0", exp+1-len(digits)) + ".0"
	}
	return sign + digits[:exp+1] + "." + digits[exp+1:]
}

// toStr is Python's str(); Undefined renders as "".
func toStr(v any) string {
	switch v := v.(type) {
	case string:
		return v
	case Undefined:
		return ""
	case nil:
		return "None"
	case bool:
		if v {
			return "True"
		}
		return "False"
	case int64:
		return strconv.FormatInt(v, 10)
	case float64:
		return pyFloatRepr(v)
	}
	return repr(v)
}

func pyStrRepr(s string) string {
	q := byte('\'')
	if strings.IndexByte(s, '\'') >= 0 && strings.IndexByte(s, '"') < 0 {
		q = '"'
	}
	var b strings.Builder
	b.WriteByte(q)
	for _, r := range s {
		switch {
		case r == '\\':
			b.WriteString(`\\`)
		case r == rune(q):
			b.WriteByte('\\')
			b.WriteByte(q)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r == '\t':
			b.WriteString(`\t`)
		case r == ' ' || (r != utf8.RuneError && unicode.IsPrint(r)):
			b.WriteRune(r)
		case r < 0x100:
			fmt.Fprintf(&b, `\x%02x`, r)
		case r < 0x10000:
			fmt.Fprintf(&b, `\u%04x`, r)
		default:
			fmt.Fprintf(&b, `\U%08x`, r)
		}
	}
	b.WriteByte(q)
	return b.String()
}

// repr is Python's repr() for the value kinds templates meet.
func repr(v any) string {
	switch v := v.(type) {
	case string:
		return pyStrRepr(v)
	case nil, bool, int64, float64:
		return toStr(v)
	case Undefined:
		return "Undefined"
	case *List:
		parts := make([]string, len(v.Items))
		for i, it := range v.Items {
			parts[i] = repr(it)
		}
		if v.Tuple {
			if len(parts) == 1 {
				return "(" + parts[0] + ",)"
			}
			return "(" + strings.Join(parts, ", ") + ")"
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case *Dict:
		parts := make([]string, 0, v.Len())
		for _, k := range v.keys {
			val, _ := v.Get(k)
			parts = append(parts, repr(k)+": "+repr(val))
		}
		return "{" + strings.Join(parts, ", ") + "}"
	case *Namespace:
		return "<Namespace " + repr(v.attrs) + ">"
	case *loopValue:
		return fmt.Sprintf("<LoopContext %d/%d>", v.index+1, len(v.items))
	}
	return "<" + typeName(v) + ">"
}

// ---- JSON (Python json.dumps) ---------------------------------------------

type jsonOpts struct {
	ensureASCII bool
	indent      string // "" with hasIndent=false means compact one-line
	hasIndent   bool
	itemSep     string
	keySep      string
	sortKeys    bool
}

func toJSON(v any, o jsonOpts) (string, error) {
	var b strings.Builder
	if err := writeJSON(&b, v, o, 0); err != nil {
		return "", err
	}
	return b.String(), nil
}

func writeJSONString(b *strings.Builder, s string, ascii bool) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			switch {
			case r < 0x20:
				fmt.Fprintf(b, `\u%04x`, r)
			case ascii && r > 0x7e:
				if r >= 0x10000 {
					r -= 0x10000
					fmt.Fprintf(b, `\u%04x\u%04x`, 0xd800+(r>>10), 0xdc00+(r&0x3ff))
				} else {
					fmt.Fprintf(b, `\u%04x`, r)
				}
			default:
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
}

func writeJSON(b *strings.Builder, v any, o jsonOpts, depth int) error {
	nl := func(d int) {
		if o.hasIndent {
			b.WriteByte('\n')
			b.WriteString(strings.Repeat(o.indent, d))
		}
	}
	switch v := v.(type) {
	case nil:
		b.WriteString("null")
	case bool:
		if v {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case int64:
		b.WriteString(strconv.FormatInt(v, 10))
	case float64:
		switch {
		case math.IsInf(v, 1):
			b.WriteString("Infinity")
		case math.IsInf(v, -1):
			b.WriteString("-Infinity")
		case math.IsNaN(v):
			b.WriteString("NaN")
		default:
			b.WriteString(pyFloatRepr(v))
		}
	case string:
		writeJSONString(b, v, o.ensureASCII)
	case *List:
		if len(v.Items) == 0 {
			b.WriteString("[]")
			return nil
		}
		b.WriteByte('[')
		for i, it := range v.Items {
			if i > 0 {
				b.WriteString(o.itemSep)
			}
			nl(depth + 1)
			if err := writeJSON(b, it, o, depth+1); err != nil {
				return err
			}
		}
		nl(depth)
		b.WriteByte(']')
	case *Dict:
		if v.Len() == 0 {
			b.WriteString("{}")
			return nil
		}
		keys := append([]any(nil), v.keys...)
		if o.sortKeys {
			var err error
			sort.SliceStable(keys, func(i, j int) bool {
				c, e := compare(keys[i], keys[j])
				if e != nil {
					err = e
				}
				return c < 0
			})
			if err != nil {
				return err
			}
		}
		b.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				b.WriteString(o.itemSep)
			}
			nl(depth + 1)
			var ks string
			switch kk := k.(type) {
			case string:
				ks = kk
			case nil:
				ks = "null"
			case bool:
				ks = map[bool]string{true: "true", false: "false"}[kk]
			default:
				ks = toStr(kk)
			}
			writeJSONString(b, ks, o.ensureASCII)
			b.WriteString(o.keySep)
			val, _ := v.Get(k)
			if err := writeJSON(b, val, o, depth+1); err != nil {
				return err
			}
		}
		nl(depth)
		b.WriteByte('}')
	default:
		return fmt.Errorf("Object of type %s is not JSON serializable", typeName(v))
	}
	return nil
}

// ---- equality, ordering ----------------------------------------------------

func numeric(v any) (int64, float64, bool, bool) { // i, f, isFloat, ok
	switch v := v.(type) {
	case bool:
		if v {
			return 1, 1, false, true
		}
		return 0, 0, false, true
	case int64:
		return v, float64(v), false, true
	case float64:
		return 0, v, true, true
	}
	return 0, 0, false, false
}

func equal(a, b any) bool {
	ai, af, aF, aok := numeric(a)
	bi, bf, bF, bok := numeric(b)
	if aok && bok {
		if aF || bF {
			return af == bf
		}
		return ai == bi
	}
	switch a := a.(type) {
	case nil:
		return b == nil
	case string:
		bs, ok := b.(string)
		return ok && a == bs
	case Undefined:
		_, ok := b.(Undefined)
		return ok
	case *List:
		bl, ok := b.(*List)
		if !ok || bl.Tuple != a.Tuple || len(bl.Items) != len(a.Items) {
			return false
		}
		for i := range a.Items {
			if !equal(a.Items[i], bl.Items[i]) {
				return false
			}
		}
		return true
	case *Dict:
		bd, ok := b.(*Dict)
		if !ok || bd.Len() != a.Len() {
			return false
		}
		for _, k := range a.keys {
			av, _ := a.Get(k)
			bv, ok := bd.Get(k)
			if !ok || !equal(av, bv) {
				return false
			}
		}
		return true
	case *Namespace:
		return a == b
	}
	return false
}

// compare orders numbers, strings and lists like Python 3 (anything else
// is a TypeError).
func compare(a, b any) (int, error) {
	ai, af, aF, aok := numeric(a)
	bi, bf, bF, bok := numeric(b)
	if aok && bok {
		if aF || bF {
			switch {
			case af < bf:
				return -1, nil
			case af > bf:
				return 1, nil
			}
			return 0, nil
		}
		switch {
		case ai < bi:
			return -1, nil
		case ai > bi:
			return 1, nil
		}
		return 0, nil
	}
	if as, ok := a.(string); ok {
		if bs, ok := b.(string); ok {
			return strings.Compare(as, bs), nil
		}
	}
	if al, ok := a.(*List); ok {
		if bl, ok := b.(*List); ok {
			for i := 0; i < len(al.Items) && i < len(bl.Items); i++ {
				if equal(al.Items[i], bl.Items[i]) {
					continue
				}
				return compare(al.Items[i], bl.Items[i])
			}
			return compare(int64(len(al.Items)), int64(len(bl.Items)))
		}
	}
	if u, ok := a.(Undefined); ok {
		return 0, undefinedErr(u)
	}
	if u, ok := b.(Undefined); ok {
		return 0, undefinedErr(u)
	}
	return 0, fmt.Errorf("'<' not supported between instances of '%s' and '%s'", typeName(a), typeName(b))
}

// ---- arithmetic -----------------------------------------------------------

func floorDiv(a, b int64) int64 {
	q := a / b
	if (a%b != 0) && ((a < 0) != (b < 0)) {
		q--
	}
	return q
}

func pyMod(a, b int64) int64 {
	m := a % b
	if m != 0 && ((m < 0) != (b < 0)) {
		m += b
	}
	return m
}

func binop(op string, a, b any) (any, error) {
	if u, ok := a.(Undefined); ok {
		return nil, undefinedErr(u)
	}
	if u, ok := b.(Undefined); ok {
		return nil, undefinedErr(u)
	}
	ai, af, aF, aok := numeric(a)
	bi, bf, bF, bok := numeric(b)
	if aok && bok {
		isF := aF || bF
		switch op {
		case "+":
			if isF {
				return af + bf, nil
			}
			return ai + bi, nil
		case "-":
			if isF {
				return af - bf, nil
			}
			return ai - bi, nil
		case "*":
			if isF {
				return af * bf, nil
			}
			return ai * bi, nil
		case "/":
			if bf == 0 {
				return nil, fmt.Errorf("division by zero")
			}
			return af / bf, nil
		case "//":
			if bf == 0 {
				return nil, fmt.Errorf("integer division or modulo by zero")
			}
			if isF {
				return math.Floor(af / bf), nil
			}
			return floorDiv(ai, bi), nil
		case "%":
			if bf == 0 {
				return nil, fmt.Errorf("integer division or modulo by zero")
			}
			if isF {
				m := math.Mod(af, bf)
				if m != 0 && ((m < 0) != (bf < 0)) {
					m += bf
				}
				return m, nil
			}
			return pyMod(ai, bi), nil
		case "**":
			if !isF && bi >= 0 {
				r := int64(1)
				for i := int64(0); i < bi; i++ {
					r *= ai
				}
				return r, nil
			}
			return math.Pow(af, bf), nil
		}
	}
	switch op {
	case "+":
		if as, ok := a.(string); ok {
			if bs, ok := b.(string); ok {
				return as + bs, nil
			}
			return nil, fmt.Errorf("can only concatenate str (not \"%s\") to str", typeName(b))
		}
		if al, ok := a.(*List); ok {
			if bl, ok := b.(*List); ok && al.Tuple == bl.Tuple {
				items := append(append([]any{}, al.Items...), bl.Items...)
				return &List{Items: items, Tuple: al.Tuple}, nil
			}
		}
	case "*":
		if n, ok := b.(int64); ok {
			switch a := a.(type) {
			case string:
				return strings.Repeat(a, max(0, int(n))), nil
			case *List:
				var items []any
				for i := int64(0); i < n; i++ {
					items = append(items, a.Items...)
				}
				return &List{Items: items, Tuple: a.Tuple}, nil
			}
		}
		if n, ok := a.(int64); ok {
			if _, ok := b.(string); ok {
				return binop(op, b, n)
			}
		}
	case "%":
		if as, ok := a.(string); ok {
			return pyPercentFormat(as, b)
		}
	}
	return nil, fmt.Errorf("unsupported operand type(s) for %s: '%s' and '%s'", op, typeName(a), typeName(b))
}

// pyPercentFormat implements the common subset of `"fmt" % args`:
// %s %r %d %i %f %.Nf %%.
func pyPercentFormat(f string, arg any) (any, error) {
	var args []any
	if l, ok := arg.(*List); ok && l.Tuple {
		args = l.Items
	} else {
		args = []any{arg}
	}
	var b strings.Builder
	ai := 0
	for i := 0; i < len(f); i++ {
		if f[i] != '%' {
			b.WriteByte(f[i])
			continue
		}
		j := i + 1
		for j < len(f) && strings.IndexByte("0123456789.-+ ", f[j]) >= 0 {
			j++
		}
		if j >= len(f) {
			return nil, fmt.Errorf("incomplete format")
		}
		spec, verb := f[i+1:j], f[j]
		i = j
		if verb == '%' {
			b.WriteByte('%')
			continue
		}
		if ai >= len(args) {
			return nil, fmt.Errorf("not enough arguments for format string")
		}
		a := args[ai]
		ai++
		switch verb {
		case 's':
			b.WriteString(fmt.Sprintf("%"+spec+"s", toStr(a)))
		case 'r':
			b.WriteString(fmt.Sprintf("%"+spec+"s", repr(a)))
		case 'd', 'i':
			n, nf, isF, ok := numeric(a)
			if !ok {
				return nil, fmt.Errorf("%%d format: a real number is required, not %s", typeName(a))
			}
			if isF {
				n = int64(nf)
			}
			b.WriteString(fmt.Sprintf("%"+spec+"d", n))
		case 'f', 'e', 'g':
			_, nf, _, ok := numeric(a)
			if !ok {
				return nil, fmt.Errorf("must be real number, not %s", typeName(a))
			}
			if !strings.Contains(spec, ".") {
				spec += ".6"
			}
			b.WriteString(fmt.Sprintf("%"+spec+string(verb), nf))
		default:
			return nil, fmt.Errorf("unsupported format character '%c'", verb)
		}
	}
	if ai < len(args) {
		return nil, fmt.Errorf("not all arguments converted during string formatting")
	}
	return b.String(), nil
}

// iterate returns the items a for-loop / filter sees.
func iterate(v any) ([]any, error) {
	switch v := v.(type) {
	case *List:
		return v.Items, nil
	case *Dict:
		return append([]any(nil), v.keys...), nil
	case string:
		out := make([]any, 0, len(v))
		for _, r := range v {
			out = append(out, string(r))
		}
		return out, nil
	case Undefined:
		return nil, nil
	}
	return nil, fmt.Errorf("'%s' object is not iterable", typeName(v))
}

func length(v any) (int, error) {
	switch v := v.(type) {
	case string:
		return utf8.RuneCountInString(v), nil
	case *List:
		return len(v.Items), nil
	case *Dict:
		return v.Len(), nil
	case Undefined:
		return 0, nil
	}
	return 0, fmt.Errorf("object of type '%s' has no len()", typeName(v))
}

func contains(container, item any) (bool, error) {
	switch c := container.(type) {
	case string:
		s, ok := item.(string)
		if !ok {
			return false, fmt.Errorf("'in <string>' requires string as left operand, not %s", typeName(item))
		}
		return strings.Contains(c, s), nil
	case *List:
		for _, it := range c.Items {
			if equal(it, item) {
				return true, nil
			}
		}
		return false, nil
	case *Dict:
		_, ok := c.Get(item)
		return ok, nil
	case Undefined:
		return false, nil
	}
	return false, fmt.Errorf("argument of type '%s' is not iterable", typeName(container))
}
