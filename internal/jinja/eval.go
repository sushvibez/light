package jinja

import (
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"
)

// Options mirror the jinja2.Environment whitespace settings.
type Options struct {
	TrimBlocks          bool // drop the first newline after a block / comment tag
	LstripBlocks        bool // strip spaces/tabs before a block / comment tag at line start
	KeepTrailingNewline bool // keep a single trailing newline of the source
}

// Error is a syntax or render error with the template line it happened on.
// Errors raised by Func values are kept as Err (see errors.As / Unwrap).
type Error struct {
	Line int
	Msg  string
	Err  error
}

func (e *Error) Error() string {
	if e.Line > 0 {
		return fmt.Sprintf("jinja: line %d: %s", e.Line, e.Msg)
	}
	return "jinja: " + e.Msg
}

func (e *Error) Unwrap() error { return e.Err }

// UndefinedError is the error raised when an undefined value is used in a
// way other than printing, testing or iterating it.
type UndefinedError struct{ Msg string }

func (e *UndefinedError) Error() string { return e.Msg }

func undefinedErr(u Undefined) error { return &UndefinedError{Msg: u.message()} }

func wrapErr(line int, err error) error {
	if err == nil {
		return nil
	}
	var je *Error
	if errors.As(err, &je) {
		return err
	}
	return &Error{Line: line, Msg: err.Error(), Err: err}
}

// Template is a parsed template; it is safe for concurrent Render calls.
type Template struct {
	body []node
}

// Parse compiles src. Every construct the engine does not implement is a
// parse error, so a template that parses never renders something silently
// different from jinja2 because of a missing feature.
func Parse(src string, opt Options) (*Template, error) {
	toks, err := tokenize(src, opt)
	if err != nil {
		return nil, err
	}
	body, err := parse(toks)
	if err != nil {
		return nil, err
	}
	return &Template{body: body}, nil
}

// Render executes the template with vars as the global context. Values
// must be template values (see FromGo / FromJSON); Go strings, bools,
// int64, float64 and nil are used as-is.
func (t *Template) Render(vars map[string]any) (string, error) {
	globals := &scope{vars: map[string]any{}}
	r := &renderer{}
	for k, v := range builtinGlobals(r) {
		globals.vars[k] = v
	}
	for k, v := range vars {
		globals.vars[k] = v
	}
	top := &scope{vars: map[string]any{}, parent: globals}
	var out strings.Builder
	if _, err := r.exec(t.body, top, &out); err != nil {
		return "", err
	}
	return out.String(), nil
}

type scope struct {
	vars   map[string]any
	parent *scope
}

func newScope(parent *scope) *scope { return &scope{vars: map[string]any{}, parent: parent} }

func (s *scope) lookup(name string) (any, bool) {
	for c := s; c != nil; c = c.parent {
		if v, ok := c.vars[name]; ok {
			return v, true
		}
	}
	return nil, false
}

type ctrl int

const (
	ctrlNone ctrl = iota
	ctrlBreak
	ctrlContinue
)

type renderer struct {
	depth int
}

type macroValue struct {
	r       *renderer
	n       *macroNode
	closure *scope
}

type loopValue struct {
	items []any
	index int
}

func (r *renderer) exec(nodes []node, sc *scope, out *strings.Builder) (ctrl, error) {
	for _, n := range nodes {
		switch n := n.(type) {
		case *textNode:
			out.WriteString(n.s)
		case *printNode:
			v, err := r.eval(n.x, sc)
			if err != nil {
				return 0, wrapErr(n.line, err)
			}
			out.WriteString(toStr(v))
		case *ifNode:
			done := false
			for i, c := range n.conds {
				v, err := r.eval(c, sc)
				if err != nil {
					return 0, err
				}
				if truthy(v) {
					if c, err := r.exec(n.bodies[i], sc, out); err != nil || c != ctrlNone {
						return c, err
					}
					done = true
					break
				}
			}
			if !done && n.els != nil {
				if c, err := r.exec(n.els, sc, out); err != nil || c != ctrlNone {
					return c, err
				}
			}
		case *forNode:
			if err := r.execFor(n, sc, out); err != nil {
				return 0, err
			}
		case *setNode:
			v, err := r.eval(n.x, sc)
			if err != nil {
				return 0, wrapErr(n.line, err)
			}
			if err := assign(n.target, v, sc); err != nil {
				return 0, wrapErr(n.line, err)
			}
		case *setBlockNode:
			var buf strings.Builder
			if _, err := r.exec(n.body, sc, &buf); err != nil {
				return 0, err
			}
			v, err := r.applyFilters(n.filters, buf.String(), sc)
			if err != nil {
				return 0, wrapErr(n.line, err)
			}
			if err := assign(n.target, v, sc); err != nil {
				return 0, wrapErr(n.line, err)
			}
		case *macroNode:
			sc.vars[n.name] = &macroValue{r: r, n: n, closure: sc}
		case *filterBlockNode:
			var buf strings.Builder
			if c, err := r.exec(n.body, sc, &buf); err != nil || c != ctrlNone {
				return c, err
			}
			v, err := r.applyFilters(n.filters, buf.String(), sc)
			if err != nil {
				return 0, wrapErr(n.line, err)
			}
			out.WriteString(toStr(v))
		case *withNode:
			inner := newScope(sc)
			for i, t := range n.targets {
				v, err := r.eval(n.values[i], sc)
				if err != nil {
					return 0, err
				}
				if err := assign(t, v, inner); err != nil {
					return 0, err
				}
			}
			if c, err := r.exec(n.body, inner, out); err != nil || c != ctrlNone {
				return c, err
			}
		case *breakNode:
			return ctrlBreak, nil
		case *continueNode:
			return ctrlContinue, nil
		default:
			return 0, fmt.Errorf("jinja: internal error: unknown node %T", n)
		}
	}
	return ctrlNone, nil
}

func (r *renderer) applyFilters(fs []*filterExpr, v any, sc *scope) (any, error) {
	for _, f := range fs {
		var err error
		if v, err = r.callFilter(f, v, sc); err != nil {
			return nil, err
		}
	}
	return v, nil
}

func assign(t assignTarget, v any, sc *scope) error {
	if t.attr != "" {
		obj, ok := sc.lookup(t.names[0])
		ns, isNS := obj.(*Namespace)
		if !ok || !isNS {
			return fmt.Errorf("cannot assign attribute on non-namespace object")
		}
		ns.attrs.Set(t.attr, v)
		return nil
	}
	if !t.tuple {
		sc.vars[t.names[0]] = v
		return nil
	}
	items, err := iterate(v)
	if err != nil {
		return err
	}
	if len(items) != len(t.names) {
		return fmt.Errorf("cannot unpack %d values into %d names", len(items), len(t.names))
	}
	for i, n := range t.names {
		sc.vars[n] = items[i]
	}
	return nil
}

func (r *renderer) execFor(n *forNode, sc *scope, out *strings.Builder) error {
	iv, err := r.eval(n.iter, sc)
	if err != nil {
		return wrapErr(n.line, err)
	}
	items, err := iterate(iv)
	if err != nil {
		return wrapErr(n.line, err)
	}
	if n.cond != nil {
		var kept []any
		for _, it := range items {
			s := newScope(sc)
			if err := assign(n.target, it, s); err != nil {
				return wrapErr(n.line, err)
			}
			c, err := r.eval(n.cond, s)
			if err != nil {
				return wrapErr(n.line, err)
			}
			if truthy(c) {
				kept = append(kept, it)
			}
		}
		items = kept
	}
	if len(items) == 0 {
		if n.els != nil {
			_, err := r.exec(n.els, newScope(sc), out)
			return err
		}
		return nil
	}
	for i, it := range items {
		s := newScope(sc)
		if err := assign(n.target, it, s); err != nil {
			return wrapErr(n.line, err)
		}
		s.vars["loop"] = &loopValue{items: items, index: i}
		c, err := r.exec(n.body, s, out)
		if err != nil {
			return err
		}
		if c == ctrlBreak {
			break
		}
	}
	return nil
}

func lineOf(x expr) int {
	switch x := x.(type) {
	case *nameExpr:
		return x.line
	case *getattrExpr:
		return x.line
	case *getitemExpr:
		return x.line
	case *callExpr:
		return x.line
	case *filterExpr:
		return x.line
	case *testExpr:
		return x.line
	case *binaryExpr:
		return x.line
	case *compareExpr:
		return x.line
	}
	return 0
}

func (r *renderer) eval(x expr, sc *scope) (any, error) {
	v, err := r.eval1(x, sc)
	if err != nil {
		if l := lineOf(x); l > 0 {
			return nil, wrapErr(l, err)
		}
	}
	return v, err
}

func (r *renderer) eval1(x expr, sc *scope) (any, error) {
	switch x := x.(type) {
	case *constExpr:
		return x.v, nil
	case *nameExpr:
		if v, ok := sc.lookup(x.name); ok {
			return v, nil
		}
		return Undefined{Name: x.name}, nil
	case *getattrExpr:
		obj, err := r.eval(x.x, sc)
		if err != nil {
			return nil, err
		}
		return getattr(obj, x.name)
	case *getitemExpr:
		obj, err := r.eval(x.x, sc)
		if err != nil {
			return nil, err
		}
		k, err := r.eval(x.key, sc)
		if err != nil {
			return nil, err
		}
		return getitem(obj, k)
	case *sliceExpr:
		return r.evalSlice(x, sc)
	case *callExpr:
		fn, err := r.eval(x.fn, sc)
		if err != nil {
			return nil, err
		}
		args, kw, err := r.evalArgs(x.args, x.kwargs, sc)
		if err != nil {
			return nil, err
		}
		return call(fn, args, kw)
	case *filterExpr:
		v, err := r.eval(x.x, sc)
		if err != nil {
			return nil, err
		}
		return r.callFilter(x, v, sc)
	case *testExpr:
		v, err := r.eval(x.x, sc)
		if err != nil {
			return nil, err
		}
		args, _, err := r.evalArgs(x.args, nil, sc)
		if err != nil {
			return nil, err
		}
		ok, err := runTest(x.name, v, args)
		if err != nil {
			return nil, err
		}
		return ok != x.negate, nil
	case *unaryExpr:
		v, err := r.eval(x.x, sc)
		if err != nil {
			return nil, err
		}
		switch x.op {
		case "not":
			return !truthy(v), nil
		case "-":
			switch v := v.(type) {
			case int64:
				return -v, nil
			case float64:
				return -v, nil
			case bool:
				if v {
					return int64(-1), nil
				}
				return int64(0), nil
			case Undefined:
				return nil, undefinedErr(v)
			}
			return nil, fmt.Errorf("bad operand type for unary -: '%s'", typeName(v))
		default:
			switch v := v.(type) {
			case int64, float64:
				return v, nil
			case bool:
				if v {
					return int64(1), nil
				}
				return int64(0), nil
			case Undefined:
				return nil, undefinedErr(v)
			}
			return nil, fmt.Errorf("bad operand type for unary +: '%s'", typeName(v))
		}
	case *binaryExpr:
		l, err := r.eval(x.l, sc)
		if err != nil {
			return nil, err
		}
		switch x.op {
		case "and":
			if !truthy(l) {
				return l, nil
			}
			return r.eval(x.r, sc)
		case "or":
			if truthy(l) {
				return l, nil
			}
			return r.eval(x.r, sc)
		}
		rv, err := r.eval(x.r, sc)
		if err != nil {
			return nil, err
		}
		if x.op == "~" {
			return toStr(l) + toStr(rv), nil
		}
		return binop(x.op, l, rv)
	case *compareExpr:
		l, err := r.eval(x.first, sc)
		if err != nil {
			return nil, err
		}
		for i, op := range x.ops {
			rv, err := r.eval(x.rest[i], sc)
			if err != nil {
				return nil, err
			}
			ok, err := compareOp(op, l, rv)
			if err != nil {
				return nil, err
			}
			if !ok {
				return false, nil
			}
			l = rv
		}
		return true, nil
	case *condExpr:
		t, err := r.eval(x.test, sc)
		if err != nil {
			return nil, err
		}
		if truthy(t) {
			return r.eval(x.a, sc)
		}
		if x.b == nil {
			return Undefined{}, nil
		}
		return r.eval(x.b, sc)
	case *listExpr:
		l := &List{Items: make([]any, 0, len(x.items))}
		for _, it := range x.items {
			v, err := r.eval(it, sc)
			if err != nil {
				return nil, err
			}
			l.Items = append(l.Items, v)
		}
		return l, nil
	case *tupleExpr:
		l := &List{Items: make([]any, 0, len(x.items)), Tuple: true}
		for _, it := range x.items {
			v, err := r.eval(it, sc)
			if err != nil {
				return nil, err
			}
			l.Items = append(l.Items, v)
		}
		return l, nil
	case *dictExpr:
		d := NewDict()
		for i := range x.keys {
			k, err := r.eval(x.keys[i], sc)
			if err != nil {
				return nil, err
			}
			if !hashable(k) {
				return nil, fmt.Errorf("unhashable type: '%s'", typeName(k))
			}
			v, err := r.eval(x.vals[i], sc)
			if err != nil {
				return nil, err
			}
			d.Set(k, v)
		}
		return d, nil
	}
	return nil, fmt.Errorf("jinja: internal error: unknown expression %T", x)
}

func compareOp(op string, l, r any) (bool, error) {
	switch op {
	case "==":
		return equal(l, r), nil
	case "!=":
		return !equal(l, r), nil
	case "in":
		return contains(r, l)
	case "notin":
		ok, err := contains(r, l)
		return !ok, err
	}
	c, err := compare(l, r)
	if err != nil {
		return false, err
	}
	switch op {
	case "<":
		return c < 0, nil
	case ">":
		return c > 0, nil
	case "<=":
		return c <= 0, nil
	case ">=":
		return c >= 0, nil
	}
	return false, fmt.Errorf("unknown comparison %s", op)
}

func (r *renderer) evalArgs(args []expr, kwargs []kwarg, sc *scope) ([]any, *Dict, error) {
	out := make([]any, len(args))
	for i, a := range args {
		v, err := r.eval(a, sc)
		if err != nil {
			return nil, nil, err
		}
		out[i] = v
	}
	kw := NewDict()
	for _, k := range kwargs {
		v, err := r.eval(k.x, sc)
		if err != nil {
			return nil, nil, err
		}
		kw.Set(k.name, v)
	}
	return out, kw, nil
}

func call(fn any, args []any, kw *Dict) (any, error) {
	if kw == nil {
		kw = NewDict()
	}
	switch f := fn.(type) {
	case Func:
		return f(args, kw)
	case *macroValue:
		return f.call(args, kw)
	case Undefined:
		return nil, undefinedErr(f)
	}
	return nil, fmt.Errorf("'%s' object is not callable", typeName(fn))
}

func (m *macroValue) call(args []any, kw *Dict) (any, error) {
	n := m.n
	if len(args) > len(n.params) {
		return nil, fmt.Errorf("macro '%s' takes not more than %d argument(s)", n.name, len(n.params))
	}
	if m.r.depth > 100 {
		return nil, fmt.Errorf("macro recursion too deep")
	}
	sc := newScope(m.closure)
	firstDefault := len(n.params) - len(n.defaults)
	for i, p := range n.params {
		switch {
		case i < len(args):
			if _, dup := kw.Get(p); dup {
				return nil, fmt.Errorf("macro '%s' got multiple values for argument '%s'", n.name, p)
			}
			sc.vars[p] = args[i]
		default:
			if v, ok := kw.Get(p); ok {
				sc.vars[p] = v
			} else if i >= firstDefault {
				v, err := m.r.eval(n.defaults[i-firstDefault], sc)
				if err != nil {
					return nil, err
				}
				sc.vars[p] = v
			} else {
				sc.vars[p] = Undefined{Name: p}
			}
		}
	}
	for _, k := range kw.Keys() {
		ks, _ := k.(string)
		found := false
		for _, p := range n.params {
			found = found || p == ks
		}
		if !found {
			return nil, fmt.Errorf("macro '%s' takes no keyword argument '%s'", n.name, ks)
		}
	}
	m.r.depth++
	defer func() { m.r.depth-- }()
	var out strings.Builder
	if _, err := m.r.exec(n.body, sc, &out); err != nil {
		return nil, err
	}
	return out.String(), nil
}

func (r *renderer) evalSlice(x *sliceExpr, sc *scope) (any, error) {
	obj, err := r.eval(x.x, sc)
	if err != nil {
		return nil, err
	}
	var parts [3]any
	for i, e := range []expr{x.start, x.stop, x.step} {
		if e == nil {
			continue
		}
		if parts[i], err = r.eval(e, sc); err != nil {
			return nil, err
		}
	}
	return slice(obj, parts[0], parts[1], parts[2])
}

func sliceIndex(v any) (int, bool, error) {
	switch v := v.(type) {
	case nil, Undefined:
		return 0, false, nil
	case int64:
		return int(v), true, nil
	case bool:
		if v {
			return 1, true, nil
		}
		return 0, true, nil
	}
	return 0, false, fmt.Errorf("slice indices must be integers or None")
}

// pySliceRange returns the indices Python's seq[start:stop:step] selects.
func pySliceRange(n int, startV, stopV, stepV any) ([]int, error) {
	step, ok, err := sliceIndex(stepV)
	if err != nil {
		return nil, err
	}
	if !ok {
		step = 1
	}
	if step == 0 {
		return nil, fmt.Errorf("slice step cannot be zero")
	}
	clamp := func(v any, def int) (int, error) {
		i, ok, err := sliceIndex(v)
		if err != nil || !ok {
			return def, err
		}
		if i < 0 {
			i += n
			if i < 0 {
				if step < 0 {
					return -1, nil
				}
				return 0, nil
			}
		} else if i >= n {
			if step < 0 {
				return n - 1, nil
			}
			return n, nil
		}
		return i, nil
	}
	var start, stop int
	if step > 0 {
		if start, err = clamp(startV, 0); err != nil {
			return nil, err
		}
		if stop, err = clamp(stopV, n); err != nil {
			return nil, err
		}
	} else {
		if start, err = clamp(startV, n-1); err != nil {
			return nil, err
		}
		if stop, err = clamp(stopV, -1); err != nil {
			return nil, err
		}
	}
	var idx []int
	for i := start; (step > 0 && i < stop) || (step < 0 && i > stop); i += step {
		idx = append(idx, i)
	}
	return idx, nil
}

func slice(obj, start, stop, step any) (any, error) {
	switch o := obj.(type) {
	case *List:
		idx, err := pySliceRange(len(o.Items), start, stop, step)
		if err != nil {
			return nil, err
		}
		l := &List{Items: make([]any, len(idx)), Tuple: o.Tuple}
		for i, j := range idx {
			l.Items[i] = o.Items[j]
		}
		return l, nil
	case string:
		rs := []rune(o)
		idx, err := pySliceRange(len(rs), start, stop, step)
		if err != nil {
			return nil, err
		}
		out := make([]rune, len(idx))
		for i, j := range idx {
			out[i] = rs[j]
		}
		return string(out), nil
	case Undefined:
		return nil, undefinedErr(o)
	}
	return nil, fmt.Errorf("'%s' object is not subscriptable", typeName(obj))
}

// getattr implements jinja's `obj.name`: attribute (method) first, then item.
func getattr(obj any, name string) (any, error) {
	switch o := obj.(type) {
	case Undefined:
		return nil, undefinedErr(o)
	case *Dict:
		if m, ok := dictMethod(o, name); ok {
			return m, nil
		}
		if v, ok := o.Get(name); ok {
			return v, nil
		}
	case string:
		if m, ok := strMethod(o, name); ok {
			return m, nil
		}
	case *List:
		if m, ok := listMethod(o, name); ok {
			return m, nil
		}
	case *Namespace:
		if v, ok := o.attrs.Get(name); ok {
			return v, nil
		}
	case *loopValue:
		return o.attr(name)
	}
	return Undefined{Name: name, Obj: typeName(obj) + " object"}, nil
}

// getitem implements jinja's `obj[key]`: item first, then attribute.
func getitem(obj, key any) (any, error) {
	switch o := obj.(type) {
	case Undefined:
		return nil, undefinedErr(o)
	case *Dict:
		if v, ok := o.Get(key); ok {
			return v, nil
		}
	case *List:
		if i, ok, _ := sliceIndex(key); ok {
			if i < 0 {
				i += len(o.Items)
			}
			if i >= 0 && i < len(o.Items) {
				return o.Items[i], nil
			}
			return Undefined{Name: toStr(key), Obj: typeName(obj) + " object"}, nil
		}
	case string:
		if i, ok, _ := sliceIndex(key); ok {
			n := utf8.RuneCountInString(o)
			if i < 0 {
				i += n
			}
			if i >= 0 && i < n {
				return string([]rune(o)[i]), nil
			}
			return Undefined{Name: toStr(key), Obj: "str object"}, nil
		}
	}
	if s, ok := key.(string); ok {
		return getattr(obj, s)
	}
	return Undefined{Name: toStr(key), Obj: typeName(obj) + " object"}, nil
}

func (l *loopValue) attr(name string) (any, error) {
	n := len(l.items)
	switch name {
	case "index":
		return int64(l.index + 1), nil
	case "index0":
		return int64(l.index), nil
	case "revindex":
		return int64(n - l.index), nil
	case "revindex0":
		return int64(n - l.index - 1), nil
	case "first":
		return l.index == 0, nil
	case "last":
		return l.index == n-1, nil
	case "length":
		return int64(n), nil
	case "depth":
		return int64(1), nil
	case "depth0":
		return int64(0), nil
	case "previtem":
		if l.index == 0 {
			return Undefined{Name: "there is no previous item"}, nil
		}
		return l.items[l.index-1], nil
	case "nextitem":
		if l.index == n-1 {
			return Undefined{Name: "there is no next item"}, nil
		}
		return l.items[l.index+1], nil
	case "cycle":
		return Func(func(args []any, kw *Dict) (any, error) {
			if len(args) == 0 {
				return nil, fmt.Errorf("no items for cycling given")
			}
			return args[l.index%len(args)], nil
		}), nil
	}
	return Undefined{Name: name, Obj: "LoopContext object"}, nil
}

func builtinGlobals(r *renderer) map[string]any {
	return map[string]any{
		"range": Func(func(args []any, kw *Dict) (any, error) {
			var start, stop, step int64 = 0, 0, 1
			ints := make([]int64, len(args))
			for i, a := range args {
				n, ok := a.(int64)
				if !ok {
					return nil, fmt.Errorf("range() argument must be int, not %s", typeName(a))
				}
				ints[i] = n
			}
			switch len(ints) {
			case 1:
				stop = ints[0]
			case 2:
				start, stop = ints[0], ints[1]
			case 3:
				start, stop, step = ints[0], ints[1], ints[2]
			default:
				return nil, fmt.Errorf("range expected 1 to 3 arguments, got %d", len(args))
			}
			if step == 0 {
				return nil, fmt.Errorf("range() arg 3 must not be zero")
			}
			l := &List{}
			for i := start; (step > 0 && i < stop) || (step < 0 && i > stop); i += step {
				if len(l.Items) >= 100000 {
					return nil, fmt.Errorf("range too big")
				}
				l.Items = append(l.Items, i)
			}
			return l, nil
		}),
		"namespace": Func(func(args []any, kw *Dict) (any, error) {
			ns := &Namespace{attrs: NewDict()}
			for _, a := range args {
				d, ok := a.(*Dict)
				if !ok {
					return nil, fmt.Errorf("namespace() positional argument must be a dict")
				}
				for _, k := range d.Keys() {
					v, _ := d.Get(k)
					ns.attrs.Set(k, v)
				}
			}
			for _, k := range kw.Keys() {
				v, _ := kw.Get(k)
				ns.attrs.Set(k, v)
			}
			return ns, nil
		}),
		"dict": Func(func(args []any, kw *Dict) (any, error) {
			d := NewDict()
			for _, a := range args {
				src, ok := a.(*Dict)
				if !ok {
					return nil, fmt.Errorf("dict() positional argument must be a dict")
				}
				for _, k := range src.Keys() {
					v, _ := src.Get(k)
					d.Set(k, v)
				}
			}
			for _, k := range kw.Keys() {
				v, _ := kw.Get(k)
				d.Set(k, v)
			}
			return d, nil
		}),
	}
}
