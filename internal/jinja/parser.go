package jinja

import (
	"fmt"
	"slices"
)

// parser is a recursive-descent port of jinja2.parser.Parser restricted to
// the statements chat templates use.
type parser struct {
	toks []token
	i    int
}

func (p *parser) cur() token { return p.toks[p.i] }

func (p *parser) peek(n int) token {
	if p.i+n < len(p.toks) {
		return p.toks[p.i+n]
	}
	return p.toks[len(p.toks)-1]
}

func (p *parser) next() token {
	t := p.toks[p.i]
	if p.i < len(p.toks)-1 {
		p.i++
	}
	return t
}

func (p *parser) failf(format string, a ...any) error {
	return &Error{Line: p.cur().line, Msg: fmt.Sprintf(format, a...)}
}

func (p *parser) isOp(op string) bool { t := p.cur(); return t.kind == tOp && t.val == op }

func (p *parser) isName(name string) bool { t := p.cur(); return t.kind == tName && t.val == name }

func (p *parser) skipOp(op string) bool {
	if p.isOp(op) {
		p.next()
		return true
	}
	return false
}

func (p *parser) skipName(name string) bool {
	if p.isName(name) {
		p.next()
		return true
	}
	return false
}

func (p *parser) expectOp(op string) error {
	if !p.skipOp(op) {
		return p.failf("expected '%s', got %v", op, p.cur())
	}
	return nil
}

func (p *parser) expectKind(k tokKind) error {
	if p.cur().kind != k {
		return p.failf("expected %v, got %v", k, p.cur())
	}
	p.next()
	return nil
}

func (p *parser) expectName() (string, error) {
	t := p.cur()
	if t.kind != tName {
		return "", p.failf("expected name, got %v", t)
	}
	p.next()
	return t.val, nil
}

func parse(toks []token) ([]node, error) {
	p := &parser{toks: toks}
	body, _, err := p.subparse(nil)
	return body, err
}

// subparse reads nodes until EOF (endNames == nil) or a block whose first
// name is in endNames; that name is consumed and returned, the rest of the
// tag is left for the caller.
func (p *parser) subparse(endNames []string) ([]node, string, error) {
	var body []node
	for {
		t := p.cur()
		switch t.kind {
		case tData:
			p.next()
			body = append(body, &textNode{t.val})
		case tVarBegin:
			p.next()
			x, err := p.parseTuple(false, true, nil, false)
			if err != nil {
				return nil, "", err
			}
			if err := p.expectKind(tVarEnd); err != nil {
				return nil, "", err
			}
			body = append(body, &printNode{x: x, line: t.line})
		case tBlockBegin:
			p.next()
			if n := p.cur(); n.kind == tName && slices.Contains(endNames, n.val) {
				p.next()
				return body, n.val, nil
			}
			nodes, err := p.parseStatement()
			if err != nil {
				return nil, "", err
			}
			body = append(body, nodes...)
		case tEOF:
			if endNames != nil {
				return nil, "", p.failf("unexpected end of template, expected '%s'", endNames[len(endNames)-1])
			}
			return body, "", nil
		default:
			return nil, "", p.failf("unexpected %v", t)
		}
	}
}

func (p *parser) parseStatement() ([]node, error) {
	t := p.cur()
	if t.kind != tName {
		return nil, p.failf("tag name expected, got %v", t)
	}
	p.next()
	one := func(n node, err error) ([]node, error) {
		if err != nil {
			return nil, err
		}
		return []node{n}, nil
	}
	switch t.val {
	case "if":
		return one(p.parseIf())
	case "for":
		return one(p.parseFor(t.line))
	case "set":
		return one(p.parseSet(t.line))
	case "macro":
		return one(p.parseMacro())
	case "filter":
		return one(p.parseFilterBlock(t.line))
	case "with":
		return one(p.parseWith())
	case "break", "continue":
		if err := p.expectKind(tBlockEnd); err != nil {
			return nil, err
		}
		if t.val == "break" {
			return []node{&breakNode{}}, nil
		}
		return []node{&continueNode{}}, nil
	case "generation":
		// transformers' AssistantTracker extension: renders its body
		// unchanged (it only records assistant-token offsets).
		if err := p.expectKind(tBlockEnd); err != nil {
			return nil, err
		}
		body, _, err := p.subparse([]string{"endgeneration"})
		if err != nil {
			return nil, err
		}
		return body, p.expectKind(tBlockEnd)
	case "include", "extends", "block", "import", "from", "call", "autoescape", "do", "trans", "pluralize", "print":
		return nil, &Error{Line: t.line, Msg: fmt.Sprintf("unsupported tag '%s'", t.val)}
	}
	return nil, &Error{Line: t.line, Msg: fmt.Sprintf("unknown tag '%s'", t.val)}
}

func (p *parser) parseIf() (node, error) {
	n := &ifNode{}
	for {
		cond, err := p.parseTuple(false, false, nil, false)
		if err != nil {
			return nil, err
		}
		if err := p.expectKind(tBlockEnd); err != nil {
			return nil, err
		}
		body, end, err := p.subparse([]string{"elif", "else", "endif"})
		if err != nil {
			return nil, err
		}
		n.conds = append(n.conds, cond)
		n.bodies = append(n.bodies, body)
		switch end {
		case "elif":
			continue
		case "else":
			if err := p.expectKind(tBlockEnd); err != nil {
				return nil, err
			}
			if n.els, _, err = p.subparse([]string{"endif"}); err != nil {
				return nil, err
			}
		}
		return n, p.expectKind(tBlockEnd)
	}
}

func (p *parser) parseFor(line int) (node, error) {
	target, err := p.parseAssignTarget(true, false, []string{"in"})
	if err != nil {
		return nil, err
	}
	if !p.skipName("in") {
		return nil, p.failf("expected 'in', got %v", p.cur())
	}
	iter, err := p.parseTuple(false, false, []string{"recursive"}, false)
	if err != nil {
		return nil, err
	}
	n := &forNode{target: target, iter: iter, line: line}
	if p.isName("recursive") {
		return nil, p.failf("recursive for loops are not supported")
	}
	if p.skipName("if") {
		if n.cond, err = p.parseExpression(true); err != nil {
			return nil, err
		}
	}
	if err := p.expectKind(tBlockEnd); err != nil {
		return nil, err
	}
	body, end, err := p.subparse([]string{"endfor", "else"})
	if err != nil {
		return nil, err
	}
	n.body = body
	if end == "else" {
		if err := p.expectKind(tBlockEnd); err != nil {
			return nil, err
		}
		if n.els, _, err = p.subparse([]string{"endfor"}); err != nil {
			return nil, err
		}
	}
	return n, p.expectKind(tBlockEnd)
}

func (p *parser) parseSet(line int) (node, error) {
	target, err := p.parseAssignTarget(true, true, nil)
	if err != nil {
		return nil, err
	}
	if p.skipOp("=") {
		x, err := p.parseTuple(false, true, nil, false)
		if err != nil {
			return nil, err
		}
		return &setNode{target: target, x: x, line: line}, p.expectKind(tBlockEnd)
	}
	n := &setBlockNode{target: target, line: line}
	for p.isOp("|") {
		f, err := p.parseFilter(nil)
		if err != nil {
			return nil, err
		}
		n.filters = append(n.filters, f)
	}
	if err := p.expectKind(tBlockEnd); err != nil {
		return nil, err
	}
	if n.body, _, err = p.subparse([]string{"endset"}); err != nil {
		return nil, err
	}
	return n, p.expectKind(tBlockEnd)
}

func (p *parser) parseMacro() (node, error) {
	name, err := p.expectName()
	if err != nil {
		return nil, err
	}
	n := &macroNode{name: name}
	if err := p.expectOp("("); err != nil {
		return nil, err
	}
	for !p.isOp(")") {
		if len(n.params) > 0 {
			if err := p.expectOp(","); err != nil {
				return nil, err
			}
		}
		arg, err := p.expectName()
		if err != nil {
			return nil, err
		}
		n.params = append(n.params, arg)
		if p.skipOp("=") {
			d, err := p.parseExpression(true)
			if err != nil {
				return nil, err
			}
			n.defaults = append(n.defaults, d)
		} else if len(n.defaults) > 0 {
			return nil, p.failf("non-default argument follows default argument")
		}
	}
	p.next()
	if err := p.expectKind(tBlockEnd); err != nil {
		return nil, err
	}
	if n.body, _, err = p.subparse([]string{"endmacro"}); err != nil {
		return nil, err
	}
	return n, p.expectKind(tBlockEnd)
}

func (p *parser) parseFilterBlock(line int) (node, error) {
	n := &filterBlockNode{line: line}
	f, err := p.parseFilterBody(nil)
	if err != nil {
		return nil, err
	}
	n.filters = append(n.filters, f)
	for p.isOp("|") {
		if f, err = p.parseFilter(nil); err != nil {
			return nil, err
		}
		n.filters = append(n.filters, f)
	}
	if err := p.expectKind(tBlockEnd); err != nil {
		return nil, err
	}
	if n.body, _, err = p.subparse([]string{"endfilter"}); err != nil {
		return nil, err
	}
	return n, p.expectKind(tBlockEnd)
}

func (p *parser) parseWith() (node, error) {
	n := &withNode{}
	for p.cur().kind != tBlockEnd {
		if len(n.targets) > 0 {
			if err := p.expectOp(","); err != nil {
				return nil, err
			}
		}
		t, err := p.parseAssignTarget(true, false, nil)
		if err != nil {
			return nil, err
		}
		if err := p.expectOp("="); err != nil {
			return nil, err
		}
		x, err := p.parseExpression(true)
		if err != nil {
			return nil, err
		}
		n.targets = append(n.targets, t)
		n.values = append(n.values, x)
	}
	p.next()
	var err error
	if n.body, _, err = p.subparse([]string{"endwith"}); err != nil {
		return nil, err
	}
	return n, p.expectKind(tBlockEnd)
}

func (p *parser) parseAssignTarget(withTuple, withNamespace bool, extraEnd []string) (assignTarget, error) {
	if withNamespace && p.cur().kind == tName && p.peek(1).kind == tOp && p.peek(1).val == "." {
		ns := p.next().val
		p.next()
		attr, err := p.expectName()
		if err != nil {
			return assignTarget{}, err
		}
		return assignTarget{names: []string{ns}, attr: attr}, nil
	}
	var x expr
	var err error
	if withTuple {
		x, err = p.parseTuple(true, false, extraEnd, false)
	} else {
		x, err = p.parsePrimary()
	}
	if err != nil {
		return assignTarget{}, err
	}
	switch x := x.(type) {
	case *nameExpr:
		return assignTarget{names: []string{x.name}}, nil
	case *tupleExpr:
		t := assignTarget{tuple: true}
		for _, it := range x.items {
			n, ok := it.(*nameExpr)
			if !ok {
				return assignTarget{}, p.failf("can't assign to this expression (only names and tuples of names)")
			}
			t.names = append(t.names, n.name)
		}
		return t, nil
	}
	return assignTarget{}, p.failf("can't assign to this expression")
}

func (p *parser) isTupleEnd(extraEnd []string) bool {
	t := p.cur()
	switch {
	case t.kind == tVarEnd, t.kind == tBlockEnd, t.kind == tEOF:
		return true
	case t.kind == tOp && t.val == ")":
		return true
	case t.kind == tName && slices.Contains(extraEnd, t.val):
		return true
	}
	return false
}

func (p *parser) parseTuple(simplified, withCondexpr bool, extraEnd []string, explicitParens bool) (expr, error) {
	var args []expr
	isTuple := false
	for {
		if len(args) > 0 {
			if err := p.expectOp(","); err != nil {
				return nil, err
			}
		}
		if p.isTupleEnd(extraEnd) {
			break
		}
		var x expr
		var err error
		if simplified {
			x, err = p.parsePrimary()
		} else {
			x, err = p.parseExpression(withCondexpr)
		}
		if err != nil {
			return nil, err
		}
		args = append(args, x)
		if p.isOp(",") {
			isTuple = true
		} else {
			break
		}
	}
	if !isTuple {
		if len(args) > 0 {
			return args[0], nil
		}
		if !explicitParens {
			return nil, p.failf("expected an expression, got %v", p.cur())
		}
	}
	return &tupleExpr{items: args}, nil
}

func (p *parser) parseExpression(withCondexpr bool) (expr, error) {
	if !withCondexpr {
		return p.parseOr()
	}
	x, err := p.parseOr()
	if err != nil {
		return nil, err
	}
	for p.skipName("if") {
		test, err := p.parseOr()
		if err != nil {
			return nil, err
		}
		c := &condExpr{test: test, a: x}
		if p.skipName("else") {
			if c.b, err = p.parseExpression(true); err != nil {
				return nil, err
			}
		}
		x = c
	}
	return x, nil
}

func (p *parser) parseOr() (expr, error) {
	l, err := p.parseAnd()
	if err != nil {
		return nil, err
	}
	for p.isName("or") {
		line := p.next().line
		r, err := p.parseAnd()
		if err != nil {
			return nil, err
		}
		l = &binaryExpr{op: "or", l: l, r: r, line: line}
	}
	return l, nil
}

func (p *parser) parseAnd() (expr, error) {
	l, err := p.parseNot()
	if err != nil {
		return nil, err
	}
	for p.isName("and") {
		line := p.next().line
		r, err := p.parseNot()
		if err != nil {
			return nil, err
		}
		l = &binaryExpr{op: "and", l: l, r: r, line: line}
	}
	return l, nil
}

func (p *parser) parseNot() (expr, error) {
	if p.isName("not") {
		line := p.next().line
		x, err := p.parseNot()
		if err != nil {
			return nil, err
		}
		return &unaryExpr{op: "not", x: x, line: line}, nil
	}
	return p.parseCompare()
}

func (p *parser) parseCompare() (expr, error) {
	line := p.cur().line
	x, err := p.parseMath1()
	if err != nil {
		return nil, err
	}
	c := &compareExpr{first: x, line: line}
	for {
		t := p.cur()
		var op string
		switch {
		case t.kind == tOp && (t.val == "==" || t.val == "!=" || t.val == "<" || t.val == ">" || t.val == "<=" || t.val == ">="):
			op = t.val
			p.next()
		case t.kind == tName && t.val == "in":
			op = "in"
			p.next()
		case t.kind == tName && t.val == "not" && p.peek(1).kind == tName && p.peek(1).val == "in":
			op = "notin"
			p.next()
			p.next()
		}
		if op == "" {
			break
		}
		r, err := p.parseMath1()
		if err != nil {
			return nil, err
		}
		c.ops = append(c.ops, op)
		c.rest = append(c.rest, r)
	}
	if len(c.ops) == 0 {
		return x, nil
	}
	return c, nil
}

func (p *parser) parseBinary(ops []string, sub func() (expr, error)) (expr, error) {
	l, err := sub()
	if err != nil {
		return nil, err
	}
	for {
		t := p.cur()
		if t.kind != tOp || !slices.Contains(ops, t.val) {
			return l, nil
		}
		p.next()
		r, err := sub()
		if err != nil {
			return nil, err
		}
		l = &binaryExpr{op: t.val, l: l, r: r, line: t.line}
	}
}

func (p *parser) parseMath1() (expr, error) {
	return p.parseBinary([]string{"+", "-"}, p.parseConcat)
}

func (p *parser) parseConcat() (expr, error) {
	return p.parseBinary([]string{"~"}, p.parseMath2)
}

func (p *parser) parseMath2() (expr, error) {
	return p.parseBinary([]string{"*", "/", "//", "%"}, p.parsePow)
}

func (p *parser) parsePow() (expr, error) {
	return p.parseBinary([]string{"**"}, func() (expr, error) { return p.parseUnary(true) })
}

func (p *parser) parseUnary(withFilter bool) (expr, error) {
	t := p.cur()
	var x expr
	var err error
	if t.kind == tOp && (t.val == "-" || t.val == "+") {
		p.next()
		inner, err := p.parseUnary(false)
		if err != nil {
			return nil, err
		}
		x = &unaryExpr{op: t.val, x: inner, line: t.line}
	} else if x, err = p.parsePrimary(); err != nil {
		return nil, err
	}
	if x, err = p.parsePostfix(x); err != nil {
		return nil, err
	}
	if withFilter {
		return p.parseFilterExpr(x)
	}
	return x, nil
}

func (p *parser) parsePrimary() (expr, error) {
	t := p.cur()
	switch t.kind {
	case tName:
		p.next()
		switch t.val {
		case "true", "True":
			return &constExpr{true}, nil
		case "false", "False":
			return &constExpr{false}, nil
		case "none", "None":
			return &constExpr{nil}, nil
		}
		return &nameExpr{name: t.val, line: t.line}, nil
	case tString:
		p.next()
		s := t.val
		for p.cur().kind == tString {
			s += p.next().val
		}
		return &constExpr{s}, nil
	case tInt, tFloat:
		p.next()
		return &constExpr{t.num}, nil
	case tOp:
		switch t.val {
		case "(":
			p.next()
			x, err := p.parseTuple(false, true, nil, true)
			if err != nil {
				return nil, err
			}
			return x, p.expectOp(")")
		case "[":
			return p.parseList()
		case "{":
			return p.parseDict()
		}
	}
	return nil, p.failf("unexpected %v", t)
}

func (p *parser) parseList() (expr, error) {
	p.next()
	l := &listExpr{}
	for !p.isOp("]") {
		if len(l.items) > 0 {
			if err := p.expectOp(","); err != nil {
				return nil, err
			}
			if p.isOp("]") {
				break
			}
		}
		x, err := p.parseExpression(true)
		if err != nil {
			return nil, err
		}
		l.items = append(l.items, x)
	}
	return l, p.expectOp("]")
}

func (p *parser) parseDict() (expr, error) {
	p.next()
	d := &dictExpr{}
	for !p.isOp("}") {
		if len(d.keys) > 0 {
			if err := p.expectOp(","); err != nil {
				return nil, err
			}
			if p.isOp("}") {
				break
			}
		}
		k, err := p.parseExpression(true)
		if err != nil {
			return nil, err
		}
		if err := p.expectOp(":"); err != nil {
			return nil, err
		}
		v, err := p.parseExpression(true)
		if err != nil {
			return nil, err
		}
		d.keys = append(d.keys, k)
		d.vals = append(d.vals, v)
	}
	return d, p.expectOp("}")
}

func (p *parser) parsePostfix(x expr) (expr, error) {
	for {
		var err error
		switch {
		case p.isOp(".") || p.isOp("["):
			x, err = p.parseSubscript(x)
		case p.isOp("("):
			x, err = p.parseCall(x)
		default:
			return x, nil
		}
		if err != nil {
			return nil, err
		}
	}
}

func (p *parser) parseFilterExpr(x expr) (expr, error) {
	for {
		var err error
		switch {
		case p.isOp("|"):
			x, err = p.parseFilter(x)
		case p.isName("is"):
			x, err = p.parseTest(x)
		case p.isOp("("):
			x, err = p.parseCall(x)
		default:
			return x, nil
		}
		if err != nil {
			return nil, err
		}
	}
}

func (p *parser) parseSubscript(x expr) (expr, error) {
	t := p.next()
	if t.val == "." {
		a := p.cur()
		switch a.kind {
		case tName:
			p.next()
			return &getattrExpr{x: x, name: a.val, line: a.line}, nil
		case tInt:
			p.next()
			return &getitemExpr{x: x, key: &constExpr{a.num}, line: a.line}, nil
		}
		return nil, p.failf("expected name or number after '.', got %v", a)
	}
	// '[' subscript: an index or a slice.
	var parts [3]expr
	if !p.isOp(":") {
		k, err := p.parseExpression(true)
		if err != nil {
			return nil, err
		}
		if !p.isOp(":") {
			if p.isOp(",") {
				return nil, p.failf("tuple subscripts are not supported")
			}
			return &getitemExpr{x: x, key: k, line: t.line}, p.expectOp("]")
		}
		parts[0] = k
	}
	p.next() // ':'
	if !p.isOp("]") && !p.isOp(":") {
		e, err := p.parseExpression(true)
		if err != nil {
			return nil, err
		}
		parts[1] = e
	}
	if p.skipOp(":") && !p.isOp("]") {
		e, err := p.parseExpression(true)
		if err != nil {
			return nil, err
		}
		parts[2] = e
	}
	return &sliceExpr{x: x, start: parts[0], stop: parts[1], step: parts[2], line: t.line}, p.expectOp("]")
}

func (p *parser) parseCallArgs() ([]expr, []kwarg, error) {
	if err := p.expectOp("("); err != nil {
		return nil, nil, err
	}
	var args []expr
	var kwargs []kwarg
	for !p.isOp(")") {
		if len(args)+len(kwargs) > 0 {
			if err := p.expectOp(","); err != nil {
				return nil, nil, err
			}
			if p.isOp(")") {
				break
			}
		}
		if p.isOp("*") || p.isOp("**") {
			return nil, nil, p.failf("*args / **kwargs in calls are not supported")
		}
		if p.cur().kind == tName && p.peek(1).kind == tOp && p.peek(1).val == "=" {
			name := p.next().val
			p.next()
			v, err := p.parseExpression(true)
			if err != nil {
				return nil, nil, err
			}
			kwargs = append(kwargs, kwarg{name, v})
			continue
		}
		if len(kwargs) > 0 {
			return nil, nil, p.failf("positional argument follows keyword argument")
		}
		v, err := p.parseExpression(true)
		if err != nil {
			return nil, nil, err
		}
		args = append(args, v)
	}
	p.next()
	return args, kwargs, nil
}

func (p *parser) parseCall(fn expr) (expr, error) {
	line := p.cur().line
	args, kwargs, err := p.parseCallArgs()
	if err != nil {
		return nil, err
	}
	return &callExpr{fn: fn, args: args, kwargs: kwargs, line: line}, nil
}

func (p *parser) dottedName() (string, error) {
	name, err := p.expectName()
	if err != nil {
		return "", err
	}
	for p.isOp(".") {
		p.next()
		n, err := p.expectName()
		if err != nil {
			return "", err
		}
		name += "." + n
	}
	return name, nil
}

// parseFilter parses "| name(args)" applied to x.
func (p *parser) parseFilter(x expr) (*filterExpr, error) {
	if err := p.expectOp("|"); err != nil {
		return nil, err
	}
	return p.parseFilterBody(x)
}

func (p *parser) parseFilterBody(x expr) (*filterExpr, error) {
	line := p.cur().line
	name, err := p.dottedName()
	if err != nil {
		return nil, err
	}
	if _, ok := filters[name]; !ok {
		return nil, &Error{Line: line, Msg: fmt.Sprintf("unsupported filter '%s'", name)}
	}
	f := &filterExpr{x: x, name: name, line: line}
	if p.isOp("(") {
		if f.args, f.kwargs, err = p.parseCallArgs(); err != nil {
			return nil, err
		}
	}
	return f, nil
}

func (p *parser) parseTest(x expr) (expr, error) {
	line := p.next().line // 'is'
	negate := p.skipName("not")
	name, err := p.dottedName()
	if err != nil {
		return nil, err
	}
	if _, ok := tests[name]; !ok {
		return nil, &Error{Line: line, Msg: fmt.Sprintf("unsupported test '%s'", name)}
	}
	t := &testExpr{x: x, name: name, negate: negate, line: line}
	c := p.cur()
	switch {
	case p.isOp("("):
		args, kwargs, err := p.parseCallArgs()
		if err != nil {
			return nil, err
		}
		if len(kwargs) > 0 {
			return nil, p.failf("keyword arguments to tests are not supported")
		}
		t.args = args
	case c.kind == tName && c.val != "else" && c.val != "or" && c.val != "and",
		c.kind == tString, c.kind == tInt, c.kind == tFloat,
		c.kind == tOp && (c.val == "[" || c.val == "{"):
		if c.kind == tName && c.val == "is" {
			return nil, p.failf("you cannot chain multiple tests with 'is'")
		}
		arg, err := p.parsePrimary()
		if err != nil {
			return nil, err
		}
		if arg, err = p.parsePostfix(arg); err != nil {
			return nil, err
		}
		t.args = []expr{arg}
	}
	return t, nil
}
