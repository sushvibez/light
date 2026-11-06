package jinja

// Statement nodes.
type node interface{}

type (
	textNode  struct{ s string }
	printNode struct {
		x    expr
		line int
	}
	ifNode struct {
		conds  []expr
		bodies [][]node
		els    []node
	}
	forNode struct {
		target    assignTarget
		iter      expr
		cond      expr // optional `for x in y if cond`
		body, els []node
		line      int
	}
	setNode struct {
		target assignTarget
		x      expr
		line   int
	}
	setBlockNode struct {
		target  assignTarget
		filters []*filterExpr // applied to the captured body; arg slot 0 is nil
		body    []node
		line    int
	}
	macroNode struct {
		name     string
		params   []string
		defaults []expr // aligned to the tail of params
		body     []node
	}
	filterBlockNode struct {
		filters []*filterExpr
		body    []node
		line    int
	}
	withNode struct {
		targets []assignTarget
		values  []expr
		body    []node
	}
	breakNode    struct{}
	continueNode struct{}
)

// assignTarget is a name, a tuple of names, or a namespace attribute.
type assignTarget struct {
	names []string // len 1 for a plain name
	tuple bool
	attr  string // set ns.attr = ...: names[0] is the namespace
}

// Expression nodes.
type expr interface{}

type (
	constExpr struct{ v any }
	nameExpr  struct {
		name string
		line int
	}
	getattrExpr struct {
		x    expr
		name string
		line int
	}
	getitemExpr struct {
		x, key expr
		line   int
	}
	sliceExpr struct {
		x                 expr
		start, stop, step expr // nil when omitted
		line              int
	}
	callExpr struct {
		fn     expr
		args   []expr
		kwargs []kwarg
		line   int
	}
	kwarg struct {
		name string
		x    expr
	}
	filterExpr struct {
		x      expr // nil inside {% filter %} / {% set %} blocks
		name   string
		args   []expr
		kwargs []kwarg
		line   int
	}
	testExpr struct {
		x      expr
		name   string
		args   []expr
		negate bool
		line   int
	}
	unaryExpr struct {
		op   string // "not", "-", "+"
		x    expr
		line int
	}
	binaryExpr struct {
		op   string // and or + - * / // % ** ~
		l, r expr
		line int
	}
	compareExpr struct {
		first expr
		ops   []string // == != < > <= >= in notin
		rest  []expr
		line  int
	}
	condExpr struct {
		test, a, b expr // b may be nil
	}
	listExpr  struct{ items []expr }
	tupleExpr struct{ items []expr }
	dictExpr  struct{ keys, vals []expr }
)
