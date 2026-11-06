package pth

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"strings"
)

// A minimal pickle virtual machine: enough of protocols 2–5 to decode what
// torch.save emits (state dicts, checkpoints, whole nn.Modules). Objects are
// kept symbolic — a GLOBAL becomes a class ref, REDUCE/NEWOBJ/BUILD become
// nodes holding their callable and arguments — and interpreted afterwards by
// the tensor reconstruction in load.go. Mirrors candle-core/src/pickle.rs.

type opcode byte

const (
	opProto          opcode = 0x80
	opGlobal         opcode = 'c'
	opStackGlobal    opcode = 0x93
	opBinInt1        opcode = 'K'
	opBinInt2        opcode = 'M'
	opBinInt         opcode = 'J'
	opLong1          opcode = 0x8a
	opLong4          opcode = 0x8b
	opInt            opcode = 'I'
	opLong           opcode = 'L'
	opBinFloat       opcode = 'G'
	opBinUnicode     opcode = 'X'
	opShortBinUni    opcode = 0x8c
	opBinUnicode8    opcode = 0x8d
	opBinString      opcode = 'T'
	opShortBinString opcode = 'U'
	opBinBytes       opcode = 'B'
	opShortBinBytes  opcode = 'C'
	opBinBytes8      opcode = 0x8e
	opByteArray8     opcode = 0x96
	opBinPersID      opcode = 'Q'
	opPersID         opcode = 'P'
	opTuple          opcode = 't'
	opTuple1         opcode = 0x85
	opTuple2         opcode = 0x86
	opTuple3         opcode = 0x87
	opEmptyTuple     opcode = ')'
	opNewTrue        opcode = 0x88
	opNewFalse       opcode = 0x89
	opNone           opcode = 'N'
	opEmptyList      opcode = ']'
	opList           opcode = 'l'
	opAppend         opcode = 'a'
	opAppends        opcode = 'e'
	opEmptyDict      opcode = '}'
	opDict           opcode = 'd'
	opSetItem        opcode = 's'
	opSetItems       opcode = 'u'
	opEmptySet       opcode = 0x8f
	opAddItems       opcode = 0x90
	opFrozenSet      opcode = 0x91
	opMark           opcode = '('
	opPop            opcode = '0'
	opPopMark        opcode = '1'
	opDup            opcode = '2'
	opGet            opcode = 'g'
	opBinGet         opcode = 'h'
	opLongBinGet     opcode = 'j'
	opPut            opcode = 'p'
	opBinPut         opcode = 'q'
	opLongBinPut     opcode = 'r'
	opMemoize        opcode = 0x94
	opFrame          opcode = 0x95
	opReduce         opcode = 'R'
	opBuild          opcode = 'b'
	opNewObj         opcode = 0x81
	opNewObjEx       opcode = 0x92
	opStop           opcode = '.'
)

// Object model.
type (
	class struct{ module, name string }
	mark  struct{}
	tuple struct{ items []any }
	list  struct{ items []any }
	dict  struct {
		keys, vals []any
	}
	reduce struct{ callable, args any }
	build  struct{ obj, args any }
	persID struct{ id any }
)

func (d *dict) get(key string) (any, bool) {
	for i, k := range d.keys {
		if s, ok := k.(string); ok && s == key {
			return d.vals[i], true
		}
	}
	return nil, false
}

func (c class) String() string { return c.module + "." + c.name }

type vm struct {
	r     *bufio.Reader
	stack []any
	memo  map[uint32]any
}

// unpickle decodes one pickle stream and returns the resulting object.
func unpickle(r io.Reader) (any, error) {
	m := &vm{r: bufio.NewReader(r), memo: make(map[uint32]any)}
	for {
		done, err := m.step()
		if err != nil {
			return nil, err
		}
		if done {
			break
		}
	}
	if len(m.stack) != 1 {
		return nil, fmt.Errorf("pickle: %d objects left on the stack", len(m.stack))
	}
	return m.stack[0], nil
}

func (m *vm) push(o any) { m.stack = append(m.stack, o) }

func (m *vm) pop() (any, error) {
	if len(m.stack) == 0 {
		return nil, fmt.Errorf("pickle: pop from empty stack")
	}
	o := m.stack[len(m.stack)-1]
	m.stack = m.stack[:len(m.stack)-1]
	return o, nil
}

func (m *vm) top() (any, error) {
	if len(m.stack) == 0 {
		return nil, fmt.Errorf("pickle: empty stack")
	}
	return m.stack[len(m.stack)-1], nil
}

func (m *vm) popMark() ([]any, error) {
	for i := len(m.stack) - 1; i >= 0; i-- {
		if _, ok := m.stack[i].(mark); ok {
			items := append([]any(nil), m.stack[i+1:]...)
			m.stack = m.stack[:i]
			return items, nil
		}
	}
	return nil, fmt.Errorf("pickle: no mark on stack")
}

func (m *vm) readN(n int) ([]byte, error) {
	b := make([]byte, n)
	_, err := io.ReadFull(m.r, b)
	return b, err
}

func (m *vm) readU8() (uint8, error) { return m.r.ReadByte() }

func (m *vm) readU16() (uint16, error) {
	b, err := m.readN(2)
	return binary.LittleEndian.Uint16(b), err
}

func (m *vm) readU32() (uint32, error) {
	b, err := m.readN(4)
	return binary.LittleEndian.Uint32(b), err
}

func (m *vm) readU64() (uint64, error) {
	b, err := m.readN(8)
	return binary.LittleEndian.Uint64(b), err
}

func (m *vm) readLine() (string, error) {
	s, err := m.r.ReadString('\n')
	return strings.TrimRight(s, "\r\n"), err
}

func (m *vm) memoPut(id uint32) error {
	o, err := m.top()
	if err != nil {
		return err
	}
	m.memo[id] = o
	return nil
}

func (m *vm) memoGet(id uint32) error {
	o, ok := m.memo[id]
	if !ok {
		return fmt.Errorf("pickle: memo %d missing", id)
	}
	m.push(o)
	return nil
}

// decodeLong reads a little-endian two's-complement integer of n bytes.
func decodeLong(b []byte) int64 {
	var v int64
	for i := len(b) - 1; i >= 0; i-- {
		v = v<<8 | int64(b[i])
	}
	if len(b) > 0 && len(b) < 8 && b[len(b)-1]&0x80 != 0 {
		v -= 1 << (8 * uint(len(b)))
	}
	return v
}

func (m *vm) step() (bool, error) {
	b, err := m.readU8()
	if err != nil {
		return false, fmt.Errorf("pickle: %w", err)
	}
	op := opcode(b)
	switch op {
	case opProto:
		_, err = m.readU8()
	case opFrame:
		_, err = m.readU64()
	case opStop:
		return true, nil
	case opGlobal:
		mod, e1 := m.readLine()
		name, e2 := m.readLine()
		if e1 != nil || e2 != nil {
			return false, fmt.Errorf("pickle: GLOBAL: %v %v", e1, e2)
		}
		m.push(class{mod, name})
	case opStackGlobal:
		name, e1 := m.pop()
		mod, e2 := m.pop()
		if e1 != nil || e2 != nil {
			return false, fmt.Errorf("pickle: STACK_GLOBAL underflow")
		}
		ms, ok1 := mod.(string)
		ns, ok2 := name.(string)
		if !ok1 || !ok2 {
			return false, fmt.Errorf("pickle: STACK_GLOBAL expects two strings")
		}
		m.push(class{ms, ns})
	case opBinInt1:
		var v uint8
		v, err = m.readU8()
		m.push(int64(v))
	case opBinInt2:
		var v uint16
		v, err = m.readU16()
		m.push(int64(v))
	case opBinInt:
		var v uint32
		v, err = m.readU32()
		m.push(int64(int32(v)))
	case opLong1:
		var n uint8
		if n, err = m.readU8(); err == nil {
			var bs []byte
			bs, err = m.readN(int(n))
			m.push(decodeLong(bs))
		}
	case opLong4:
		var n uint32
		if n, err = m.readU32(); err == nil {
			var bs []byte
			bs, err = m.readN(int(n))
			m.push(decodeLong(bs))
		}
	case opInt, opLong:
		var s string
		if s, err = m.readLine(); err == nil {
			s = strings.TrimSuffix(s, "L")
			switch s {
			case "00":
				m.push(false)
			case "01":
				m.push(true)
			default:
				var v int64
				_, err = fmt.Sscan(s, &v)
				m.push(v)
			}
		}
	case opBinFloat:
		var bs []byte
		if bs, err = m.readN(8); err == nil {
			m.push(math.Float64frombits(binary.BigEndian.Uint64(bs)))
		}
	case opBinUnicode, opBinString, opBinBytes:
		var n uint32
		if n, err = m.readU32(); err == nil {
			var bs []byte
			bs, err = m.readN(int(n))
			if op == opBinBytes {
				m.push(bs)
			} else {
				m.push(string(bs))
			}
		}
	case opShortBinUni, opShortBinString, opShortBinBytes:
		var n uint8
		if n, err = m.readU8(); err == nil {
			var bs []byte
			bs, err = m.readN(int(n))
			if op == opShortBinBytes {
				m.push(bs)
			} else {
				m.push(string(bs))
			}
		}
	case opBinUnicode8, opBinBytes8, opByteArray8:
		var n uint64
		if n, err = m.readU64(); err == nil {
			var bs []byte
			bs, err = m.readN(int(n))
			if op == opBinUnicode8 {
				m.push(string(bs))
			} else {
				m.push(bs)
			}
		}
	case opBinPersID:
		var id any
		if id, err = m.pop(); err == nil {
			m.push(persID{id})
		}
	case opPersID:
		var s string
		if s, err = m.readLine(); err == nil {
			m.push(persID{s})
		}
	case opTuple:
		var items []any
		if items, err = m.popMark(); err == nil {
			m.push(&tuple{items})
		}
	case opTuple1, opTuple2, opTuple3:
		n := int(op-opTuple1) + 1
		if len(m.stack) < n {
			return false, fmt.Errorf("pickle: TUPLE%d underflow", n)
		}
		items := append([]any(nil), m.stack[len(m.stack)-n:]...)
		m.stack = m.stack[:len(m.stack)-n]
		m.push(&tuple{items})
	case opEmptyTuple:
		m.push(&tuple{})
	case opNewTrue:
		m.push(true)
	case opNewFalse:
		m.push(false)
	case opNone:
		m.push(nil)
	case opEmptyList:
		m.push(&list{})
	case opList:
		var items []any
		if items, err = m.popMark(); err == nil {
			m.push(&list{items})
		}
	case opAppend:
		var v any
		if v, err = m.pop(); err == nil {
			err = m.appendItems([]any{v})
		}
	case opAppends:
		var items []any
		if items, err = m.popMark(); err == nil {
			err = m.appendItems(items)
		}
	case opEmptyDict:
		m.push(&dict{})
	case opDict:
		var items []any
		if items, err = m.popMark(); err == nil {
			d := &dict{}
			for i := 0; i+1 < len(items); i += 2 {
				d.keys = append(d.keys, items[i])
				d.vals = append(d.vals, items[i+1])
			}
			m.push(d)
		}
	case opSetItem:
		v, e1 := m.pop()
		k, e2 := m.pop()
		if e1 != nil || e2 != nil {
			return false, fmt.Errorf("pickle: SETITEM underflow")
		}
		err = m.setItems([]any{k, v})
	case opSetItems:
		var items []any
		if items, err = m.popMark(); err == nil {
			err = m.setItems(items)
		}
	case opEmptySet:
		m.push(&list{})
	case opAddItems:
		var items []any
		if items, err = m.popMark(); err == nil {
			err = m.appendItems(items)
		}
	case opFrozenSet:
		var items []any
		if items, err = m.popMark(); err == nil {
			m.push(&list{items})
		}
	case opMark:
		m.push(mark{})
	case opPop:
		_, err = m.pop()
	case opPopMark:
		_, err = m.popMark()
	case opDup:
		var o any
		if o, err = m.top(); err == nil {
			m.push(o)
		}
	case opGet:
		var s string
		if s, err = m.readLine(); err == nil {
			var id uint32
			if _, err = fmt.Sscan(s, &id); err == nil {
				err = m.memoGet(id)
			}
		}
	case opBinGet:
		var id uint8
		if id, err = m.readU8(); err == nil {
			err = m.memoGet(uint32(id))
		}
	case opLongBinGet:
		var id uint32
		if id, err = m.readU32(); err == nil {
			err = m.memoGet(id)
		}
	case opPut:
		var s string
		if s, err = m.readLine(); err == nil {
			var id uint32
			if _, err = fmt.Sscan(s, &id); err == nil {
				err = m.memoPut(id)
			}
		}
	case opBinPut:
		var id uint8
		if id, err = m.readU8(); err == nil {
			err = m.memoPut(uint32(id))
		}
	case opLongBinPut:
		var id uint32
		if id, err = m.readU32(); err == nil {
			err = m.memoPut(id)
		}
	case opMemoize:
		err = m.memoPut(uint32(len(m.memo)))
	case opReduce, opNewObj:
		args, e1 := m.pop()
		callable, e2 := m.pop()
		if e1 != nil || e2 != nil {
			return false, fmt.Errorf("pickle: REDUCE underflow")
		}
		if c, ok := callable.(class); ok && c.module == "collections" && (c.name == "OrderedDict" || c.name == "defaultdict") {
			m.push(&dict{})
		} else {
			m.push(&reduce{callable, args})
		}
	case opNewObjEx:
		_, e0 := m.pop() // kwargs
		args, e1 := m.pop()
		callable, e2 := m.pop()
		if e0 != nil || e1 != nil || e2 != nil {
			return false, fmt.Errorf("pickle: NEWOBJ_EX underflow")
		}
		m.push(&reduce{callable, args})
	case opBuild:
		args, e1 := m.pop()
		obj, e2 := m.pop()
		if e1 != nil || e2 != nil {
			return false, fmt.Errorf("pickle: BUILD underflow")
		}
		if d, ok := obj.(*dict); ok {
			if a, ok := args.(*dict); ok { // OrderedDict state: merge
				d.keys = append(d.keys, a.keys...)
				d.vals = append(d.vals, a.vals...)
				m.push(d)
				break
			}
		}
		m.push(&build{obj, args})
	default:
		return false, fmt.Errorf("pickle: unsupported opcode 0x%02x (%q)", b, b)
	}
	if err != nil {
		return false, fmt.Errorf("pickle: opcode 0x%02x: %w", b, err)
	}
	return false, nil
}

func (m *vm) appendItems(items []any) error {
	t, err := m.top()
	if err != nil {
		return err
	}
	l, ok := t.(*list)
	if !ok {
		return fmt.Errorf("pickle: APPEND onto %T", t)
	}
	l.items = append(l.items, items...)
	return nil
}

func (m *vm) setItems(items []any) error {
	t, err := m.top()
	if err != nil {
		return err
	}
	d, ok := t.(*dict)
	if !ok {
		return fmt.Errorf("pickle: SETITEMS onto %T", t)
	}
	for i := 0; i+1 < len(items); i += 2 {
		d.keys = append(d.keys, items[i])
		d.vals = append(d.vals, items[i+1])
	}
	return nil
}
