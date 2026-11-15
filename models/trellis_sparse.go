package models

import (
	"fmt"
	"math"
	"sort"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// TRELLIS's sparse tensors (trellis/modules/sparse) for one object (batch
// 1): the features live on the device as (N, C) rows, the voxel coordinates
// on the host in a TrellisGrid, which also builds the index maps the sparse
// layers gather with:
//
//	SubMConv3d (spconv)  out[p] = b + Σ_k W_k·x[p + k − 1] over the 3³
//	                     offsets present (cross-correlation, absent sites
//	                     zero): per offset a gather of the voxels that have
//	                     that neighbour, a GEMM, and a gather back into the
//	                     output rows (no scatter, so it runs on every device)
//	SparseDownsample(2)  parents = unique(coords // 2) in (x, y, z) order;
//	                     torch.scatter_reduce(zeros, 'mean') includes the
//	                     zero it starts from, so a parent is Σ children /
//	                     (n + 1), as in TRELLIS
//	SparseUpsample(2)    each voxel takes its parent's features
//	windowed attention   (swin) voxels grouped by (coords + shift) // window;
//	                     windows of equal size attend as one batch
//
// The grid's order is the caller's (TRELLIS keeps any order; only the
// downsampled grid is sorted, as unique() returns it).

// TrellisGrid is the voxel coordinates of one sparse tensor (batch 1) with
// lazily built neighbour, pooling and window maps on dev.
type TrellisGrid struct {
	Coords [][3]int32
	dev    device.Device
	index  map[uint32]int32

	nbr      *[27]trellisPairs // 3³ submanifold neighbour pairs
	down, up *TrellisGrid      // the grid downsampled by 2 / the one this came from
	parent   []int32           // per voxel, its index in down
	parentT  *tensor.Tensor    // parent as I64 on dev
	pool     []*tensor.Tensor  // per child slot (M,) I64 into the padded rows
	poolInv  *tensor.Tensor    // (M, 1) 1 / (children + 1)
	pe       map[int]*tensor.Tensor
	windows  map[[2]int]*trellisWindows
}

// trellisPairs is one kernel offset's neighbour pairs: src (P,) indices of
// the input voxels, dst (N,) each output's position in src or P (absent).
// identity: every output's neighbour is itself (the centre offset).
type trellisPairs struct {
	src, dst *tensor.Tensor
	all      bool // P == N in output order: no gather back needed
	identity bool
	n        int // P
}

func trellisKey(x, y, z int32) uint32 { return uint32(x)<<20 | uint32(y)<<10 | uint32(z) }

// NewTrellisGrid wraps the occupied voxels (b, x, y, z) of one object
// (b = 0, coordinates in [0, 1023]) for the sparse layers on dev.
func NewTrellisGrid(coords [][4]int32, dev device.Device) (*TrellisGrid, error) {
	c := make([][3]int32, len(coords))
	for i, v := range coords {
		if v[0] != 0 {
			return nil, fmt.Errorf("trellis grid: batch index %d (one object at a time)", v[0])
		}
		c[i] = [3]int32{v[1], v[2], v[3]}
	}
	return newTrellisGrid(c, dev)
}

func newTrellisGrid(c [][3]int32, dev device.Device) (*TrellisGrid, error) {
	if len(c) == 0 {
		return nil, fmt.Errorf("trellis grid: no voxels")
	}
	g := &TrellisGrid{Coords: c, dev: dev, index: make(map[uint32]int32, len(c))}
	for i, v := range c {
		for _, x := range v {
			if x < 0 || x > 1023 {
				return nil, fmt.Errorf("trellis grid: coordinate %v outside [0, 1023]", v)
			}
		}
		k := trellisKey(v[0], v[1], v[2])
		if _, dup := g.index[k]; dup {
			return nil, fmt.Errorf("trellis grid: voxel %v twice", v)
		}
		g.index[k] = int32(i)
	}
	return g, nil
}

// N is the number of voxels.
func (g *TrellisGrid) N() int { return len(g.Coords) }

func (g *TrellisGrid) i64(v []int64) (*tensor.Tensor, error) {
	return tensor.New(v, []int{len(v)}, g.dev)
}

// neighbours builds the 3³ submanifold pairs (offset k = (kx·3 + ky)·3 +
// kz, neighbour at p + (kx − 1, ky − 1, kz − 1)).
func (g *TrellisGrid) neighbours() (*[27]trellisPairs, error) {
	if g.nbr != nil {
		return g.nbr, nil
	}
	var nb [27]trellisPairs
	N := g.N()
	for k := range 27 {
		d := [3]int32{int32(k/9) - 1, int32(k/3%3) - 1, int32(k%3) - 1}
		if d == [3]int32{} {
			nb[k] = trellisPairs{identity: true, n: N}
			continue
		}
		var src []int64
		dst := make([]int64, N)
		for i, v := range g.Coords {
			x, y, z := v[0]+d[0], v[1]+d[1], v[2]+d[2]
			j, ok := int32(-1), false
			if x >= 0 && y >= 0 && z >= 0 && x <= 1023 && y <= 1023 && z <= 1023 {
				j, ok = g.index[trellisKey(x, y, z)]
			}
			if !ok {
				dst[i] = -1
				continue
			}
			dst[i] = int64(len(src))
			src = append(src, int64(j))
		}
		P := len(src)
		p := trellisPairs{n: P, all: P == N}
		if P == 0 {
			nb[k] = p
			continue
		}
		for i := range dst {
			if dst[i] < 0 {
				dst[i] = int64(P)
			}
		}
		var err error
		if p.src, err = g.i64(src); err != nil {
			return nil, err
		}
		if !p.all {
			if p.dst, err = g.i64(dst); err != nil {
				return nil, err
			}
		}
		nb[k] = p
	}
	g.nbr = &nb
	return g.nbr, nil
}

// Down returns the grid downsampled by 2 (SparseDownsample's coords:
// unique(coords // 2), sorted).
func (g *TrellisGrid) Down() (*TrellisGrid, error) {
	if g.down != nil {
		return g.down, nil
	}
	keys := make([]uint32, 0, g.N())
	seen := map[uint32]bool{}
	for _, v := range g.Coords {
		k := trellisKey(v[0]/2, v[1]/2, v[2]/2)
		if !seen[k] {
			seen[k] = true
			keys = append(keys, k)
		}
	}
	sort.Slice(keys, func(a, b int) bool { return keys[a] < keys[b] })
	c := make([][3]int32, len(keys))
	for i, k := range keys {
		c[i] = [3]int32{int32(k >> 20), int32(k >> 10 & 1023), int32(k & 1023)}
	}
	d, err := newTrellisGrid(c, g.dev)
	if err != nil {
		return nil, err
	}
	M := d.N()
	g.parent = make([]int32, g.N())
	children := make([][]int64, M)
	for i, v := range g.Coords {
		p := d.index[trellisKey(v[0]/2, v[1]/2, v[2]/2)]
		g.parent[i] = p
		children[p] = append(children[p], int64(i))
	}
	par := make([]int64, g.N())
	for i, p := range g.parent {
		par[i] = int64(p)
	}
	if g.parentT, err = g.i64(par); err != nil {
		return nil, err
	}
	inv := make([]float32, M)
	for s := 0; ; s++ {
		idx := make([]int64, M)
		found := false
		for p, ch := range children {
			if s < len(ch) {
				idx[p], found = ch[s], true
			} else {
				idx[p] = int64(g.N()) // the zero row
			}
		}
		if !found {
			break
		}
		t, err := g.i64(idx)
		if err != nil {
			return nil, err
		}
		g.pool = append(g.pool, t)
	}
	for p, ch := range children {
		inv[p] = float32(1 / float64(len(ch)+1))
	}
	if g.poolInv, err = tensor.New(inv, []int{M, 1}, g.dev); err != nil {
		return nil, err
	}
	d.up = g
	g.down = d
	return d, nil
}

// trellisPad appends a zero row to x (N, C).
func trellisPad(o *iop, x *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	z := o.do(func() (*tensor.Tensor, error) { return tensor.Zeros(x.DType(), []int{1, x.Dim(1)}, x.Device()) })
	return o.cat(0, x, z)
}

func (o *iop) gather(x, idx *tensor.Tensor) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) { return ops.IndexSelect(x, idx) })
}

// downsample is SparseDownsample(2) on x (N, C): (M, C) rows of g.Down().
func (g *TrellisGrid) downsample(o *iop, x *tensor.Tensor) (*tensor.Tensor, *TrellisGrid) {
	if o.err != nil {
		return nil, nil
	}
	d, err := g.Down()
	if err != nil {
		o.err = err
		return nil, nil
	}
	xp := trellisPad(o, x)
	var sum *tensor.Tensor
	for _, idx := range g.pool {
		s := o.gather(xp, idx)
		if sum == nil {
			sum = s
		} else {
			sum = o.add(sum, s)
		}
	}
	return o.mul(sum, g.poolInv), d
}

// upsample is SparseUpsample(2) on the coarse rows x (M, C) of g: the
// (N, C) rows of the grid g was downsampled from.
func (g *TrellisGrid) upsample(o *iop, x *tensor.Tensor) (*tensor.Tensor, *TrellisGrid) {
	if o.err != nil {
		return nil, nil
	}
	if g.up == nil {
		o.err = fmt.Errorf("trellis: SparseUpsample without a matching SparseDownsample")
		return nil, nil
	}
	return o.gather(x, g.up.parentT), g.up
}

// posEmbed is AbsolutePositionEmbedder(C) of the voxel coordinates, (N, C).
func (g *TrellisGrid) posEmbed(o *iop, C int) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	if t, ok := g.pe[C]; ok {
		return t
	}
	t := o.do(func() (*tensor.Tensor, error) {
		return tensor.New(trellisAPE(g.Coords, C), []int{g.N(), C}, g.dev)
	})
	if o.err == nil {
		if g.pe == nil {
			g.pe = map[int]*tensor.Tensor{}
		}
		g.pe[C] = t
	}
	return t
}

// trellisAPE is AbsolutePositionEmbedder(C, 3): per axis [sin | cos] of
// x·10000^(−i/F), F = C / 3 / 2, zero-padded to C. Row-major (N, C).
func trellisAPE(coords [][3]int32, C int) []float32 {
	fd := C / 3 / 2
	freqs := make([]float32, fd)
	for i := range freqs {
		freqs[i] = float32(1 / math.Pow(10000, float64(float32(i)/float32(fd))))
	}
	out := make([]float32, len(coords)*C)
	for t, xyz := range coords {
		for k, x := range xyz {
			base := t*C + k*2*fd
			for i, f := range freqs {
				a := float64(float32(x) * f)
				out[base+i] = float32(math.Sin(a))
				out[base+fd+i] = float32(math.Cos(a))
			}
		}
	}
	return out
}

// --- submanifold convolution ---------------------------------------------------

// trellisSparseConv is SparseConv3d(in, out, 3) (spconv's SubMConv3d): one
// (out, in) linear per kernel offset, the bias on the centre one.
type trellisSparseConv struct {
	W [27]*nn.Linear
}

func (cv *trellisSparseConv) forward(o *iop, g *TrellisGrid, x *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	nb, err := g.neighbours()
	if err != nil {
		o.err = err
		return nil
	}
	out := o.lin(cv.W[13], x)
	cout := cv.W[13].W.Dim(0)
	var zero *tensor.Tensor
	for k := range 27 {
		p := &nb[k]
		if p.identity || p.n == 0 {
			continue
		}
		y := o.lin(cv.W[k], o.gather(x, p.src)) // (P, out)
		if !p.all {
			if zero == nil {
				zero = o.do(func() (*tensor.Tensor, error) { return tensor.Zeros(dtype.F32, []int{1, cout}, x.Device()) })
			}
			y = o.gather(o.cat(0, y, zero), p.dst)
		}
		out = o.add(out, y)
	}
	return out
}

// sparseConv loads an spconv SubMConv3d (<name>.conv.weight (out, 3, 3, 3,
// in), KRSC as spconv 2 stores it, and .bias) as a trellisSparseConv.
func (l *idolLoader) sparseConv(name string) *trellisSparseConv {
	if l.err != nil {
		return nil
	}
	w, ok := l.hf[name+".conv.weight"]
	if !ok || w.Rank() != 5 || w.Dim(1) != 3 || w.Dim(2) != 3 || w.Dim(3) != 3 {
		l.err = fmt.Errorf("missing or non-3³ sparse conv %s.conv.weight", name)
		return nil
	}
	co, ci := w.Dim(0), w.Dim(4)
	src := l.host(name + ".conv.weight")
	if l.err != nil {
		return nil
	}
	cv := &trellisSparseConv{}
	for k := range 27 {
		wk := make([]float32, co*ci)
		for oc := range co {
			copy(wk[oc*ci:(oc+1)*ci], src[(oc*27+k)*ci:(oc*27+k+1)*ci])
		}
		key := fmt.Sprintf("%s.k%d", name, k)
		if l.hf[key+".weight"], l.err = tensor.New(wk, []int{co, ci}, device.CPU()); l.err != nil {
			return nil
		}
		if k == 13 {
			if b, ok := l.hf[name+".conv.bias"]; ok {
				l.hf[key+".bias"] = b
			}
		}
		cv.W[k] = l.linear(key)
	}
	delete(l.hf, name+".conv.weight")
	return cv
}

// --- windowed attention -----------------------------------------------------------

// trellisWindows is calc_window_partition for one (window, shift): the
// voxels grouped into windows, windows of equal size batched.
type trellisWindows struct {
	groups []trellisWindowGroup
	inv    *tensor.Tensor // (N,) I64: each voxel's row in the concatenated groups
}

type trellisWindowGroup struct {
	idx  *tensor.Tensor // (n·L,) I64 voxel indices, window by window
	n, L int
}

func (g *TrellisGrid) windowPartition(size, shift int) (*trellisWindows, error) {
	key := [2]int{size, shift}
	if w, ok := g.windows[key]; ok {
		return w, nil
	}
	win := map[uint32][]int64{}
	var order []uint32
	for i, v := range g.Coords {
		k := trellisKey((v[0]+int32(shift))/int32(size), (v[1]+int32(shift))/int32(size), (v[2]+int32(shift))/int32(size))
		if _, ok := win[k]; !ok {
			order = append(order, k)
		}
		win[k] = append(win[k], int64(i))
	}
	sort.Slice(order, func(a, b int) bool { return order[a] < order[b] })
	byLen := map[int][]uint32{}
	var lens []int
	for _, k := range order {
		L := len(win[k])
		if _, ok := byLen[L]; !ok {
			lens = append(lens, L)
		}
		byLen[L] = append(byLen[L], k)
	}
	sort.Ints(lens)
	w := &trellisWindows{}
	inv := make([]int64, g.N())
	row := 0
	for _, L := range lens {
		ks := byLen[L]
		idx := make([]int64, 0, len(ks)*L)
		for _, k := range ks {
			for _, i := range win[k] {
				inv[i] = int64(row)
				row++
				idx = append(idx, i)
			}
		}
		t, err := g.i64(idx)
		if err != nil {
			return nil, err
		}
		w.groups = append(w.groups, trellisWindowGroup{idx: t, n: len(ks), L: L})
	}
	var err error
	if w.inv, err = g.i64(inv); err != nil {
		return nil, err
	}
	if g.windows == nil {
		g.windows = map[[2]int]*trellisWindows{}
	}
	g.windows[key] = w
	return w, nil
}

// windowedAttend is sparse_windowed_scaled_dot_product_self_attention for
// qkv (N, 3·H·D) rows ((3, H, D) per row): (N, H·D).
func (g *TrellisGrid) windowedAttend(o *iop, qkv *tensor.Tensor, H, size, shift int) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	w, err := g.windowPartition(size, shift)
	if err != nil {
		o.err = err
		return nil
	}
	C := qkv.Dim(1) / 3
	D := C / H
	outs := make([]*tensor.Tensor, 0, len(w.groups))
	for _, gr := range w.groups {
		s := o.reshape(o.gather(qkv, gr.idx), gr.n, gr.L, 3, H, D)
		q := o.reshape(o.narrow(s, 2, 0, 1), gr.n, gr.L, H, D)
		k := o.reshape(o.narrow(s, 2, 1, 1), gr.n, gr.L, H, D)
		v := o.reshape(o.narrow(s, 2, 2, 1), gr.n, gr.L, H, D)
		outs = append(outs, o.reshape(idolAttend(o, q, k, v), gr.n*gr.L, C))
	}
	all := outs[0]
	if len(outs) > 1 {
		all = o.cat(0, outs...)
	}
	return o.gather(all, w.inv)
}
