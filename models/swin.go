package models

import (
	"fmt"
	"math"

	"light/device"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// Swin is the Swin Transformer backbone as BiRefNet uses it (idol-sdk's
// _SwinLarge, the original Swin's detection variant): a 4 x 4 patch
// embedding, then stages of window attention blocks (every second one on
// windows shifted by half a window, masked so tokens only attend within
// their own region of the un-rolled image; a learned relative position
// bias per head) joined by 2 x 2 patch merging. Feature maps whose size is
// not a multiple of the window are zero-padded per block. Forward returns
// each stage's LayerNormed output as (B, C, H, W).
type Swin struct {
	Patch      *birefConv // stride = kernel = patch size
	PatchNorm  idolNorm
	Stages     []*swinStage
	Norms      []idolNorm // norm0 .. norm{n-1}
	dev        device.Device
	shiftMasks map[[4]int]*tensor.Tensor // (Hp, Wp, window, shift) -> (nW, 1, N, N)
}

type swinStage struct {
	Blocks    []*swinBlock
	Window    int
	MergeNorm idolNorm   // patch merging; nil Reduction on the last stage
	Reduction *nn.Linear // 4C -> 2C, no bias
}

type swinBlock struct {
	Norm1, Norm2 idolNorm
	QKV, Proj    *nn.Linear
	FC1, FC2     *nn.Linear
	Heads, Shift int
	bias         *tensor.Tensor // (H, N, N) relative position bias
}

const swinEps = 1e-5

// swinRelBias gathers the relative position bias table (T, H) into (H, N,
// N) for an ws x ws window (relative_position_index).
func swinRelBias(table []float32, heads, ws int, dev device.Device) (*tensor.Tensor, error) {
	N := ws * ws
	out := make([]float32, heads*N*N)
	for i := range N {
		yi, xi := i/ws, i%ws
		for j := range N {
			yj, xj := j/ws, j%ws
			idx := (yi-yj+ws-1)*(2*ws-1) + (xi - xj + ws - 1)
			for h := range heads {
				out[(h*N+i)*N+j] = table[idx*heads+h]
			}
		}
	}
	return tensor.New(out, []int{heads, N, N}, dev)
}

// shiftMask is _SwinBasicLayer's attention mask for an (Hp, Wp) padded map:
// 0 within a region, -100 across regions, (nW, 1, N, N).
func (s *Swin) shiftMask(Hp, Wp, ws, shift int) (*tensor.Tensor, error) {
	key := [4]int{Hp, Wp, ws, shift}
	if m, ok := s.shiftMasks[key]; ok {
		return m, nil
	}
	region := func(i, n int) int {
		switch {
		case i < n-ws:
			return 0
		case i < n-shift:
			return 1
		}
		return 2
	}
	nh, nw := Hp/ws, Wp/ws
	N := ws * ws
	out := make([]float32, nh*nw*N*N)
	lab := make([]int, N)
	for wy := range nh {
		for wx := range nw {
			for t := range N {
				lab[t] = region(wy*ws+t/ws, Hp)*3 + region(wx*ws+t%ws, Wp)
			}
			base := (wy*nw + wx) * N * N
			for i := range N {
				for j := range N {
					if lab[i] != lab[j] {
						out[base+i*N+j] = -100
					}
				}
			}
		}
	}
	m, err := tensor.New(out, []int{nh * nw, 1, N, N}, s.dev)
	if err != nil {
		return nil, err
	}
	if s.shiftMasks == nil {
		s.shiftMasks = map[[4]int]*tensor.Tensor{}
	}
	s.shiftMasks[key] = m
	return m, nil
}

// roll is torch.roll along one axis.
func swinRoll(o *iop, x *tensor.Tensor, shift, dim int) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	n := x.Dim(dim)
	s := ((shift % n) + n) % n
	if s == 0 {
		return x
	}
	return o.cat(dim, o.narrow(x, dim, n-s, s), o.narrow(x, dim, 0, n-s))
}

// forward runs one block on tokens x (B, H·W, C).
func (b *swinBlock) forward(o *iop, s *Swin, x *tensor.Tensor, H, W, ws int) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	B, C := x.Dim(0), x.Dim(2)
	h := o.reshape(o.ln(x, b.Norm1.W, b.Norm1.B, swinEps), B, H, W, C)
	pr, pb := (ws-W%ws)%ws, (ws-H%ws)%ws
	if pb > 0 {
		h = o.do(func() (*tensor.Tensor, error) { return ops.Pad(h, 1, 0, pb) })
	}
	if pr > 0 {
		h = o.do(func() (*tensor.Tensor, error) { return ops.Pad(h, 2, 0, pr) })
	}
	Hp, Wp := H+pb, W+pr
	var mask *tensor.Tensor = b.bias
	if b.Shift > 0 {
		h = swinRoll(o, swinRoll(o, h, -b.Shift, 1), -b.Shift, 2)
		sm := o.do(func() (*tensor.Tensor, error) { return s.shiftMask(Hp, Wp, ws, b.Shift) })
		mask = o.add(sm, b.bias) // (nW, H, N, N)
		if B > 1 {
			ms := make([]*tensor.Tensor, B)
			for i := range ms {
				ms[i] = mask
			}
			mask = o.cat(0, ms...)
		}
	}
	nh, nw, N := Hp/ws, Wp/ws, ws*ws
	Bw := B * nh * nw
	win := o.reshape(o.perm(o.reshape(h, B, nh, ws, nw, ws, C), 0, 1, 3, 2, 4, 5), Bw, N, C)
	D := C / b.Heads
	qkv := o.perm(o.reshape(o.lin(b.QKV, win), Bw, N, 3, b.Heads, D), 2, 0, 3, 1, 4) // (3, Bw, H, N, D)
	q := o.reshape(o.narrow(qkv, 0, 0, 1), Bw, b.Heads, N, D)
	k := o.reshape(o.narrow(qkv, 0, 1, 1), Bw, b.Heads, N, D)
	v := o.reshape(o.narrow(qkv, 0, 2, 1), Bw, b.Heads, N, D)
	a := o.do(func() (*tensor.Tensor, error) { return nn.SDPAScaled(q, k, v, mask, 1/math.Sqrt(float64(D))) })
	a = o.lin(b.Proj, o.reshape(o.perm(a, 0, 2, 1, 3), Bw, N, C))
	h = o.reshape(o.perm(o.reshape(a, B, nh, nw, ws, ws, C), 0, 1, 3, 2, 4, 5), B, Hp, Wp, C)
	if b.Shift > 0 {
		h = swinRoll(o, swinRoll(o, h, b.Shift, 1), b.Shift, 2)
	}
	if pb > 0 || pr > 0 {
		h = o.narrow(o.narrow(h, 1, 0, H), 2, 0, W)
	}
	x = o.add(x, o.reshape(h, B, H*W, C))
	m := o.lin(b.FC2, o.un(ops.GeluErf, o.lin(b.FC1, o.ln(x, b.Norm2.W, b.Norm2.B, swinEps))))
	return o.add(x, m)
}

// merge is _PatchMerging: (B, H·W, C) -> (B, H/2·W/2, 2C), concatenating
// the 2 x 2 neighbours in the order (0, 0), (1, 0), (0, 1), (1, 1) (row,
// column).
func (st *swinStage) merge(o *iop, x *tensor.Tensor, H, W int) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	if H%2 != 0 || W%2 != 0 {
		o.err = fmt.Errorf("swin: patch merging needs an even feature map, got %d x %d", H, W)
		return nil
	}
	B, C := x.Dim(0), x.Dim(2)
	t := o.reshape(o.perm(o.reshape(x, B, H/2, 2, W/2, 2, C), 0, 1, 3, 4, 2, 5), B, H/2*W/2, 4*C)
	return o.lin(st.Reduction, o.ln(t, st.MergeNorm.W, st.MergeNorm.B, swinEps))
}

// Forward returns the stage outputs (B, C_i, H_i, W_i) for images x (B, 3,
// H, W); H and W must be multiples of the patch size times 2^(stages-1).
func (s *Swin) Forward(x *tensor.Tensor) ([]*tensor.Tensor, error) {
	o := &iop{}
	t := s.Patch.forward(o, x)
	if o.err != nil {
		return nil, o.err
	}
	B, C, H, W := t.Dim(0), t.Dim(1), t.Dim(2), t.Dim(3)
	t = o.perm(o.reshape(t, B, C, H*W), 0, 2, 1)
	t = o.ln(t, s.PatchNorm.W, s.PatchNorm.B, swinEps)
	outs := make([]*tensor.Tensor, len(s.Stages))
	for i, st := range s.Stages {
		for _, b := range st.Blocks {
			t = b.forward(o, s, t, H, W, st.Window)
		}
		if o.err != nil {
			return nil, o.err
		}
		C = t.Dim(2)
		n := o.ln(t, s.Norms[i].W, s.Norms[i].B, swinEps)
		outs[i] = o.reshape(o.perm(o.reshape(n, B, H, W, C), 0, 3, 1, 2), B, C, H, W)
		if st.Reduction != nil {
			t = st.merge(o, t, H, W)
			H, W = H/2, W/2
		}
	}
	return outs, o.err
}

// loadSwin reads a Swin backbone under prefix (e.g. "bb."): the stage,
// block, head and window counts all follow from the tensors.
func (l *idolLoader) swin(p string) *Swin {
	s := &Swin{dev: l.dev, Patch: l.conv(p+"patch_embed.proj", "", true), PatchNorm: l.norm(p + "patch_embed.norm")}
	for i := 0; ; i++ {
		lp := fmt.Sprintf("%slayers.%d.", p, i)
		if _, ok := l.hf[lp+"blocks.0.norm1.weight"]; !ok || l.err != nil {
			break
		}
		st := &swinStage{}
		for j := 0; ; j++ {
			bp := fmt.Sprintf("%sblocks.%d.", lp, j)
			tab, ok := l.hf[bp+"attn.relative_position_bias_table"]
			if !ok {
				break
			}
			heads := tab.Dim(1)
			ws := (int(math.Round(math.Sqrt(float64(tab.Dim(0))))) + 1) / 2
			if (2*ws-1)*(2*ws-1) != tab.Dim(0) {
				l.err = fmt.Errorf("swin: %s bias table %v is not square", bp, tab.Shape())
				return nil
			}
			st.Window = ws
			b := &swinBlock{Norm1: l.norm(bp + "norm1"), Norm2: l.norm(bp + "norm2"),
				QKV: l.linear(bp + "attn.qkv"), Proj: l.linear(bp + "attn.proj"),
				FC1: l.linear(bp + "mlp.fc1"), FC2: l.linear(bp + "mlp.fc2"), Heads: heads}
			if j%2 == 1 {
				b.Shift = ws / 2
			}
			if tv := l.host(bp + "attn.relative_position_bias_table"); l.err == nil {
				b.bias, l.err = swinRelBias(tv, heads, ws, l.dev)
			}
			st.Blocks = append(st.Blocks, b)
		}
		if _, ok := l.hf[lp+"downsample.reduction.weight"]; ok {
			st.MergeNorm, st.Reduction = l.norm(lp+"downsample.norm"), l.linear(lp+"downsample.reduction")
		}
		s.Stages = append(s.Stages, st)
		s.Norms = append(s.Norms, l.norm(fmt.Sprintf("%snorm%d", p, i)))
	}
	if l.err == nil && len(s.Stages) == 0 {
		l.err = fmt.Errorf("swin: no %slayers.* tensors", p)
	}
	return s
}
