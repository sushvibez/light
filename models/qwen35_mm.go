package models

import (
	"fmt"
	"math"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// Qwen35RopeIndex is transformers' get_rope_index for one sequence with
// images: text tokens advance all three (t, h, w) streams together; each
// run of image placeholder tokens (one image, merged grid grid/merge) takes
// t = start + frame, h = start + row, w = start + col, and the text after
// it resumes at start + max(h, w) / merge. Returns the three streams and
// the delta to add to later positions (decode: position = cache length +
// delta).
func Qwen35RopeIndex(ids []int64, imageToken int64, grids [][3]int, merge int) ([3][]int64, int64, error) {
	var pos [3][]int64
	cur, g := int64(0), 0
	for i := 0; i < len(ids); {
		if ids[i] != imageToken {
			for s := range 3 {
				pos[s] = append(pos[s], cur)
			}
			cur++
			i++
			continue
		}
		if g >= len(grids) {
			return pos, 0, fmt.Errorf("Qwen35RopeIndex: more image runs than grids (%d)", len(grids))
		}
		t, h, w := grids[g][0], grids[g][1]/merge, grids[g][2]/merge
		n := t * h * w
		for j := range n {
			if i+j >= len(ids) || ids[i+j] != imageToken {
				return pos, 0, fmt.Errorf("Qwen35RopeIndex: image %d has %d placeholder tokens, grid needs %d", g, j, n)
			}
			pos[0] = append(pos[0], cur+int64(j/(h*w)))
			pos[1] = append(pos[1], cur+int64((j/w)%h))
			pos[2] = append(pos[2], cur+int64(j%w))
		}
		i += n
		cur += int64(max(h, w))
		g++
	}
	maxPos := int64(0)
	for s := range 3 {
		for _, p := range pos[s] {
			maxPos = max(maxPos, p)
		}
	}
	return pos, maxPos + 1 - int64(len(ids)), nil
}

// mropeTables builds the (T, rotaryDim) cos / sin tables for three-stream
// positions: frequency j takes its position from stream h when j % 3 == 1
// and j < 3·section[1], w when j % 3 == 2 and j < 3·section[2], else t
// (interleaved MRoPE), then the halves repeat for rotate_half.
func (m *Qwen35) mropeTables(pos [3][]int64, dev device.Device) (*tensor.Tensor, *tensor.Tensor, error) {
	rot := m.Cfg.rotaryDim()
	nf := rot / 2
	sec := m.Cfg.MRoPESection
	if len(sec) != 3 {
		return nil, nil, fmt.Errorf("qwen3.5: no mrope_section in the config")
	}
	T := len(pos[0])
	cs, sn := make([]float32, T*rot), make([]float32, T*rot)
	for j := range nf {
		inv := float32(1 / math.Pow(m.Cfg.RopeBase, float64(2*j)/float64(rot)))
		s := 0
		switch {
		case j%3 == 1 && j < 3*sec[1]:
			s = 1
		case j%3 == 2 && j < 3*sec[2]:
			s = 2
		}
		for t := range T {
			a := float32(pos[s][t]) * inv
			c, si := float32(math.Cos(float64(a))), float32(math.Sin(float64(a)))
			cs[t*rot+j], cs[t*rot+nf+j] = c, c
			sn[t*rot+j], sn[t*rot+nf+j] = si, si
		}
	}
	c, err := tensor.New(cs, []int{T, rot}, dev)
	if err != nil {
		return nil, nil, err
	}
	s, err := tensor.New(sn, []int{T, rot}, dev)
	if err != nil {
		return nil, nil, err
	}
	if act := m.Cfg.activationDType(); act != dtype.F32 {
		if c, err = ops.Cast(c, act); err != nil {
			return nil, nil, err
		}
		if s, err = ops.Cast(s, act); err != nil {
			return nil, nil, err
		}
	}
	return c, s, nil
}

// ForwardMultimodal runs one sequence (1, T) whose image placeholder tokens
// (Cfg.ImageTokenID) are replaced, in order, by the rows of image (the
// vision tower's merged tokens, (N, HiddenSize)), with three-stream MRoPE
// positions pos (Qwen35RopeIndex, offset for a cache prefix). image may be
// nil for a text-only continuation that still needs pos (decode after an
// image: every stream = cache length + delta).
func (m *Qwen35) ForwardMultimodal(tokens, image *tensor.Tensor, pos [3][]int64, cache *KVCache) (*tensor.Tensor, error) {
	if tokens.Rank() != 2 || tokens.Dim(0) != 1 {
		return nil, fmt.Errorf("Qwen35.ForwardMultimodal: tokens must be (1, T), got %v", tokens.Shape())
	}
	T := tokens.Dim(1)
	if len(pos[0]) != T || len(pos[1]) != T || len(pos[2]) != T {
		return nil, fmt.Errorf("Qwen35.ForwardMultimodal: %d positions for %d tokens", len(pos[0]), T)
	}
	dev := tokens.Device()
	prior := 0
	if cache != nil {
		prior = cache.Len
	}
	h, err := m.TokenEmb.Forward(tokens)
	if err != nil {
		return nil, err
	}
	if image != nil {
		ids, err := tokens.To(device.CPU())
		if err != nil {
			return nil, err
		}
		if h, err = scatterImage(h, tensor.MustSlice[int64](ids), m.Cfg.ImageTokenID, image); err != nil {
			return nil, err
		}
	}
	c, s, err := m.mropeTables(pos, dev)
	if err != nil {
		return nil, err
	}
	m.mrope = &[2]*tensor.Tensor{c, s}
	defer func() { m.mrope = nil }()
	return m.forwardHidden(h, 1, T, prior, dev, cache)
}

// ScatterImage replaces h's rows (1, T, D) at the placeholder positions
// (ids == tok) with image's rows, in order.
func ScatterImage(h *tensor.Tensor, ids []int64, tok int64, image *tensor.Tensor) (*tensor.Tensor, error) {
	return scatterImage(h, ids, tok, image)
}

func scatterImage(h *tensor.Tensor, ids []int64, tok int64, image *tensor.Tensor) (*tensor.Tensor, error) {
	T, D := h.Dim(1), h.Dim(2)
	if image.Dim(1) != D {
		return nil, fmt.Errorf("qwen3.5: image tokens are %d wide, the LM %d", image.Dim(1), D)
	}
	if image.DType() != h.DType() {
		var err error
		if image, err = ops.Cast(image, h.DType()); err != nil {
			return nil, err
		}
	}
	var parts []*tensor.Tensor
	next := 0
	for i := 0; i < T; {
		j := i
		img := ids[i] == tok
		for j < T && (ids[j] == tok) == img {
			j++
		}
		var p *tensor.Tensor
		var err error
		if img {
			if next+(j-i) > image.Dim(0) {
				return nil, fmt.Errorf("qwen3.5: %d+ placeholder tokens, %d image tokens", next+(j-i), image.Dim(0))
			}
			if p, err = image.Narrow(0, next, j-i); err == nil {
				if p, err = p.Contiguous(); err == nil {
					p, err = p.Reshape(1, j-i, D)
				}
			}
			next += j - i
		} else {
			p, err = h.Narrow(1, i, j-i)
		}
		if err != nil {
			return nil, err
		}
		parts = append(parts, p)
		i = j
	}
	if next != image.Dim(0) {
		return nil, fmt.Errorf("qwen3.5: %d placeholder tokens, %d image tokens", next, image.Dim(0))
	}
	if len(parts) == 1 {
		return parts[0].Contiguous()
	}
	return ops.Concat(parts, 1)
}
