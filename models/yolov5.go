package models

import (
	"fmt"

	"light/autograd"
	"light/device"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// YOLOv5 implements the Ultralytics YOLOv5 architecture (version 6.0+):
// CSPDarknet backbone (Conv / C3 / SPPF) + PAN-FPN neck + Detect head.
//
// Variants are picked via depth-mult (d) and width-mult (w):
//
//	YOLOv5n  d=0.33 w=0.25  (~1.9M params)
//	YOLOv5s  d=0.33 w=0.50  (~7.2M)
//	YOLOv5m  d=0.67 w=0.75  (~21.2M)
//	YOLOv5l  d=1.00 w=1.00  (~46.5M)
//	YOLOv5x  d=1.33 w=1.25  (~86.7M)
//
// `Forward(x)` returns the three multi-scale feature tensors from the Detect
// head — strides 8 / 16 / 32. Each output has shape (B, NumAnchors·(5+NC),
// H', W') and still needs anchor decoding + NMS to produce final boxes.
type YOLOv5 struct {
	// Backbone
	Stem  *yoloConv
	Down1 *yoloConv
	C3_2  *yoloC3
	Down2 *yoloConv
	C3_4  *yoloC3
	Down3 *yoloConv
	C3_6  *yoloC3
	Down4 *yoloConv
	C3_8  *yoloC3
	SPPF  *yoloSPPF

	// PANet neck
	Lat10  *yoloConv
	C3_13  *yoloC3
	Lat14  *yoloConv
	C3_17  *yoloC3
	Down18 *yoloConv
	C3_20  *yoloC3
	Down21 *yoloConv
	C3_23  *yoloC3

	// Detect head
	Det8  *nn.Conv2d
	Det16 *nn.Conv2d
	Det32 *nn.Conv2d

	NumClasses int
	NumAnchors int
	Strides    [3]int
	training   bool
}

// YOLOv5Config picks a variant.
type YOLOv5Config struct {
	NumClasses int
	DepthMult  float64
	WidthMult  float64
}

// YOLOv5n returns the nano config (1.9M params).
func YOLOv5nConfig() YOLOv5Config {
	return YOLOv5Config{NumClasses: 80, DepthMult: 0.33, WidthMult: 0.25}
}
func YOLOv5sConfig() YOLOv5Config {
	return YOLOv5Config{NumClasses: 80, DepthMult: 0.33, WidthMult: 0.50}
}
func YOLOv5mConfig() YOLOv5Config {
	return YOLOv5Config{NumClasses: 80, DepthMult: 0.67, WidthMult: 0.75}
}
func YOLOv5lConfig() YOLOv5Config {
	return YOLOv5Config{NumClasses: 80, DepthMult: 1.00, WidthMult: 1.00}
}
func YOLOv5xConfig() YOLOv5Config {
	return YOLOv5Config{NumClasses: 80, DepthMult: 1.33, WidthMult: 1.25}
}

// NewYOLOv5 builds the network. Use one of the *Config helpers above.
func NewYOLOv5(cfg YOLOv5Config, dev device.Device) (*YOLOv5, error) {
	if cfg.NumClasses <= 0 {
		return nil, fmt.Errorf("YOLOv5: NumClasses must be positive")
	}
	w := cfg.WidthMult
	d := cfg.DepthMult
	roundCh := func(c int) int {
		v := int(float64(c)*w + 0.5)
		v = ((v + 7) / 8) * 8
		if v < 8 {
			v = 8
		}
		return v
	}
	roundN := func(n int) int {
		v := int(float64(n)*d + 0.5)
		if v < 1 {
			v = 1
		}
		return v
	}

	c1 := roundCh(64)   // 16  (P1/2)
	c2 := roundCh(128)  // 32  (P2/4)
	c3 := roundCh(256)  // 64  (P3/8)
	c4 := roundCh(512)  // 128 (P4/16)
	c5 := roundCh(1024) // 256 (P5/32)

	mk := func(layer any, err error) (any, error) {
		if err != nil {
			return nil, err
		}
		return layer, nil
	}
	_ = mk

	stem, err := newYoloConv(3, c1, 6, 2, 2, dev) // k6 s2 p2
	if err != nil {
		return nil, err
	}
	down1, err := newYoloConv(c1, c2, 3, 2, 1, dev)
	if err != nil {
		return nil, err
	}
	c3a, err := newYoloC3(c2, c2, roundN(3), true, dev)
	if err != nil {
		return nil, err
	}
	down2, err := newYoloConv(c2, c3, 3, 2, 1, dev)
	if err != nil {
		return nil, err
	}
	c3b, err := newYoloC3(c3, c3, roundN(6), true, dev)
	if err != nil {
		return nil, err
	}
	down3, err := newYoloConv(c3, c4, 3, 2, 1, dev)
	if err != nil {
		return nil, err
	}
	c3c, err := newYoloC3(c4, c4, roundN(9), true, dev)
	if err != nil {
		return nil, err
	}
	down4, err := newYoloConv(c4, c5, 3, 2, 1, dev)
	if err != nil {
		return nil, err
	}
	c3d, err := newYoloC3(c5, c5, roundN(3), true, dev)
	if err != nil {
		return nil, err
	}
	sppf, err := newYoloSPPF(c5, c5, 5, dev)
	if err != nil {
		return nil, err
	}

	// PANet
	lat10, err := newYoloConv(c5, c4, 1, 1, 0, dev)
	if err != nil {
		return nil, err
	}
	c3_13, err := newYoloC3(c5, c4, roundN(3), false, dev) // 256→128 input after concat
	if err != nil {
		return nil, err
	}
	lat14, err := newYoloConv(c4, c3, 1, 1, 0, dev)
	if err != nil {
		return nil, err
	}
	c3_17, err := newYoloC3(c4, c3, roundN(3), false, dev) // 128→64 after concat
	if err != nil {
		return nil, err
	}
	down18, err := newYoloConv(c3, c3, 3, 2, 1, dev)
	if err != nil {
		return nil, err
	}
	c3_20, err := newYoloC3(c4, c4, roundN(3), false, dev)
	if err != nil {
		return nil, err
	}
	down21, err := newYoloConv(c4, c4, 3, 2, 1, dev)
	if err != nil {
		return nil, err
	}
	c3_23, err := newYoloC3(c5, c5, roundN(3), false, dev)
	if err != nil {
		return nil, err
	}

	// Detect head: 3 anchors × (5 + nc) outputs per scale.
	numAnchors := 3
	headCh := numAnchors * (5 + cfg.NumClasses)
	det8, err := nn.NewConv2d(c3, headCh, 1, 1, [2]int{1, 1}, [2]int{0, 0}, true, dev)
	if err != nil {
		return nil, err
	}
	det16, err := nn.NewConv2d(c4, headCh, 1, 1, [2]int{1, 1}, [2]int{0, 0}, true, dev)
	if err != nil {
		return nil, err
	}
	det32, err := nn.NewConv2d(c5, headCh, 1, 1, [2]int{1, 1}, [2]int{0, 0}, true, dev)
	if err != nil {
		return nil, err
	}

	return &YOLOv5{
		Stem: stem, Down1: down1, C3_2: c3a, Down2: down2, C3_4: c3b,
		Down3: down3, C3_6: c3c, Down4: down4, C3_8: c3d, SPPF: sppf,
		Lat10: lat10, C3_13: c3_13, Lat14: lat14, C3_17: c3_17,
		Down18: down18, C3_20: c3_20, Down21: down21, C3_23: c3_23,
		Det8: det8, Det16: det16, Det32: det32,
		NumClasses: cfg.NumClasses, NumAnchors: numAnchors,
		Strides: [3]int{8, 16, 32},
	}, nil
}

// Forward returns the three Detect-head feature maps (P3/8, P4/16, P5/32).
// Each has shape (B, NumAnchors·(5+NC), H', W'); decode anchors / NMS in user code.
func (y *YOLOv5) Forward(x *tensor.Tensor) ([3]*tensor.Tensor, error) {
	var out [3]*tensor.Tensor
	h, err := y.Stem.Forward(x)
	if err != nil {
		return out, err
	}
	h, err = y.Down1.Forward(h)
	if err != nil {
		return out, err
	}
	h, err = y.C3_2.Forward(h)
	if err != nil {
		return out, err
	}
	h, err = y.Down2.Forward(h)
	if err != nil {
		return out, err
	}
	p3, err := y.C3_4.Forward(h)
	if err != nil {
		return out, err
	}
	h, err = y.Down3.Forward(p3)
	if err != nil {
		return out, err
	}
	p4, err := y.C3_6.Forward(h)
	if err != nil {
		return out, err
	}
	h, err = y.Down4.Forward(p4)
	if err != nil {
		return out, err
	}
	h, err = y.C3_8.Forward(h)
	if err != nil {
		return out, err
	}
	p5, err := y.SPPF.Forward(h)
	if err != nil {
		return out, err
	}

	// PANet
	lat10, err := y.Lat10.Forward(p5)
	if err != nil {
		return out, err
	}
	up11, err := ops.UpsampleNearest(lat10, [2]int{2, 2})
	if err != nil {
		return out, err
	}
	cat12, err := ops.Concat([]*tensor.Tensor{up11, p4}, 1)
	if err != nil {
		return out, err
	}
	c13, err := y.C3_13.Forward(cat12)
	if err != nil {
		return out, err
	}
	lat14, err := y.Lat14.Forward(c13)
	if err != nil {
		return out, err
	}
	up15, err := ops.UpsampleNearest(lat14, [2]int{2, 2})
	if err != nil {
		return out, err
	}
	cat16, err := ops.Concat([]*tensor.Tensor{up15, p3}, 1)
	if err != nil {
		return out, err
	}
	head8, err := y.C3_17.Forward(cat16) // P3 head (stride 8)
	if err != nil {
		return out, err
	}
	d18, err := y.Down18.Forward(head8)
	if err != nil {
		return out, err
	}
	cat19, err := ops.Concat([]*tensor.Tensor{d18, lat14}, 1)
	if err != nil {
		return out, err
	}
	head16, err := y.C3_20.Forward(cat19) // P4 head (stride 16)
	if err != nil {
		return out, err
	}
	d21, err := y.Down21.Forward(head16)
	if err != nil {
		return out, err
	}
	cat22, err := ops.Concat([]*tensor.Tensor{d21, lat10}, 1)
	if err != nil {
		return out, err
	}
	head32, err := y.C3_23.Forward(cat22) // P5 head (stride 32)
	if err != nil {
		return out, err
	}

	// Detect.
	o8, err := y.Det8.Forward(head8)
	if err != nil {
		return out, err
	}
	o16, err := y.Det16.Forward(head16)
	if err != nil {
		return out, err
	}
	o32, err := y.Det32.Forward(head32)
	if err != nil {
		return out, err
	}
	out[0], out[1], out[2] = o8, o16, o32
	return out, nil
}

func (y *YOLOv5) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, m := range []nn.Module{
		y.Stem, y.Down1, y.C3_2, y.Down2, y.C3_4, y.Down3, y.C3_6,
		y.Down4, y.C3_8, y.SPPF,
		y.Lat10, y.C3_13, y.Lat14, y.C3_17, y.Down18, y.C3_20, y.Down21, y.C3_23,
		y.Det8, y.Det16, y.Det32,
	} {
		ps = append(ps, m.Parameters()...)
	}
	return ps
}

func (y *YOLOv5) ParamCount() int {
	total := 0
	for _, p := range y.Parameters() {
		total += p.NumElements()
	}
	return total
}

func (y *YOLOv5) SetTraining(t bool) {
	y.training = t
	for _, m := range []interface{ SetTraining(bool) }{
		y.Stem, y.Down1, y.C3_2, y.Down2, y.C3_4, y.Down3, y.C3_6,
		y.Down4, y.C3_8, y.SPPF,
		y.Lat10, y.C3_13, y.Lat14, y.C3_17, y.Down18, y.C3_20, y.Down21, y.C3_23,
	} {
		m.SetTraining(t)
	}
}

// ---- YOLOv5 building blocks ----

// yoloConv: Conv → BN → SiLU.
type yoloConv struct {
	Conv *nn.Conv2d
	BN   *nn.BatchNorm2d
}

func newYoloConv(cin, cout, kernel, stride, padding int, dev device.Device) (*yoloConv, error) {
	conv, err := nn.NewConv2d(cin, cout, kernel, kernel,
		[2]int{stride, stride}, [2]int{padding, padding}, false, dev)
	if err != nil {
		return nil, err
	}
	bn, err := nn.NewBatchNorm2d(cout, 1e-3, 0.03, dev)
	if err != nil {
		return nil, err
	}
	return &yoloConv{Conv: conv, BN: bn}, nil
}

func (c *yoloConv) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := c.Conv.Forward(x)
	if err != nil {
		return nil, err
	}
	h, err = c.BN.Forward(h)
	if err != nil {
		return nil, err
	}
	return autograd.SiLU(h)
}
func (c *yoloConv) Parameters() []*tensor.Tensor {
	return append(append([]*tensor.Tensor{}, c.Conv.Parameters()...), c.BN.Parameters()...)
}
func (c *yoloConv) SetTraining(t bool) { c.BN.SetTraining(t) }

// yoloBottleneck: residual block with two conv layers.
type yoloBottleneck struct {
	Conv1, Conv2 *yoloConv
	Shortcut     bool
}

func newYoloBottleneck(cin, cout int, shortcut bool, dev device.Device) (*yoloBottleneck, error) {
	hid := cout
	c1, err := newYoloConv(cin, hid, 1, 1, 0, dev)
	if err != nil {
		return nil, err
	}
	c2, err := newYoloConv(hid, cout, 3, 1, 1, dev)
	if err != nil {
		return nil, err
	}
	return &yoloBottleneck{Conv1: c1, Conv2: c2, Shortcut: shortcut && cin == cout}, nil
}

func (b *yoloBottleneck) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := b.Conv1.Forward(x)
	if err != nil {
		return nil, err
	}
	h, err = b.Conv2.Forward(h)
	if err != nil {
		return nil, err
	}
	if b.Shortcut {
		return autograd.Add(h, x)
	}
	return h, nil
}
func (b *yoloBottleneck) Parameters() []*tensor.Tensor {
	return append(b.Conv1.Parameters(), b.Conv2.Parameters()...)
}
func (b *yoloBottleneck) SetTraining(t bool) { b.Conv1.SetTraining(t); b.Conv2.SetTraining(t) }

// yoloC3: CSP block — split → bottleneck stack → concat → conv.
type yoloC3 struct {
	Cv1, Cv2, Cv3 *yoloConv
	Blocks        []*yoloBottleneck
}

func newYoloC3(cin, cout, n int, shortcut bool, dev device.Device) (*yoloC3, error) {
	hid := cout / 2
	cv1, err := newYoloConv(cin, hid, 1, 1, 0, dev)
	if err != nil {
		return nil, err
	}
	cv2, err := newYoloConv(cin, hid, 1, 1, 0, dev)
	if err != nil {
		return nil, err
	}
	cv3, err := newYoloConv(2*hid, cout, 1, 1, 0, dev)
	if err != nil {
		return nil, err
	}
	blocks := make([]*yoloBottleneck, n)
	for i := range blocks {
		b, err := newYoloBottleneck(hid, hid, shortcut, dev)
		if err != nil {
			return nil, err
		}
		blocks[i] = b
	}
	return &yoloC3{Cv1: cv1, Cv2: cv2, Cv3: cv3, Blocks: blocks}, nil
}

func (c *yoloC3) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	a, err := c.Cv1.Forward(x)
	if err != nil {
		return nil, err
	}
	for _, b := range c.Blocks {
		a, err = b.Forward(a)
		if err != nil {
			return nil, err
		}
	}
	bx, err := c.Cv2.Forward(x)
	if err != nil {
		return nil, err
	}
	cat, err := ops.Concat([]*tensor.Tensor{a, bx}, 1)
	if err != nil {
		return nil, err
	}
	return c.Cv3.Forward(cat)
}
func (c *yoloC3) Parameters() []*tensor.Tensor {
	ps := append([]*tensor.Tensor{}, c.Cv1.Parameters()...)
	ps = append(ps, c.Cv2.Parameters()...)
	for _, b := range c.Blocks {
		ps = append(ps, b.Parameters()...)
	}
	ps = append(ps, c.Cv3.Parameters()...)
	return ps
}
func (c *yoloC3) SetTraining(t bool) {
	c.Cv1.SetTraining(t)
	c.Cv2.SetTraining(t)
	c.Cv3.SetTraining(t)
	for _, b := range c.Blocks {
		b.SetTraining(t)
	}
}

// yoloSPPF: Spatial Pyramid Pooling Fast — three sequential maxpools, concat.
type yoloSPPF struct {
	Cv1, Cv2 *yoloConv
	Pool     *nn.MaxPool2d
}

func newYoloSPPF(cin, cout, k int, dev device.Device) (*yoloSPPF, error) {
	hid := cin / 2
	cv1, err := newYoloConv(cin, hid, 1, 1, 0, dev)
	if err != nil {
		return nil, err
	}
	cv2, err := newYoloConv(hid*4, cout, 1, 1, 0, dev)
	if err != nil {
		return nil, err
	}
	pool := nn.NewMaxPool2d([2]int{k, k}, [2]int{1, 1}, [2]int{k / 2, k / 2})
	return &yoloSPPF{Cv1: cv1, Cv2: cv2, Pool: pool}, nil
}

func (s *yoloSPPF) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := s.Cv1.Forward(x)
	if err != nil {
		return nil, err
	}
	p1, err := s.Pool.Forward(h)
	if err != nil {
		return nil, err
	}
	p2, err := s.Pool.Forward(p1)
	if err != nil {
		return nil, err
	}
	p3, err := s.Pool.Forward(p2)
	if err != nil {
		return nil, err
	}
	cat, err := ops.Concat([]*tensor.Tensor{h, p1, p2, p3}, 1)
	if err != nil {
		return nil, err
	}
	return s.Cv2.Forward(cat)
}
func (s *yoloSPPF) Parameters() []*tensor.Tensor {
	return append(s.Cv1.Parameters(), s.Cv2.Parameters()...)
}
func (s *yoloSPPF) SetTraining(t bool) { s.Cv1.SetTraining(t); s.Cv2.SetTraining(t) }
