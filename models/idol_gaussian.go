package models

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// IdolSampleAnchors is OctreeProbabilityFixedlenDecoder.sample for one
// latent (cond (1, Lc, 16)): starting from the unit cube holding all
// numPoints points, each level asks the decoder how a cell's points split
// among its 8 children (systematic resampling), keeps the children that
// received points, and at `level` places each point uniformly inside its
// leaf cell. rnd draws uniform [0, 1) numbers (the resampling offsets and
// the in-cell jitter). Returns numPoints × 3 coordinates in [0, 1].
func IdolSampleAnchors(dec *IdolOctreeDecoder, cond *tensor.Tensor, numPoints, level int, rnd func() float32) ([]float32, error) {
	type cell struct{ x, y, z, n int }
	cells := []cell{{0, 0, 0, numPoints}}
	dev := cond.Device()
	for l := 1; l <= level; l++ {
		resP := float32(int(1) << (l - 1))
		centres := make([]float32, 3*len(cells))
		for i, c := range cells {
			centres[3*i] = (float32(c.x) + 0.5) / resP
			centres[3*i+1] = (float32(c.y) + 0.5) / resP
			centres[3*i+2] = (float32(c.z) + 0.5) / resP
		}
		x, err := tensor.New(centres, []int{1, len(cells), 3}, dev)
		if err != nil {
			return nil, err
		}
		logits, err := dec.Forward(x, 1<<l, cond)
		if err != nil {
			return nil, err
		}
		probs, err := ops.Softmax(logits, -1)
		if err != nil {
			return nil, err
		}
		if probs, err = probs.To(device.CPU()); err != nil {
			return nil, err
		}
		p := tensor.MustSlice[float32](probs)
		var next []cell
		for i, c := range cells {
			counts := idolSystematic(p[8*i:8*i+8], c.n, rnd)
			// Children in idol's order: x fastest, then y, then z.
			for ch, n := range counts {
				if n > 0 {
					next = append(next, cell{2*c.x + ch&1, 2*c.y + ch>>1&1, 2*c.z + ch>>2&1, n})
				}
			}
		}
		cells = next
	}
	res := float32(int(1) << level)
	out := make([]float32, 0, 3*numPoints)
	for _, c := range cells {
		for range c.n {
			out = append(out, (float32(c.x)+rnd())/res, (float32(c.y)+rnd())/res, (float32(c.z)+rnd())/res)
		}
	}
	if len(out) != 3*numPoints {
		return nil, fmt.Errorf("idol anchors: placed %d points, want %d", len(out)/3, numPoints)
	}
	return out, nil
}

// idolSystematic is systematic_sample for one row: splits n items among
// the bins of probs by stratified offsets u0 + i/n, u0 uniform in [0, 1/n).
func idolSystematic(probs []float32, n int, rnd func() float32) []int {
	P := len(probs)
	out := make([]int, P)
	if n == 0 {
		return out
	}
	p := make([]float32, P)
	var sum float32
	for i, v := range probs {
		p[i] = max(v, 0)
		sum += p[i]
	}
	if sum == 0 {
		for i := range p {
			p[i] = 1 / float32(P)
		}
	} else {
		d := max(sum, 1) // idol divides by the row sum clamped to at least 1
		for i := range p {
			p[i] /= d
		}
	}
	cdf := make([]float32, P)
	var acc float32
	for i, v := range p {
		acc += v
		cdf[i] = min(acc, 1) // clamp(max=1 - 1e-12) is 1 in F32
	}
	u0 := rnd() / float32(n)
	for i := range n {
		u := min(u0+float32(i)/float32(n), 1)
		idx := sort.Search(P, func(j int) bool { return cdf[j] >= u })
		out[min(idx, P-1)]++
	}
	return out
}

// IdolGaussians are activated 3D Gaussians (idol.gaussian.Gaussian): N
// splats in the z-up object frame inside [-0.5, 0.5]^3.
type IdolGaussians struct {
	XYZ        []float32 // N × 3
	FeaturesDC []float32 // N × 3, degree-0 spherical harmonics
	Scaling    []float32 // N × 3
	Rotation   []float32 // N × 4, unit quaternions (w, x, y, z)
	Opacity    []float32 // N
}

// N is the number of Gaussians.
func (g *IdolGaussians) N() int { return len(g.Opacity) }

// ToGaussians is GaussianFixedlenDecoder.to_representation for one object:
// anchors (L × 3) and the decoder's outputs (L × G·15) to activated
// Gaussians (L·G).
func (m *IdolGaussianDecoder) ToGaussians(anchors []float32, features *tensor.Tensor) (*IdolGaussians, error) {
	f, err := features.To(device.CPU())
	if err != nil {
		return nil, err
	}
	if f.DType() != dtype.F32 {
		if f, err = ops.Cast(f, dtype.F32); err != nil {
			return nil, err
		}
	}
	fv := tensor.MustSlice[float32](f)
	rc := m.Args.Representation
	G := rc.NumGaussians
	L := len(anchors) / 3
	W := 15 * G
	if len(fv) != L*W {
		return nil, fmt.Errorf("idol gaussians: %d features for %d anchors × %d", len(fv), L, W)
	}
	lr := func(k string) float64 {
		if v, ok := rc.LR[k]; ok {
			return v
		}
		return 1
	}
	softplus := func(x float64) float64 {
		if x > 20 {
			return x
		}
		return math.Log1p(math.Exp(x))
	}
	scaleOff := rc.ScalingBias + math.Log(-math.Expm1(-rc.ScalingBias)) // inverse softplus
	opOff := math.Log(rc.OpacityBias / (1 - rc.OpacityBias))            // inverse sigmoid
	k2 := rc.FilterKernelSize * rc.FilterKernelSize
	N := L * G
	g := &IdolGaussians{XYZ: make([]float32, 3*N), FeaturesDC: make([]float32, 3*N), Scaling: make([]float32, 3*N),
		Rotation: make([]float32, 4*N), Opacity: make([]float32, N)}
	for a := range L {
		row := fv[a*W : (a+1)*W]
		for j := range G {
			i := a*G + j
			os := softplus(float64(row[14*G+j]) + float64(m.BaseOffset))
			for k := range 3 {
				off := math.Tanh(float64(row[3*j+k])*lr("_xyz")+float64(m.Perturbation[3*j+k])) * 0.5 * rc.PerturbeSize * os
				g.XYZ[3*i+k] = float32(off + float64(anchors[3*a+k]) - 0.5)
				g.FeaturesDC[3*i+k] = float32(float64(row[3*G+3*j+k]) * lr("_features_dc"))
				s := softplus(float64(row[6*G+3*j+k])*lr("_scaling") + scaleOff)
				g.Scaling[3*i+k] = float32(math.Sqrt(s*s + k2))
			}
			var q [4]float64
			var qn float64
			for k := range 4 {
				q[k] = float64(row[9*G+4*j+k]) * lr("_rotation")
				if k == 0 {
					q[k]++
				}
				qn += q[k] * q[k]
			}
			qn = max(math.Sqrt(qn), 1e-12)
			for k := range 4 {
				g.Rotation[4*i+k] = float32(q[k] / qn)
			}
			g.Opacity[i] = float32(1 / (1 + math.Exp(-(float64(row[13*G+j])*lr("_opacity") + opOff))))
		}
	}
	return g, nil
}

// WritePLY writes the standard 3D Gaussian Splatting PLY (binary little
// endian), as idol's Gaussian.save_ply: positions turned from z-up to the
// y-up convention viewers use, opacities and scales stored as their
// pre-activations (logit, log), clamped so nothing is infinite.
func (g *IdolGaussians) WritePLY(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	names := []string{"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
		"scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
	fmt.Fprintf(w, "ply\nformat binary_little_endian 1.0\nelement vertex %d\n", g.N())
	for _, n := range names {
		fmt.Fprintf(w, "property float %s\n", n)
	}
	fmt.Fprint(w, "end_header\n")
	const eps = 1e-6
	s := math.Sqrt(0.5) // +90° about x: (w, x, y, z) = (√½, √½, 0, 0)
	row := make([]float32, len(names))
	buf := make([]byte, 4*len(names))
	for i := range g.N() {
		x, y, z := g.XYZ[3*i], g.XYZ[3*i+1], g.XYZ[3*i+2]
		op := math.Min(math.Max(float64(g.Opacity[i]), eps), 1-eps)
		bw, bx, by, bz := float64(g.Rotation[4*i]), float64(g.Rotation[4*i+1]), float64(g.Rotation[4*i+2]), float64(g.Rotation[4*i+3])
		row = append(row[:0], x, -z, y, 0, 0, 0, g.FeaturesDC[3*i], g.FeaturesDC[3*i+1], g.FeaturesDC[3*i+2],
			float32(math.Log(op/(1-op))),
			float32(math.Log(math.Max(float64(g.Scaling[3*i]), eps))),
			float32(math.Log(math.Max(float64(g.Scaling[3*i+1]), eps))),
			float32(math.Log(math.Max(float64(g.Scaling[3*i+2]), eps))),
			float32(s*bw-s*bx), float32(s*bx+s*bw), float32(s*by-s*bz), float32(s*bz+s*by))
		for k, v := range row {
			binary.LittleEndian.PutUint32(buf[4*k:], math.Float32bits(v))
		}
		if _, err := w.Write(buf); err != nil {
			f.Close()
			return err
		}
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
