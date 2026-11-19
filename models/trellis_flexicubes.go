package models

import (
	"fmt"
	"math"
	"sort"

	"light/mesh"
)

// TRELLIS's SparseFeatures2Mesh with FlexiCubes (MaxtirError/FlexiCubes,
// the version TRELLIS pins, training off) on the CPU. TRELLIS scatters the
// sparse cubes into a dense (res + 1)³ vertex grid (unset corners: SDF 1,
// no deformation, colour 0; unset cubes: weights 0) and runs FlexiCubes on
// all res³ cubes; only cubes whose corners change sign matter, and those
// all touch a corner of a sparse cube, so this walks the sparse cubes'
// 3³ neighbourhoods instead. Everything else is as TRELLIS computes it, in
// F32 and in its order (cubes by dense index, edges by their sorted vertex
// pair, dual vertices grouped by how many their cube emits), so the
// vertices, faces and colours come out in the same order:
//
//	per corner   mean over the cubes sharing it of sdf + (−1 / res),
//	             deform and colour; position v / res − 0.5 +
//	             tanh(deform) / (2 res); colour sigmoid
//	per cube     β = tanh(w[0:12])·0.99 + 1, α = tanh(w[12:20])·0.99 + 1,
//	             γ = sigmoid(w[20])·0.99 + 0.005; case = corner signs
//	             (sdf < 0) with check_table's C16 / C19 fix-up
//	dual vertex  per edge group of dmc_table: Σ β_e·u_e / Σ β_e with u_e
//	             the zero crossing of the edge under weights s·α
//	faces        per sign-changing edge shared by 4 cubes, the quad of
//	             their dual vertices (winding towards positive SDF), split
//	             along the diagonal with the larger γ product

// trellisMeshChannels is SparseFeatures2Mesh's feats_channels: 8 SDF, 8 × 3
// deform, 21 weights (12 β, 8 α, γ) and, with colour, 8 × 6 colour.
func trellisMeshChannels(color bool) int {
	if color {
		return 101
	}
	return 53
}

// flexiEdges is FlexiCubes' cube_edges: the corner pairs of the 12 edges
// (corner k at (k & 1, k >> 1 & 1, k >> 2 & 1)).
var flexiEdges = [24]int{0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6, 2, 0, 3, 1, 7, 5, 6, 4}

func tanh32(x float32) float32    { return float32(math.Tanh(float64(x))) }
func sigmoid32(x float32) float32 { return float32(1 / (1 + math.Exp(-float64(x)))) }

// TrellisFeaturesToMesh is SparseFeatures2Mesh(res, use_color) on the
// decoder outputs feats (N × 101, or 53 without colour) of the cubes
// coords (each in [0, res)³): the mesh in [-0.5, 0.5]³, z up, with the six
// interpolated colour channels per vertex (RGB first) when colored.
func TrellisFeaturesToMesh(coords [][3]int32, feats []float32, res int, color bool) (*mesh.Mesh, error) {
	W := trellisMeshChannels(color)
	N := len(coords)
	if len(feats) != N*W {
		return nil, fmt.Errorf("trellis mesh: %d values for %d cubes × %d", len(feats), N, W)
	}
	const offDeform, offWeights, offColor = 8, 32, 53
	A := 4 // per-corner attributes: sdf, deform (3), colour (6)
	if color {
		A = 10
	}
	rv := res + 1
	vlin := func(x, y, z int) int32 { return int32((x*rv+y)*rv + z) }
	clin := func(x, y, z int) int32 { return int32((x*res+y)*res + z) }

	// sparse_cube2verts: per-corner means over the cubes sharing it.
	vslot := make([]int32, rv*rv*rv)
	for i := range vslot {
		vslot[i] = -1
	}
	cslot := make([]int32, res*res*res)
	for i := range cslot {
		cslot[i] = -1
	}
	sdfBias := float32(-1.0 / float64(res))
	var vsum []float32
	var vcnt []int32
	for i, c := range coords {
		x, y, z := int(c[0]), int(c[1]), int(c[2])
		if min(x, y, z) < 0 || max(x, y, z) >= res {
			return nil, fmt.Errorf("trellis mesh: cube %v outside [0, %d)³", c, res)
		}
		if cslot[clin(x, y, z)] >= 0 {
			return nil, fmt.Errorf("trellis mesh: cube %v twice", c)
		}
		cslot[clin(x, y, z)] = int32(i)
		row := feats[i*W : (i+1)*W]
		for k := range 8 {
			v := vlin(x+k&1, y+k>>1&1, z+k>>2&1)
			s := vslot[v]
			if s < 0 {
				s = int32(len(vcnt))
				vslot[v] = s
				vsum = append(vsum, make([]float32, A)...)
				vcnt = append(vcnt, 0)
			}
			a := vsum[int(s)*A : int(s+1)*A]
			a[0] += row[k] + sdfBias
			for j := range 3 {
				a[1+j] += row[offDeform+3*k+j]
			}
			if color {
				for j := range 6 {
					a[4+j] += row[offColor+6*k+j]
				}
			}
			vcnt[s]++
		}
	}
	for s, n := range vcnt {
		for j := range A {
			vsum[s*A+j] /= float32(n)
		}
	}
	// The dense grid's values at a corner: unset corners are outside.
	sdf := func(v int32) float32 {
		if s := vslot[v]; s >= 0 {
			return vsum[int(s)*A]
		}
		return 1
	}
	scale := float32((1 - 1e-8) / float64(2*res))
	pos := func(v int32) [3]float32 {
		p := [3]int{int(v) / (rv * rv), int(v) / rv % rv, int(v) % rv}
		var out [3]float32
		s := vslot[v]
		for k := range 3 {
			var d float32
			if s >= 0 {
				d = vsum[int(s)*A+1+k]
			}
			out[k] = float32(p[k])/float32(res) - 0.5 + scale*tanh32(d)
		}
		return out
	}
	col := func(v int32) [6]float32 {
		var out [6]float32
		s := vslot[v]
		for k := range 6 {
			var c float32
			if s >= 0 {
				c = vsum[int(s)*A+4+k]
			}
			out[k] = sigmoid32(c)
		}
		return out
	}

	// _identify_surf_cubes over the sparse cubes' neighbourhoods, in dense order.
	seen := make([]bool, res*res*res)
	var surf []int32
	for _, c := range coords {
		for d := range 27 {
			x, y, z := int(c[0])+d/9-1, int(c[1])+d/3%3-1, int(c[2])+d%3-1
			if min(x, y, z) < 0 || max(x, y, z) >= res || seen[clin(x, y, z)] {
				continue
			}
			seen[clin(x, y, z)] = true
			n := 0
			for k := range 8 {
				if sdf(vlin(x+k&1, y+k>>1&1, z+k>>2&1)) < 0 {
					n++
				}
			}
			if n > 0 && n < 8 {
				surf = append(surf, clin(x, y, z))
			}
		}
	}
	seen = nil
	if len(surf) == 0 {
		return &mesh.Mesh{}, nil
	}
	sort.Slice(surf, func(a, b int) bool { return surf[a] < surf[b] })
	S := len(surf)
	cubeXYZ := func(q int32) (int, int, int) { return int(q) / (res * res), int(q) / res % res, int(q) % res }
	corners := make([][8]int32, S)
	cases := make([]int, S)
	for si, q := range surf {
		x, y, z := cubeXYZ(q)
		for k := range 8 {
			v := vlin(x+k&1, y+k>>1&1, z+k>>2&1)
			corners[si][k] = v
			if sdf(v) < 0 {
				cases[si] |= 1 << k
			}
		}
	}
	// _get_case_id: an ambiguous cube (C16 / C19) whose neighbour across
	// the ambiguous face is ambiguous too takes the inverted case.
	problem := map[int32]bool{}
	for si, q := range surf {
		if _, ok := flexiCheck[cases[si]]; ok {
			problem[q] = true
		}
	}
	fixed := append([]int(nil), cases...)
	for si, q := range surf {
		ck, ok := flexiCheck[cases[si]]
		if !ok {
			continue
		}
		x, y, z := cubeXYZ(q)
		x, y, z = x+ck[0], y+ck[1], z+ck[2]
		if min(x, y, z) >= 0 && max(x, y, z) < res && problem[clin(x, y, z)] {
			fixed[si] = ck[3]
		}
	}
	cases = fixed

	// _normalize_weights for the surface cubes (unset cubes: weights 0).
	const ws = float32(0.99)
	gbias := float32((1 - 0.99) / 2.0)
	beta := make([][12]float32, S)
	alpha := make([][8]float32, S)
	gamma := make([]float32, S)
	for si, q := range surf {
		var w []float32
		if i := cslot[q]; i >= 0 {
			w = feats[int(i)*W+offWeights : int(i)*W+offWeights+21]
		} else {
			w = make([]float32, 21)
		}
		for e := range 12 {
			beta[si][e] = tanh32(w[e])*ws + 1
		}
		for k := range 8 {
			alpha[si][k] = tanh32(w[12+k])*ws + 1
		}
		gamma[si] = sigmoid32(w[20])*ws + gbias
	}

	// _identify_surf_edges: sign-changing edges, indexed in the order of
	// their sorted (vertex, vertex) pairs, with the cube edges sharing each.
	type edgeRec struct {
		a, b int32
		n    int
		occ  [4]int32 // cube·12 + edge, in cube order
	}
	var recs []edgeRec
	recOf := map[int64]int32{}
	edgeOf := make([]int32, 12*S)
	nv := int64(rv) * int64(rv) * int64(rv)
	for si := range surf {
		for e := range 12 {
			va, vb := corners[si][flexiEdges[2*e]], corners[si][flexiEdges[2*e+1]]
			if (sdf(va) < 0) == (sdf(vb) < 0) {
				edgeOf[12*si+e] = -1
				continue
			}
			key := int64(va)*nv + int64(vb)
			r, ok := recOf[key]
			if !ok {
				r = int32(len(recs))
				recOf[key] = r
				recs = append(recs, edgeRec{a: va, b: vb})
			}
			if recs[r].n < 4 {
				recs[r].occ[recs[r].n] = int32(12*si + e)
			}
			recs[r].n++
			edgeOf[12*si+e] = r
		}
	}
	recOf = nil
	byKey := make([]int32, len(recs))
	for i := range byKey {
		byKey[i] = int32(i)
	}
	key := func(r int32) int64 { return int64(recs[r].a)*nv + int64(recs[r].b) }
	sort.Slice(byKey, func(i, j int) bool { return key(byKey[i]) < key(byKey[j]) })

	// _compute_vd: dual vertices, cubes grouped by their dual-vertex count.
	nVD := 0
	for _, c := range cases {
		nVD += len(flexiDMC[c])
	}
	verts := make([]float32, 0, 3*nVD)
	var colors []float32
	if color {
		colors = make([]float32, 0, 6*nVD)
	}
	vdGamma := make([]float32, 0, nVD)
	vdOf := make([]int32, 12*S) // vd_idx_map
	for num := 1; num <= 4; num++ {
		for si := range surf {
			groups := flexiDMC[cases[si]]
			if len(groups) != num {
				continue
			}
			for _, grp := range groups {
				vd := int32(len(vdGamma))
				var sx [3]float32
				var sc [6]float32
				var bs float32
				for _, ch := range grp {
					e := int(ch - '0')
					if ch >= 'a' {
						e = int(ch-'a') + 10
					}
					r := edgeOf[12*si+e]
					if r < 0 {
						// TRELLIS would index its edge list at -1 (the
						// last edge); dmc_table never names an edge
						// without a sign change.
						r = byKey[len(byKey)-1]
					}
					va, vb := recs[r].a, recs[r].b
					w0 := sdf(va) * alpha[si][flexiEdges[2*e]]
					w1 := sdf(vb) * alpha[si][flexiEdges[2*e+1]]
					den := w1 + (-w0)
					b := beta[si][e]
					pa, pb := pos(va), pos(vb)
					for k := range 3 {
						sx[k] += (pa[k]*w1 + pb[k]*(-w0)) / den * b
					}
					if color {
						ca, cb := col(va), col(vb)
						for k := range 6 {
							sc[k] += (ca[k]*w1 + cb[k]*(-w0)) / den * b
						}
					}
					bs += b
					vdOf[12*si+e] = vd
				}
				for k := range 3 {
					verts = append(verts, sx[k]/bs)
				}
				if color {
					for k := range 6 {
						colors = append(colors, sc[k]/bs)
					}
				}
				vdGamma = append(vdGamma, gamma[si])
			}
		}
	}

	// _triangulate: quads of the edges shared by 4 cubes, the ones whose
	// first vertex is outside first, then split by γ.
	var flipped, kept [][4]int32
	for _, r := range byKey {
		rec := &recs[r]
		if rec.n != 4 {
			continue
		}
		var q [4]int32
		for k := range 4 {
			q[k] = vdOf[rec.occ[k]]
		}
		if sdf(rec.a) > 0 {
			flipped = append(flipped, [4]int32{q[0], q[1], q[3], q[2]})
		} else {
			kept = append(kept, [4]int32{q[2], q[3], q[1], q[0]})
		}
	}
	faces := make([]int32, 0, 6*(len(flipped)+len(kept)))
	for _, q := range append(flipped, kept...) {
		if vdGamma[q[0]]*vdGamma[q[2]] > vdGamma[q[1]]*vdGamma[q[3]] {
			faces = append(faces, q[0], q[1], q[2], q[0], q[2], q[3])
		} else {
			faces = append(faces, q[0], q[1], q[3], q[3], q[1], q[2])
		}
	}
	m := &mesh.Mesh{Vertices: verts, Faces: faces}
	if color {
		m.Colors, m.ColorChannels = colors, 6
	}
	return m, nil
}
