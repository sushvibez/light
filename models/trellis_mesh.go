package models

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/mesh"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// TRELLIS's SLat mesh decoder (SLatMeshDecoder): the Gaussian decoder's
// swin transformer torso on the 64³ voxels, then two
// SparseSubdivideBlock3d (each voxel split into its 8 children, 3³ convs
// with 32-group norms; channels C → C/4 → C/8) to 256³ cubes, and a linear
// to 101 values per cube: the SDF, deformation and colour of its 8
// corners and FlexiCubes' 21 weights. SparseFeatures2Mesh turns those into
// a triangle mesh (trellis_flexicubes.go).

// TrellisSLatMeshDecoderArgs are SLatMeshDecoder's constructor arguments.
type TrellisSLatMeshDecoderArgs struct {
	Resolution      int     `json:"resolution"`
	ModelChannels   int     `json:"model_channels"`
	LatentChannels  int     `json:"latent_channels"`
	NumBlocks       int     `json:"num_blocks"`
	NumHeads        int     `json:"num_heads"`
	NumHeadChannels int     `json:"num_head_channels"`
	MLPRatio        float64 `json:"mlp_ratio"`
	AttnMode        string  `json:"attn_mode"`
	WindowSize      int     `json:"window_size"`
	PEMode          string  `json:"pe_mode"`
	QKRMSNorm       bool    `json:"qk_rms_norm"`
	UseFP16         bool    `json:"use_fp16"`
	Rep             struct {
		UseColor bool `json:"use_color"`
	} `json:"representation_config"`
}

// trellisSubdivide is SparseSubdivideBlock3d: GroupNorm32, SiLU,
// subdivide, conv, GroupNorm32, SiLU, conv, plus the subdivided input
// (through a 1³ conv when the width changes).
type trellisSubdivide struct {
	Norm1, Norm2 idolNorm // GroupNorm affine
	Conv1, Conv2 *trellisSparseConv
	Skip         *nn.Linear // nil: identity
}

const trellisGroups = 32 // SparseSubdivideBlock3d's num_groups

func (b *trellisSubdivide) forward(o *iop, g *TrellisGrid, x *tensor.Tensor) (*tensor.Tensor, *TrellisGrid) {
	if o.err != nil {
		return nil, nil
	}
	s, err := g.Subdivide()
	if err != nil {
		o.err = err
		return nil, nil
	}
	h := o.gather(o.un(ops.SiLU, trellisGroupNorm(o, x, b.Norm1)), g.subParent)
	skip := x
	if b.Skip != nil {
		skip = o.lin(b.Skip, x) // per row, so before the copy to the children
	}
	skip = o.gather(skip, g.subParent)
	h = b.Conv1.forward(o, s, h)
	h = b.Conv2.forward(o, s, o.un(ops.SiLU, trellisGroupNorm(o, h, b.Norm2)))
	return o.add(h, skip), s
}

// trellisGroupNorm is SparseGroupNorm32(32, C) on the rows x (N, C) of one
// object: GroupNorm over the object's voxels as one (1, C, N) sample.
func trellisGroupNorm(o *iop, x *tensor.Tensor, n idolNorm) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	N, C := x.Dim(0), x.Dim(1)
	t := o.reshape(o.perm(x, 1, 0), 1, C, N, 1)
	y := o.do(func() (*tensor.Tensor, error) { return ops.GroupNorm(t, n.W, n.B, trellisGroups, 1e-5) })
	return o.reshape(o.perm(o.reshape(y, C, N), 1, 0), N, C)
}

// Subdivide returns SparseSubdivide's grid: each voxel p split into 2p +
// (i, j, k), i, j, k ∈ {0, 1} in that (lexicographic) order, voxel-major.
func (g *TrellisGrid) Subdivide() (*TrellisGrid, error) {
	if g.sub != nil {
		return g.sub, nil
	}
	N := g.N()
	c := make([][3]int32, 0, 8*N)
	par := make([]int64, 0, 8*N)
	for i, v := range g.Coords {
		for k := range 8 {
			c = append(c, [3]int32{2*v[0] + int32(k>>2), 2*v[1] + int32(k>>1&1), 2*v[2] + int32(k&1)})
			par = append(par, int64(i))
		}
	}
	s, err := newTrellisGrid(c, g.dev)
	if err != nil {
		return nil, err
	}
	if g.subParent, err = g.i64(par); err != nil {
		return nil, err
	}
	g.sub = s
	return s, nil
}

// TrellisSLatMeshDecoder is SLatMeshDecoder.
type TrellisSLatMeshDecoder struct {
	Args TrellisSLatMeshDecoderArgs
	trellisSwinTorso
	Upsample [2]*trellisSubdivide
	Out      *nn.Linear
}

// LoadTrellisSLatMeshDecoder loads ckpts/<name> (base path without
// extension) onto dev. Under nn.WithWeightStorage16 the linears and the
// sparse convs' per-offset linears keep 16-bit weights.
func LoadTrellisSLatMeshDecoder(base string, dev device.Device) (*TrellisSLatMeshDecoder, error) {
	a := TrellisSLatMeshDecoderArgs{NumHeadChannels: 64, MLPRatio: 4, AttnMode: "swin", WindowSize: 8, PEMode: "ape"}
	if err := trellisConfig(base+".json", "SLatMeshDecoder", &a); err != nil {
		return nil, err
	}
	if a.NumHeads == 0 {
		a.NumHeads = a.ModelChannels / a.NumHeadChannels
	}
	if a.AttnMode != "swin" {
		return nil, fmt.Errorf("trellis mesh decoder: attn_mode %q (only swin is supported)", a.AttnMode)
	}
	if a.PEMode != "ape" {
		return nil, fmt.Errorf("trellis mesh decoder: pe_mode %q (only ape is supported)", a.PEMode)
	}
	if a.ModelChannels%(8*trellisGroups) != 0 {
		return nil, fmt.Errorf("trellis mesh decoder: %d channels (the 32-group norms need a multiple of 256)", a.ModelChannels)
	}
	l, err := newIdolLoader(base + ".safetensors")
	if err != nil {
		return nil, err
	}
	l.dev = dev
	torso, err := l.swinTorso(a.ModelChannels, a.NumHeads, a.NumBlocks, a.WindowSize, a.QKRMSNorm, dev)
	if err != nil {
		return nil, err
	}
	m := &TrellisSLatMeshDecoder{Args: a, trellisSwinTorso: torso, Out: l.linear("out_layer")}
	in := a.ModelChannels
	for i, out := range []int{a.ModelChannels / 4, a.ModelChannels / 8} {
		p := fmt.Sprintf("upsample.%d", i)
		b := &trellisSubdivide{Norm1: l.norm(p + ".act_layers.0"), Norm2: l.norm(p + ".out_layers.1"),
			Conv1: l.sparseConv(p + ".out_layers.0"), Conv2: l.sparseConv(p + ".out_layers.3")}
		if in != out {
			b.Skip = l.sparseConv1(p + ".skip_connection")
		}
		if l.err == nil && (b.Norm1.W == nil || b.Norm2.W == nil) {
			l.err = fmt.Errorf("missing %s group norms", p)
		}
		m.Upsample[i] = b
		in = out
	}
	if l.err != nil {
		return nil, fmt.Errorf("trellis mesh decoder: %w", l.err)
	}
	if w := m.Out.W.Dim(0); w != trellisMeshChannels(a.Rep.UseColor) {
		return nil, fmt.Errorf("trellis mesh decoder: out_layer has %d outputs, want %d", w, trellisMeshChannels(a.Rep.UseColor))
	}
	return m, nil
}

// sparseConv1 loads a 1³ spconv SubMConv3d (<name>.conv.weight (out, 1, 1,
// 1, in), .bias) as a linear.
func (l *idolLoader) sparseConv1(name string) *nn.Linear {
	if l.err != nil {
		return nil
	}
	w, ok := l.hf[name+".conv.weight"]
	if !ok || w.Rank() != 5 || w.Dim(1) != 1 || w.Dim(2) != 1 || w.Dim(3) != 1 {
		l.err = fmt.Errorf("missing or non-1³ sparse conv %s.conv.weight", name)
		return nil
	}
	r, err := w.Reshape(w.Dim(0), w.Dim(4))
	if err != nil {
		l.err = err
		return nil
	}
	l.hf[name+".weight"] = r
	if b, ok := l.hf[name+".conv.bias"]; ok {
		l.hf[name+".bias"] = b
	}
	return l.linear(name)
}

// MeshResolution is SparseFeatures2Mesh's grid: 4 × the latent's.
func (m *TrellisSLatMeshDecoder) MeshResolution() int { return 4 * m.Args.Resolution }

// stages runs the decoder, returning the torso output (N, C), the two
// upsampling blocks' outputs (8N, C/4) and (64N, C/8), out_layer's (64N,
// 101) and the 64N-cube grid.
func (m *TrellisSLatMeshDecoder) stages(g *TrellisGrid, x *tensor.Tensor) (torso, up1, up2, feats *tensor.Tensor, fine *TrellisGrid, err error) {
	a := m.Args
	if x.Rank() != 2 || x.Dim(0) != g.N() || x.Dim(1) != a.LatentChannels {
		return nil, nil, nil, nil, nil, fmt.Errorf("trellis mesh decoder: input %v, want (%d, %d)", x.Shape(), g.N(), a.LatentChannels)
	}
	o := &iop{}
	torso = m.trellisSwinTorso.forward(o, g, x)
	up1, g1 := m.Upsample[0].forward(o, g, torso)
	up2, g2 := m.Upsample[1].forward(o, g1, up1)
	feats = o.lin(m.Out, up2)
	return torso, up1, up2, feats, g2, o.err
}

// Forward returns out_layer's per-cube values (64N, 101 with colour) of the
// (un-normalised) structured latent x (N, latent) on the voxels of g, and
// the grid of those cubes (g subdivided twice; 4 × g's resolution).
func (m *TrellisSLatMeshDecoder) Forward(g *TrellisGrid, x *tensor.Tensor) (*tensor.Tensor, *TrellisGrid, error) {
	_, _, _, feats, fine, err := m.stages(g, x)
	return feats, fine, err
}

// Mesh is to_representation: SparseFeatures2Mesh on the decoder's outputs
// feats on the cubes of fine (Forward's grid), in TRELLIS's z-up frame
// inside [-0.5, 0.5]³.
func (m *TrellisSLatMeshDecoder) Mesh(fine *TrellisGrid, feats *tensor.Tensor) (*mesh.Mesh, error) {
	f, err := feats.To(device.CPU())
	if err != nil {
		return nil, err
	}
	if f.DType() != dtype.F32 {
		if f, err = ops.Cast(f, dtype.F32); err != nil {
			return nil, err
		}
	}
	if f, err = f.Contiguous(); err != nil {
		return nil, err
	}
	return TrellisFeaturesToMesh(fine.Coords, tensor.MustSlice[float32](f), m.MeshResolution(), m.Args.Rep.UseColor)
}
