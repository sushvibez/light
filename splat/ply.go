package splat

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// PLYProperties are the vertex properties of a degree-0 3D Gaussian
// Splatting PLY, in file order.
var PLYProperties = []string{"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
	"scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}

// PLYRows returns the vertex rows (N × 17, PLYProperties) TRELLIS's
// Gaussian.save_ply writes for g, with its default transform T = [[1, 0,
// 0], [0, 0, −1], [0, 1, 0]] from the z-up object frame to the y-up one
// viewers use: positions T·p, zero normals, the SH DC colours, opacities as
// logits and scales as logs (both computed in F32 as torch does), and
// rotations as utils3d's matrix_to_quaternion(T · quaternion_to_matrix(q)).
// An opacity that rounds to 0 or 1 in F32 (an infinite logit in TRELLIS)
// is clamped to ±16.64 (the logit of 1 − 2⁻²⁴) so the file stays finite.
func PLYRows(g *Gaussians) []float32 {
	n := len(g.Opacity)
	const W = 17
	out := make([]float32, n*W)
	for i := range n {
		r := out[i*W : (i+1)*W]
		x, y, z := g.XYZ[3*i], g.XYZ[3*i+1], g.XYZ[3*i+2]
		r[0], r[1], r[2] = x, -z, y
		r[6], r[7], r[8] = g.FeaturesDC[3*i], g.FeaturesDC[3*i+1], g.FeaturesDC[3*i+2]
		p := g.Opacity[i]
		l := float32(math.Log(float64(p / (1 - p))))
		if math.IsInf(float64(l), 0) {
			l = float32(math.Copysign(16.635532, float64(l))) // logit(1 − 2⁻²⁴)
		}
		r[9] = l
		for k := range 3 {
			r[10+k] = float32(math.Log(float64(g.Scaling[3*i+k])))
		}
		q := plyRotation([4]float64{float64(g.Rotation[4*i]), float64(g.Rotation[4*i+1]), float64(g.Rotation[4*i+2]), float64(g.Rotation[4*i+3])})
		for k := range 4 {
			r[13+k] = float32(q[k])
		}
	}
	return out
}

// plyRotation is matrix_to_quaternion(T · quaternion_to_matrix(q)) as
// utils3d.numpy implements both (EasternJournalist/utils3d@9a4eb15e), q =
// (w, x, y, z).
func plyRotation(q [4]float64) [4]float64 {
	nq := math.Max(math.Sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]), 1e-12)
	w, x, y, z := q[0]/nq, q[1]/nq, q[2]/nq, q[3]/nq
	v := [3]float64{x, y, z}
	s2 := x*x + y*y + z*z
	B := [3][3]float64{{0, -z, y}, {z, 0, -x}, {-y, x, 0}}
	var R [3][3]float64
	for a := range 3 {
		for b := range 3 {
			A := v[a] * v[b]
			id := 0.0
			if a == b {
				A -= s2
				id = 1
			}
			R[a][b] = id + 2*(A+w*B[a][b])
		}
	}
	// T·R: rows (R0, −R2, R1).
	m := [3][3]float64{R[0], {-R[2][0], -R[2][1], -R[2][2]}, R[1]}
	diag := [3]float64{m[0][0], m[1][1], m[2][2]}
	M := [4][3]float64{{1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}}
	var wxyz [4]float64
	best := 0
	for i := range 4 {
		wxyz[i] = 0.5 * math.Sqrt(math.Max(1+diag[0]*M[i][0]+diag[1]*M[i][1]+diag[2]*M[i][2], 0))
		if wxyz[i] > wxyz[best] {
			best = i
		}
	}
	sign := func(v float64) float64 {
		switch {
		case v > 0:
			return 1
		case v < 0:
			return -1
		}
		return 0
	}
	xw, yw, zw := sign(m[2][1]-m[1][2]), sign(m[0][2]-m[2][0]), sign(m[1][0]-m[0][1])
	yz, xz, xy := sign(m[2][1]+m[1][2]), sign(m[0][2]+m[2][0]), sign(m[0][1]+m[1][0])
	var sg [4]float64
	switch best {
	case 0:
		sg = [4]float64{1, xw, yw, zw}
	case 1:
		sg = [4]float64{xw, 1, xy, xz}
	case 2:
		sg = [4]float64{yw, xy, 1, yz}
	default:
		sg = [4]float64{zw, xz, yz, 1}
	}
	var out [4]float64
	var n float64
	for i := range 4 {
		out[i] = sg[i] * wxyz[i]
		n += out[i] * out[i]
	}
	n = math.Max(math.Sqrt(n), 1e-12)
	for i := range out {
		out[i] /= n
	}
	return out
}

// WritePLY writes g as a binary little-endian 3DGS PLY with TRELLIS's
// Gaussian.save_ply conventions (PLYRows).
func WritePLY(path string, g *Gaussians) error {
	rows := PLYRows(g)
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	fmt.Fprintf(w, "ply\nformat binary_little_endian 1.0\nelement vertex %d\n", len(g.Opacity))
	for _, n := range PLYProperties {
		fmt.Fprintf(w, "property float %s\n", n)
	}
	fmt.Fprint(w, "end_header\n")
	var b [4]byte
	for _, v := range rows {
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
		if _, err := w.Write(b[:]); err != nil {
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

// ReadPLY reads a degree-0 3DGS PLY (binary little endian, float
// properties including PLYProperties) back into activated Gaussians in
// the z-up frame: the inverse of WritePLY (rotations up to sign).
func ReadPLY(path string) (*Gaussians, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	var props []string
	n := -1
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			return nil, fmt.Errorf("%s: header: %w", path, err)
		}
		var name string
		var cnt int
		switch {
		case line == "end_header\n":
		case len(line) > 15 && line[:15] == "element vertex ":
			if _, err := fmt.Sscanf(line, "element vertex %d", &cnt); err != nil {
				return nil, fmt.Errorf("%s: %q", path, line)
			}
			n = cnt
			continue
		case len(line) > 15 && line[:15] == "property float ":
			if _, err := fmt.Sscanf(line, "property float %s", &name); err != nil {
				return nil, fmt.Errorf("%s: %q", path, line)
			}
			props = append(props, name)
			continue
		case line == "format binary_little_endian 1.0\n" || line == "ply\n" || len(line) > 8 && line[:8] == "comment ":
			continue
		default:
			return nil, fmt.Errorf("%s: unsupported PLY header line %q", path, line)
		}
		break
	}
	col := map[string]int{}
	for i, p := range props {
		col[p] = i
	}
	for _, p := range PLYProperties[:3] {
		if _, ok := col[p]; !ok {
			return nil, fmt.Errorf("%s: no %s property", path, p)
		}
	}
	for _, p := range PLYProperties[6:] {
		if _, ok := col[p]; !ok {
			return nil, fmt.Errorf("%s: no %s property", path, p)
		}
	}
	if n < 0 {
		return nil, fmt.Errorf("%s: no vertex element", path)
	}
	W := len(props)
	buf := make([]byte, 4*W)
	g := &Gaussians{XYZ: make([]float32, 3*n), FeaturesDC: make([]float32, 3*n), Scaling: make([]float32, 3*n),
		Rotation: make([]float32, 4*n), Opacity: make([]float32, n)}
	s := math.Sqrt(0.5) // T⁻¹ as a quaternion: −90° about x
	for i := range n {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("%s: vertex %d: %w", path, i, err)
		}
		v := func(p string) float64 {
			return float64(math.Float32frombits(binary.LittleEndian.Uint32(buf[4*col[p]:])))
		}
		x, y, z := v("x"), v("y"), v("z")
		g.XYZ[3*i], g.XYZ[3*i+1], g.XYZ[3*i+2] = float32(x), float32(z), float32(-y)
		g.FeaturesDC[3*i], g.FeaturesDC[3*i+1], g.FeaturesDC[3*i+2] = float32(v("f_dc_0")), float32(v("f_dc_1")), float32(v("f_dc_2"))
		g.Opacity[i] = float32(1 / (1 + math.Exp(-v("opacity"))))
		for k := range 3 {
			g.Scaling[3*i+k] = float32(math.Exp(v(fmt.Sprint("scale_", k))))
		}
		qw, qx, qy, qz := v("rot_0"), v("rot_1"), v("rot_2"), v("rot_3")
		qn := math.Max(math.Sqrt(qw*qw+qx*qx+qy*qy+qz*qz), 1e-12)
		qw, qx, qy, qz = qw/qn, qx/qn, qy/qn, qz/qn
		// (s, −s, 0, 0) ⊗ q
		g.Rotation[4*i] = float32(s*qw + s*qx)
		g.Rotation[4*i+1] = float32(s*qx - s*qw)
		g.Rotation[4*i+2] = float32(s*qy + s*qz)
		g.Rotation[4*i+3] = float32(s*qz - s*qy)
	}
	return g, nil
}
